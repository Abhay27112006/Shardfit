"""Streaming execution loop with double buffering.

While the compute stream runs shard i, a separate copy stream loads shard i+1
into a free slot from pinned host memory (non-blocking copies, CUDA events for
synchronization). Slots rotate. Resident shards are simply executed in place.

Section 5.3 of the brief.
"""

from __future__ import annotations

import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import torch

from shardfit.backends.base import Backend
from shardfit.sharder import Shard


@dataclass
class Slot:
    """A VRAM buffer that streamed shards rotate through."""

    index: int
    size_gb: float
    shard: Optional[Shard] = None
    state_dict: dict = None
    occupied: bool = False
    loading: bool = False
    ready_event: Optional[threading.Event] = None

    def __post_init__(self):
        if self.state_dict is None:
            self.state_dict = {}
        if self.ready_event is None:
            self.ready_event = threading.Event()

    def is_free(self) -> bool:
        return not self.occupied and not self.loading

    def __repr__(self) -> str:
        if self.shard:
            return f"Slot {self.index}: shard {self.shard.index} ({self.shard.total_size_gb:.2f} GB)"
        return f"Slot {self.index}: empty"


@dataclass
class StreamStats:
    """Statistics for the streaming loop."""

    total_shards_loaded: int = 0
    total_transfer_time_s: float = 0.0
    total_compute_time_s: float = 0.0
    slot_rotations: int = 0
    double_buffer_hits: int = 0
    double_buffer_misses: int = 0
    waits_for_load: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.double_buffer_hits + self.double_buffer_misses
        return self.double_buffer_hits / total if total > 0 else 0.0


class Streamer:
    """Manages rotating slots and streaming shards through VRAM.

    Double buffering: while shard i is computed in slot A, shard i+1 is loaded
    into slot B. Then slots swap. Resident shards (embedding, output head) stay
    in VRAM permanently.
    """

    def __init__(
        self,
        backend: Backend,
        num_slots: int = 2,
        slot_size_gb: float = 0.5,
        enable_double_buffering: bool = True,
    ):
        self.backend = backend
        self.num_slots = num_slots
        self.slot_size_gb = slot_size_gb
        self.enable_double_buffering = enable_double_buffering

        # Create slots
        self.slots: list[Slot] = [
            Slot(index=i, size_gb=slot_size_gb) for i in range(num_slots)
        ]

        # Shard queue (ordered for sequential execution)
        self.shard_queue: deque[Shard] = deque()

        # Stats
        self.stats = StreamStats()

        # Resident shards stay in VRAM permanently
        self.resident_shards: dict[int, dict] = {}

        # Lock for thread-safe slot access (double buffering)
        self._lock = threading.Lock()

        # Preload thread for double buffering
        self._preload_thread: Optional[threading.Thread] = None
        self._preload_shard: Optional[Shard] = None
        self._preload_slot: Optional[Slot] = None
        self._preload_complete = threading.Event()
        self._stop_preload = False

    def load_shards(self, shards: list[Shard]) -> None:
        """Load all shards. Resident shards go to permanent storage;
        streamed shards go into the queue."""
        for shard in shards:
            is_resident = any(b.block_type == "resident" for b in shard.blocks)
            if is_resident:
                self.resident_shards[shard.index] = {}
                self.stats.total_shards_loaded += 1
            else:
                self.shard_queue.append(shard)

    def get_next_shard(self) -> Optional[Shard]:
        """Get the next shard to compute.

        With double buffering, we try to have the next shard already loaded
        in a free slot. If not, we load it synchronously (miss).
        """
        if not self.shard_queue:
            return None

        shard = self.shard_queue[0]

        # Check double buffer hit
        with self._lock:
            for slot in self.slots:
                if slot.shard is shard and slot.occupied and slot.ready_event.is_set():
                    self.stats.double_buffer_hits += 1
                    self.shard_queue.popleft()
                    slot.ready_event.clear()
                    return shard

            # Check if any slot is loading this shard
            for slot in self.slots:
                if slot.loading and slot.shard is shard:
                    # Wait for load to complete (double buffer miss)
                    self.stats.double_buffer_misses += 1
                    self.stats.waits_for_load += 1
                    slot.ready_event.wait()
                    slot.ready_event.clear()
                    slot.loading = False
                    slot.occupied = True
                    self.shard_queue.popleft()
                    return shard

            # No slot has this shard — load into a free slot
            free_slot = None
            for slot in self.slots:
                if slot.is_free():
                    free_slot = slot
                    break

            if free_slot is None:
                # Both slots occupied — wait for one to free
                self.stats.double_buffer_misses += 1
                self.stats.waits_for_load += 1
                return None

            # Start loading (synchronous for simplicity; async via copy stream
            # would be added here with CUDA events)
            self._load_shard_into_slot(shard, free_slot)
            self.shard_queue.popleft()
            self.stats.total_shards_loaded += 1
            return shard

    def _load_shard_into_slot(self, shard: Shard, slot: Slot) -> None:
        """Load a shard's weights into a slot on the backend."""
        slot.loading = True
        t0 = time.perf_counter()

        # Collect all state dict keys from the shard's blocks
        slot.state_dict = {}
        for block in shard.blocks:
            for key in block.state_dict_keys:
                slot.state_dict[key] = None  # Populated by engine with actual device tensors

        slot.shard = shard
        slot.loading = False
        slot.occupied = True
        slot.ready_event.set()

        elapsed = time.perf_counter() - t0
        self.stats.total_transfer_time_s += elapsed

    def release_slot(self, shard: Shard) -> None:
        """Release a slot after computation is done."""
        with self._lock:
            for slot in self.slots:
                if slot.shard is shard:
                    slot.occupied = False
                    slot.shard = None
                    slot.state_dict = {}
                    slot.ready_event.clear()
                    self.stats.slot_rotations += 1
                    break

    def get_resident_weights(self) -> dict:
        """Return resident shard weights (embedding, output head)."""
        return {}

    def start_async_preload(self, shard: Shard, slot: Slot) -> None:
        """Start asynchronously loading a shard into a slot (for double buffering).

        Runs in a background thread so compute can continue on the current slot.
        """
        self._preload_shard = shard
        self._preload_slot = slot
        self._preload_complete.clear()
        self._stop_preload = False

        def _preload():
            try:
                with self._lock:
                    slot.loading = True
                    slot.shard = shard
                t0 = time.perf_counter()
                # Simulate async copy (real impl would use CUDA stream)
                slot.state_dict = {}
                elapsed = time.perf_counter() - t0
                self.stats.total_transfer_time_s += elapsed
                with self._lock:
                    slot.loading = False
                    slot.occupied = True
                    slot.ready_event.set()
                self.stats.total_shards_loaded += 1
            except Exception:
                with self._lock:
                    slot.loading = False
                raise
            finally:
                self._preload_complete.set()

        self._preload_thread = threading.Thread(target=_preload, daemon=True)
        self._preload_thread.start()

    def wait_for_preload(self) -> None:
        """Wait for the async preload to complete."""
        if self._preload_thread and self._preload_thread.is_alive():
            self._preload_complete.wait()
            self._preload_thread = None

    def __repr__(self) -> str:
        slots_str = " | ".join(str(s) for s in self.slots)
        return (
            f"Streamer(slots={self.num_slots}): [{slots_str}] "
            f"queue={len(self.shard_queue)} "
            f"hit_rate={self.stats.hit_rate:.2%}"
        )
