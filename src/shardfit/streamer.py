"""Streaming execution loop.

While the compute stream runs shard i, a separate copy stream loads shard i+1
into a free slot from pinned host memory (non-blocking copies, CUDA events for
synchronization). Slots rotate. Resident shards are simply executed in place.

Section 5.3 of the brief.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import torch

from shardfit.backends.base import Backend, BackendStats
from shardfit.sharder import Shard


@dataclass
class Slot:
    """A VRAM buffer that streamed shards rotate through."""

    index: int
    size_gb: float
    shard: Optional[Shard] = None
    state_dict: dict = None  # loaded weights on device
    occupied: bool = False
    loading: bool = False

    def __post_init__(self):
        if self.state_dict is None:
            self.state_dict = {}

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
    double_buffer_hits: int = 0  # times next shard was already loaded
    double_buffer_misses: int = 0  # times we had to wait for load

    @property
    def avg_transfer_gb_s(self) -> float:
        # Placeholder: would need bytes transferred
        return 0.0


class Streamer:
    """Manages rotating slots and streaming shards through VRAM.

    The streamer maintains S slots (at least 2 for double buffering).
    While shard i is being computed, shard i+1 is preloaded into a free slot.
    """

    def __init__(
        self,
        backend,
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

        # Shard queue (ordered)
        self.shard_queue: deque[Shard] = deque()
        self.current_shard_index = 0

        # Stats
        self.stats = StreamStats()

        # Resident shards (embedding, output head) — stay in VRAM permanently
        self.resident_shards: dict[int, dict] = {}

    def load_shards(self, shards: list[Shard]) -> None:
        """Load all shards into the queue. First resident shard(s) go directly
        to resident storage; the rest are queued for streaming."""
        for shard in shards:
            is_resident = any(b.block_type == "resident" for b in shard.blocks)
            if is_resident:
                # Resident shards are loaded once and stay
                self.resident_shards[shard.index] = {}
                self.stats.total_shards_loaded += 1
            else:
                self.shard_queue.append(shard)

    def get_next_shard(self) -> Optional[Shard]:
        """Get the next shard to process, loading it into a slot if needed."""
        if not self.shard_queue:
            return None

        shard = self.shard_queue[0]

        # Check if shard is already loaded in a slot (double buffer hit)
        for slot in self.slots:
            if slot.shard is shard and slot.occupied:
                self.stats.double_buffer_hits += 1
                self.shard_queue.popleft()
                return shard

        # Find a free slot
        free_slot = None
        for slot in self.slots:
            if slot.is_free():
                free_slot = slot
                break

        if free_slot is None:
            # No free slot — need to wait or evict
            # In double buffering, one slot is computing while other loads
            # If both occupied, we wait for the compute slot to free
            self.stats.double_buffer_misses += 1
            return None

        # Load shard into free slot
        self._load_shard_into_slot(shard, free_slot)
        self.shard_queue.popleft()
        self.stats.total_shards_loaded += 1

        return shard

    def _load_shard_into_slot(self, shard: Shard, slot: Slot) -> None:
        """Load a shard's weights into a slot on the backend."""
        t0 = time.perf_counter()

        # In a real implementation, shard.state_dict would contain the
        # device-resident weights. Here we load them via the backend.
        slot.state_dict = {}  # populated by engine with actual weights
        slot.shard = shard
        slot.occupied = True
        slot.loading = False

        elapsed = time.perf_counter() - t0
        self.stats.total_transfer_time_s += elapsed

    def release_slot(self, shard: Shard) -> None:
        """Release a slot after a shard has been computed."""
        for slot in self.slots:
            if slot.shard is shard:
                slot.occupied = False
                slot.shard = None
                slot.state_dict = {}
                self.stats.slot_rotations += 1
                break

    def get_resident_weights(self) -> dict:
        """Return all resident shard weights (embedding, output head)."""
        weights = {}
        for shard_idx, state_dict in self.resident_shards.items():
            weights.update(state_dict)
        return weights

    def __repr__(self) -> str:
        slots_str = " | ".join(str(s) for s in self.slots)
        return f"Streamer(slots={self.num_slots}): [{slots_str}] queue={len(self.shard_queue)}"
