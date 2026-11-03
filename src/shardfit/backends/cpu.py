"""CPU-only backend.

Runs inference on system RAM. Used when:
- No GPU is available
- The model doesn't fit in GPU VRAM (fallback)
- macOS (unified memory — no PCIe copy penalty)
"""

from __future__ import annotations

import time
from typing import Optional

import torch

from shardfit.backends.base import Backend


class CpuBackend(Backend):
    """CPU compute backend."""

    def __init__(self, num_threads: Optional[int] = None):
        super().__init__(device_name="cpu")
        if num_threads is not None:
            torch.set_num_threads(num_threads)

    def initialize(self) -> None:
        self.device = torch.device("cpu")
        self._initialized = True

    def load_shard(self, shard_state_dict: dict) -> None:
        """Load shard weights into system RAM.

        On CPU (or unified memory), this is essentially a no-op since weights
        already live in accessible memory. We record the transfer for stats.
        """
        if not self._initialized:
            self.initialize()

        t0 = time.perf_counter()
        bytes_moved = 0

        for key, tensor in shard_state_dict.items():
            if tensor is None:
                continue
            bytes_moved += tensor.element_size() * tensor.numel()
            # Ensure tensor is on CPU and contiguous
            _ = tensor.contiguous().to(device="cpu")

        elapsed = time.perf_counter() - t0
        self.record_transfer(bytes_moved, elapsed)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict] = None,
    ) -> tuple[torch.Tensor, dict]:
        """Run a forward pass on CPU."""
        if not self._initialized:
            self.initialize()

        input_ids = input_ids.to(device="cpu")
        t0 = time.perf_counter()

        # Placeholder: identity forward (real model logic in engine)
        logits = input_ids.float()
        elapsed = time.perf_counter() - t0

        self.record_forward(elapsed)

        return logits, kv_cache or {}

    def get_memory_used_mb(self) -> int:
        import psutil

        process = psutil.Process()
        return int(process.memory_info().rss / (1024**2))

    def clear_cache(self) -> None:
        pass

    def supports_cuda(self) -> bool:
        return False
