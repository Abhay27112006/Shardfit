"""Abstract base class for compute backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class BackendStats:
    """Performance statistics for a backend."""

    forward_calls: int = 0
    total_forward_time_s: float = 0.0
    total_bytes_transferred: int = 0
    total_transfer_time_s: float = 0.0

    @property
    def avg_forward_ms(self) -> float:
        if self.forward_calls == 0:
            return 0.0
        return (self.total_forward_time_s / self.forward_calls) * 1000

    @property
    def avg_transfer_gb_s(self) -> float:
        if self.total_transfer_time_s == 0:
            return 0.0
        return (self.total_bytes_transferred / (1024**3)) / self.total_transfer_time_s


class Backend(ABC):
    """Abstract compute backend.

    A backend is responsible for:
    - Allocating device memory (VRAM or system RAM)
    - Loading shard weights onto the device
    - Running forward passes
    - Managing the KV cache
    """

    def __init__(self, device_name: str = "auto"):
        self.device_name = device_name
        self.device: Optional[torch.device] = None
        self.stats = BackendStats()
        self._initialized = False

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the backend (allocate device context, etc.)."""

    @abstractmethod
    def load_shard(self, shard_state_dict: dict) -> None:
        """Load a shard's weights onto the device."""

    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict] = None,
    ) -> tuple[torch.Tensor, dict]:
        """Run a forward pass.

        Returns:
            (logits, updated_kv_cache)
        """

    @abstractmethod
    def get_memory_used_mb(self) -> int:
        """Return current memory usage in MB."""

    @abstractmethod
    def clear_cache(self) -> None:
        """Clear any cached tensors."""

    @abstractmethod
    def supports_cuda(self) -> bool:
        """Return True if this is a CUDA backend."""

    def record_forward(self, elapsed_s: float):
        self.stats.forward_calls += 1
        self.stats.total_forward_time_s += elapsed_s

    def record_transfer(self, bytes_transferred: int, elapsed_s: float):
        self.stats.total_bytes_transferred += bytes_transferred
        self.stats.total_transfer_time_s += elapsed_s
