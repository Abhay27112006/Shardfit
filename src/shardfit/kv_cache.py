"""KV Cache management for streaming inference.

The KV cache stores attention keys and values for the context. It lives in VRAM
and grows with context length, shrinking the weight budget (section 14 risk 2).

For streaming, the KV cache must remain resident while shards rotate through
slots, so it is allocated separately from the streaming slots.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class KVCacheConfig:
    """Configuration for the KV cache."""

    num_layers: int
    num_heads: int
    head_dim: int
    max_context_len: int
    dtype: torch.dtype = torch.float16
    device: str = "cuda"


@dataclass
class KVCacheStats:
    """KV cache statistics."""

    total_memory_mb: float = 0.0
    writes: int = 0
    reads: int = 0
    evictions: int = 0


class KVCache:
    """Key-Value cache for transformer attention.

    Pre-allocates space for the maximum context length. Each layer has its own
    K and V tensors of shape (max_context_len, num_heads, head_dim).

    The cache is resident in VRAM and is not evicted when shards rotate.
    """

    def __init__(self, config: KVCacheConfig):
        self.config = config
        self.stats = KVCacheStats()

        # Pre-allocate K and V for all layers
        # Shape: (num_layers, max_context_len, num_heads, head_dim)
        k_shape = (
            config.num_layers,
            config.max_context_len,
            config.num_heads,
            config.head_dim,
        )
        v_shape = k_shape

        dtype = config.dtype
        element_size = torch.tensor([], dtype=dtype).element_size()

        self.k_cache = torch.zeros(k_shape, dtype=dtype, device=config.device)
        self.v_cache = torch.zeros(v_shape, dtype=dtype, device=config.device)

        # Current position in the cache (how many tokens are stored)
        self.context_len = 0

        # Calculate memory usage
        total_elements = config.num_layers * config.max_context_len * config.num_heads * config.head_dim
        self.stats.total_memory_mb = (total_elements * 2 * element_size) / (1024 ** 2)  # K + V

    def reset(self) -> None:
        """Clear the cache for a new sequence."""
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.context_len = 0

    def store(
        self,
        layer_idx: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        start_pos: int,
    ) -> None:
        """Store keys and values for a layer at the given position.

        Args:
            layer_idx: Which layer's cache to write to
            keys: Key tensor of shape (batch, seq_len, num_heads, head_dim)
                  or (seq_len, num_heads, head_dim)
            values: Value tensor of same shape
            start_pos: Starting position in the cache
        """
        # Remove batch dimension if present (assume batch=1 for streaming)
        if keys.dim() == 4:
            keys = keys.squeeze(0)
        if values.dim() == 4:
            values = values.squeeze(0)

        seq_len = keys.shape[0]
        end_pos = start_pos + seq_len

        if end_pos > self.config.max_context_len:
            raise ValueError(
                f"KV cache overflow: need {end_pos} positions, "
                f"max is {self.config.max_context_len}"
            )

        self.k_cache[layer_idx, start_pos:end_pos] = keys
        self.v_cache[layer_idx, start_pos:end_pos] = values

        self.stats.writes += 1

        if end_pos > self.context_len:
            self.context_len = end_pos

    def get(self, layer_idx: int, end_pos: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieve keys and values up to end_pos for a layer.

        Returns:
            (keys, values) each of shape (batch, end_pos, num_heads, head_dim)
        """
        self.stats.reads += 1
        return (
            self.k_cache[layer_idx, :end_pos],
            self.v_cache[layer_idx, :end_pos],
        )

    def get_memory_mb(self) -> float:
        """Return current KV cache memory usage in MB."""
        return self.stats.total_memory_mb

    def __repr__(self) -> str:
        return (
            f"KVCache(layers={self.config.num_layers}, "
            f"context={self.context_len}/{self.config.max_context_len}, "
            f"memory={self.stats.total_memory_mb:.1f} MB)"
        )
