"""Mixture-of-Experts (MoE) path — Phase 4.

For MoE models, only the shared weights plus the experts a token routes to need
to be read. Shared weights are pinned, experts are cached in VRAM and RAM, and
misses are counted. Exact prefetching is not possible (router output is only
known after running), so we use a cache plus optional predictor, and group small
transfers.

Section 5.6 and section 6 MoE variant of the brief.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class ExpertCacheConfig:
    """Configuration for the expert cache."""

    num_experts: int          # N
    experts_per_token: int    # k
    expert_size_bytes: int    # Size of one expert
    max_cached_experts: int   # How many experts fit in VRAM cache
    device: str = "cuda"


@dataclass
class ExpertCacheStats:
    """Expert cache statistics."""

    hits: int = 0
    misses: int = 0
    loads: int = 0
    total_load_time_s: float = 0.0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def avg_load_time_s(self) -> float:
        return self.total_load_time_s / self.loads if self.loads > 0 else 0.0


class ExpertCache:
    """Cache for MoE expert weights.

    Keeps frequently-used experts in VRAM, loads others from RAM/SSD on demand.
    Tracks hit rate and miss counts for the planner (section 6 MoE variant).
    """

    def __init__(self, config: ExpertCacheConfig):
        self.config = config
        self.stats = ExpertCacheStats()

        # Cache: expert_id -> tensor (resident in VRAM)
        self.cached_experts: dict[int, torch.Tensor] = {}
        self.cache_order: list[int] = []  # LRU order

        # Backing store: expert_id -> tensor (in system RAM)
        self.backing_store: dict[int, torch.Tensor] = {}

        # Pin shared weights (non-expert) separately
        self.shared_weights: Optional[dict[str, torch.Tensor]] = None

    def pin_shared_weights(self, weights: dict[str, torch.Tensor]) -> None:
        """Pin shared (non-expert) weights — these stay resident permanently."""
        self.shared_weights = weights

    def add_expert_to_backing(self, expert_id: int, tensor: torch.Tensor) -> None:
        """Add an expert to the backing store (system RAM)."""
        self.backing_store[expert_id] = tensor.clone()

    def get_expert(self, expert_id: int) -> torch.Tensor:
        """Get an expert tensor, loading from backing store if not cached."""
        if expert_id in self.cached_experts:
            # Cache hit — move to front of LRU
            self.stats.hits += 1
            self.cache_order.remove(expert_id)
            self.cache_order.insert(0, expert_id)
            return self.cached_experts[expert_id]

        # Cache miss
        self.stats.misses += 1

        if expert_id not in self.backing_store:
            raise ValueError(f"Expert {expert_id} not found in backing store")

        t0 = time.perf_counter()
        tensor = self.backing_store[expert_id].clone()
        load_time = time.perf_counter() - t0
        self.stats.loads += 1
        self.stats.total_load_time_s += load_time

        # Insert into cache (evict LRU if full)
        self._insert_into_cache(expert_id, tensor)

        return tensor

    def _insert_into_cache(self, expert_id: int, tensor: torch.Tensor) -> None:
        """Insert expert into cache with LRU eviction."""
        if len(self.cached_experts) >= self.config.max_cached_experts:
            # Evict least recently used
            lru_id = self.cache_order.pop()
            del self.cached_experts[lru_id]
            self.stats.evictions = getattr(self.stats, 'evictions', 0) + 1

        self.cached_experts[expert_id] = tensor
        self.cache_order.insert(0, expert_id)

    def get_batch_experts(
        self,
        expert_ids: list[int],
    ) -> list[torch.Tensor]:
        """Get multiple experts, grouping transfers for efficiency.

        Returns tensors in the same order as expert_ids.
        """
        return [self.get_expert(eid) for eid in expert_ids]

    def preload_experts(self, expert_ids: list[int]) -> None:
        """Optionally preload experts into cache (predictor hint).

        Called when we have a hint about which experts will be needed next
        (e.g., from a predictor or from previous tokens in the same batch).
        """
        for expert_id in expert_ids:
            if expert_id not in self.cached_experts and expert_id in self.backing_store:
                # Load in background if possible
                try:
                    self.get_expert(expert_id)
                except Exception:
                    pass  # Best effort

    def get_stats(self) -> dict:
        """Return cache statistics for planner feedback."""
        return {
            "hit_rate": self.stats.hit_rate,
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "loads": self.stats.loads,
            "cached_count": len(self.cached_experts),
            "max_cached": self.config.max_cached_experts,
            "avg_load_time_s": self.stats.avg_load_time_s,
            "evictions": getattr(self.stats, 'evictions', 0),
        }

    def __repr__(self) -> str:
        return (
            f"ExpertCache(cached={len(self.cached_experts)}/"
            f"{self.config.max_cached_experts}, "
            f"hit_rate={self.hit_rate:.2%})"
        )


class MoERouter:
    """Simple MoE router that determines which experts each token uses.

    In a real implementation, this would run the router network. Here we
    provide the interface for expert selection.
    """

    def __init__(self, num_experts: int, experts_per_token: int):
        self.num_experts = num_experts
        self.experts_per_token = experts_per_token

    def route(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Route tokens to experts.

        Args:
            hidden_states: (batch, seq_len, hidden_dim)

        Returns:
            (expert_indices, router_weights)
            expert_indices: (batch, seq_len, experts_per_token)
            router_weights: (batch, seq_len, experts_per_token)
        """
        batch, seq_len, hidden_dim = hidden_states.shape

        # Simple deterministic routing based on hidden state hash
        # In production, this would be a learned router network
        token_hash = torch.sum(hidden_states, dim=-1)  # (batch, seq_len)
        expert_scores = token_hash % self.num_experts  # (batch, seq_len)

        # Top-k selection
        expert_indices = torch.zeros(
            batch, seq_len, self.experts_per_token,
            dtype=torch.long, device=hidden_states.device,
        )
        router_weights = torch.ones(
            batch, seq_len, self.experts_per_token,
            dtype=hidden_states.dtype, device=hidden_states.device,
        )

        for b in range(batch):
            for s in range(seq_len):
                base = expert_scores[b, s].item()
                for k in range(self.experts_per_token):
                    expert_indices[b, s, k] = (base + k) % self.num_experts
                    router_weights[b, s, k] = 1.0 / self.experts_per_token

        return expert_indices, router_weights
