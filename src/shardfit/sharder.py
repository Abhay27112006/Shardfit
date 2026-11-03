"""Model sharding.

A shard is a contiguous block of one or a few transformer layers. If a single
layer is bigger than a streaming slot, split it into attention and MLP
sub-blocks. Each shard is stored as one contiguous buffer so a transfer is one
large copy, not hundreds of small ones.

Section 5.1 of the brief.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class LayerBlock:
    """A single transformer layer or sub-block (attention or MLP)."""

    name: str
    layer_idx: int
    block_type: str  # "layer", "attention", "mlp"
    num_params: int
    size_bytes: int
    size_gb: float
    state_dict_keys: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"LayerBlock({self.name}, {self.block_type}, {self.size_gb:.3f} GB)"


@dataclass
class Shard:
    """A contiguous block of layers/sub-blocks moved as one unit."""

    index: int
    blocks: list[LayerBlock] = field(default_factory=list)
    total_params: int = 0
    total_size_bytes: int = 0
    total_size_gb: float = 0.0
    state_dict: dict = None  # populated at load time

    def __post_init__(self):
        if self.state_dict is None:
            self.state_dict = {}
        self._recalc_totals()

    def _recalc_totals(self):
        self.total_params = sum(b.num_params for b in self.blocks)
        self.total_size_bytes = sum(b.size_bytes for b in self.blocks)
        self.total_size_gb = self.total_size_bytes / (1024**3)

    def add_block(self, block: LayerBlock):
        self.blocks.append(block)
        self._recalc_totals()

    def __repr__(self) -> str:
        layer_ids = sorted(set(b.layer_idx for b in self.blocks))
        return f"Shard {self.index}: layers {layer_ids}, {self.total_size_gb:.3f} GB"


class ModelSharder:
    """Splits a model into shards of contiguous layers."""

    def __init__(
        self,
        slot_size_gb: float,
        sub_block_on_overflow: bool = True,
    ):
        self.slot_size_gb = slot_size_gb
        self.slot_size_bytes = int(slot_size_gb * 1024**3)
        self.sub_block_on_overflow = sub_block_on_overflow

    def analyze_model(self, model: torch.nn.Module) -> list[LayerBlock]:
        """Walk the model and identify transformer layers and their sizes."""
        blocks: list[LayerBlock] = []

        # Find transformer layers by looking for common patterns
        layer_idx = 0
        seen_prefixes = set()

        for name, module in model.named_modules():
            # Detect layer blocks: look for "layers.{i}" or "h.{i}" patterns
            parts = name.split(".")
            if len(parts) >= 2:
                try:
                    idx = int(parts[-2])
                    prefix = ".".join(parts[:-2])
                    if prefix not in seen_prefixes:
                        seen_prefixes.add(prefix)
                        layer_idx = idx
                except (ValueError, IndexError):
                    pass

            # We can't easily split individual tensors at this level;
            # instead we group by parameter
            for param_name, param in module.named_parameters(recurse=False):
                full_name = f"{name}.{param_name}" if name else param_name
                num_params = param.numel()
                size_bytes = param.element_size() * num_params

                blocks.append(
                    LayerBlock(
                        name=full_name,
                        layer_idx=layer_idx,
                        block_type="layer",
                        num_params=num_params,
                        size_bytes=size_bytes,
                        size_gb=size_bytes / (1024**3),
                        state_dict_keys=[full_name],
                    )
                )

        return blocks

    def shard_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        layer_to_block_map: Optional[dict[int, str]] = None,
    ) -> list[Shard]:
        """Split a state_dict into shards sized to fit in slots.

        Each shard is a contiguous group of layer parameters.
        """
        # Group parameters by layer
        layer_params: dict[int, dict[str, torch.Tensor]] = {}
        layer_order: list[int] = []

        for key, tensor in state_dict.items():
            # Extract layer index from key (e.g., "layers.5.attn.weight" -> 5)
            layer_idx = self._extract_layer_index(key)
            if layer_idx is None:
                # Non-layer params (embedding, head, norm) go to layer -1 (resident)
                layer_idx = -1
            if layer_idx not in layer_params:
                layer_params[layer_idx] = {}
                if layer_idx != -1:
                    layer_order.append(layer_idx)
            layer_params[layer_idx][key] = tensor

        # Sort layers; resident (-1) handled separately
        layer_order.sort()

        shards: list[Shard] = []
        current_shard: Optional[Shard] = None
        shard_index = 0

        def new_shard() -> Shard:
            nonlocal shard_index
            s = Shard(index=shard_index)
            shard_index += 1
            return s

        # First shard: resident params (embedding, output head, layer norms)
        resident_params = layer_params.pop(-1, None)
        if resident_params:
            current_shard = new_shard()
            for key, tensor in resident_params.items():
                block = LayerBlock(
                    name=key,
                    layer_idx=-1,
                    block_type="resident",
                    num_params=tensor.numel(),
                    size_bytes=tensor.element_size() * tensor.numel(),
                    size_gb=(tensor.element_size() * tensor.numel()) / (1024**3),
                    state_dict_keys=[key],
                )
                current_shard.add_block(block)
            shards.append(current_shard)
            current_shard = None  # force new shard for streamed layers

        # Process each layer
        for layer_idx in layer_order:
            if layer_idx not in layer_params:
                continue
            params = layer_params[layer_idx]

            # Check if this single layer fits in a slot
            layer_size = sum(
                t.element_size() * t.numel() for t in params.values()
            )

            if layer_size > self.slot_size_bytes and self.sub_block_on_overflow:
                # Split into attention and MLP sub-blocks
                attn_params = {
                    k: v for k, v in params.items()
                    if any(x in k.lower() for x in ["attn", "attention", "q_", "k_", "v_", "o_", "query", "key", "value"])
                }
                mlp_params = {
                    k: v for k, v in params.items()
                    if k not in attn_params
                }

                for sub_name, sub_params in [("attention", attn_params), ("mlp", mlp_params)]:
                    if not sub_params:
                        continue
                    sub_size = sum(t.element_size() * t.numel() for t in sub_params.values())

                    if current_shard is None:
                        current_shard = new_shard()
                        shards.append(current_shard)

                    if current_shard.total_size_bytes + sub_size > self.slot_size_bytes and current_shard.blocks:
                        # Current shard full, start new one
                        current_shard = new_shard()
                        shards.append(current_shard)

                    for key, tensor in sub_params.items():
                        block = LayerBlock(
                            name=key,
                            layer_idx=layer_idx,
                            block_type=sub_name,
                            num_params=tensor.numel(),
                            size_bytes=tensor.element_size() * tensor.numel(),
                            size_gb=(tensor.element_size() * tensor.numel()) / (1024**3),
                            state_dict_keys=[key],
                        )
                        current_shard.add_block(block)
            else:
                # Whole layer goes in one shard
                if current_shard is None:
                    current_shard = new_shard()
                    shards.append(current_shard)

                if current_shard.total_size_bytes + layer_size > self.slot_size_bytes and current_shard.blocks:
                    # Current shard full, start new one
                    current_shard = new_shard()
                    shards.append(current_shard)

                for key, tensor in params.items():
                    block = LayerBlock(
                        name=key,
                        layer_idx=layer_idx,
                        block_type="layer",
                        num_params=tensor.numel(),
                        size_bytes=tensor.element_size() * tensor.numel(),
                        size_gb=(tensor.element_size() * tensor.numel()) / (1024**3),
                        state_dict_keys=[key],
                    )
                    current_shard.add_block(block)

        # Filter out empty shards
        shards = [s for s in shards if s.blocks]

        return shards

    @staticmethod
    def _extract_layer_index(key: str) -> Optional[int]:
        """Extract transformer layer index from a parameter key.

        Looks for patterns like:
        - layers.5.attn.weight -> 5
        - h.5.attn.weight -> 5
        - model.layers.5 -> 5
        - transformer.blocks.5 -> 5
        """
        parts = key.split(".")
        for i, part in enumerate(parts):
            if part in ("layers", "h", "blocks", "layer") and i + 1 < len(parts):
                try:
                    return int(parts[i + 1])
                except ValueError:
                    continue
        return None

    def summarize(self, shards: list[Shard]) -> str:
        """Return a human-readable summary of the sharding plan."""
        lines = []
        lines.append(f"Total shards: {len(shards)}")
        lines.append(f"Slot size: {self.slot_size_gb} GB")
        lines.append("")
        for shard in shards:
            resident = "RESIDENT" if any(b.block_type == "resident" for b in shard.blocks) else "streamed"
            lines.append(
                f"  {shard}: {resident}"
            )
        return "\n".join(lines)
