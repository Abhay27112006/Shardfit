"""Model weight loading and shard preparation.

Supports loading weights from HuggingFace (safetensors/pytorch) and preparing
them as contiguous buffers for streaming. Each shard is one contiguous memory
block so transfers are single large copies.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from safetensors.torch import load_file as load_safetensors

from shardfit.sharder import Shard, LayerBlock


@dataclass
class WeightTensor:
    """A single weight tensor with its metadata."""

    name: str
    tensor: torch.Tensor
    size_bytes: int
    shard_index: int = -1
    offset_in_shard: int = 0

    @property
    def size_gb(self) -> float:
        return self.size_bytes / (1024 ** 3)


@dataclass
class ShardedWeights:
    """All weights organized into shards with contiguous buffers."""

    model_name: str
    total_size_gb: float
    shards: list[Shard] = field(default_factory=list)
    # Mapping from parameter name to (shard_index, offset) for fast lookup
    param_to_shard: dict[str, tuple[int, int]] = field(default_factory=dict)
    # Resident weights (embedding, output head) that stay in VRAM
    resident_weights: dict[str, torch.Tensor] = field(default_factory=dict)

    def get_shard_for_param(self, param_name: str) -> Optional[tuple[int, int]]:
        """Return (shard_index, offset) for a parameter, or None if resident."""
        if param_name in self.resident_weights:
            return (-1, 0)  # resident
        return self.param_to_shard.get(param_name)

    def __repr__(self) -> str:
        return (
            f"ShardedWeights(model={self.model_name}, "
            f"total={self.total_size_gb:.2f} GB, "
            f"shards={len(self.shards)})"
        )


class WeightLoader:
    """Loads model weights and organizes them into shards."""

    # Parameter name patterns that indicate resident (non-streamed) weights
    RESIDENT_PATTERNS = (
        "embed",
        "embedding",
        "lm_head",
        "output_head",
        "word_embeddings",
        "position_embeddings",
        "norm",
        "layernorm",
        "layer_norm",
    )

    def __init__(
        self,
        model_path: str,
        dtype: torch.dtype = torch.float16,
        device: str = "cpu",
    ):
        self.model_path = model_path
        self.dtype = dtype
        self.device = device

    def load(
        self,
        slot_size_gb: float = 0.5,
    ) -> ShardedWeights:
        """Load weights from disk and organize into sharded format."""
        state_dict = self._load_state_dict()
        return self._organize_into_shards(state_dict, slot_size_gb=slot_size_gb)

    def _load_state_dict(self) -> dict[str, torch.Tensor]:
        """Load state dict from safetensors or pytorch bin files."""
        if os.path.isdir(self.model_path):
            # Look for safetensors files first
            for fname in os.listdir(self.model_path):
                if fname.endswith(".safetensors"):
                    fpath = os.path.join(self.model_path, fname)
                    return load_safetensors(fpath, device=self.device)

            # Look for pytorch_model.bin
            for fname in os.listdir(self.model_path):
                if fname.endswith(".bin"):
                    fpath = os.path.join(self.model_path, fname)
                    sd = torch.load(fpath, map_location=self.device)
                    if isinstance(sd, dict) and "state_dict" in sd:
                        return sd["state_dict"]
                    return sd

        # Single file
        if self.model_path.endswith(".safetensors"):
            return load_safetensors(self.model_path, device=self.device)
        elif self.model_path.endswith(".bin"):
            sd = torch.load(self.model_path, map_location=self.device)
            if isinstance(sd, dict) and "state_dict" in sd:
                return sd["state_dict"]
            return sd

        raise FileNotFoundError(
            f"Could not find model weights in {self.model_path}"
        )

    def _organize_into_shards(
        self,
        state_dict: dict[str, torch.Tensor],
        slot_size_gb: float = 0.5,
    ) -> ShardedWeights:
        """Organize weights into shards sized to fit in slots.

        Resident weights (embedding, head, norms) are kept separate.
        Streamed weights are grouped into shards of approximately slot_size_gb.
        """
        model_name = os.path.basename(self.model_path)

        # Separate resident from streamed weights
        resident: dict[str, torch.Tensor] = {}
        streamed: dict[str, torch.Tensor] = {}

        for name, tensor in state_dict.items():
            tensor = tensor.to(dtype=self.dtype, device=self.device)
            is_resident = any(pattern in name.lower() for pattern in self.RESIDENT_PATTERNS)
            if is_resident:
                resident[name] = tensor
            else:
                streamed[name] = tensor

        total_size = sum(
            t.element_size() * t.numel() for t in state_dict.values()
        )
        total_size_gb = total_size / (1024 ** 3)

        result = ShardedWeights(
            model_name=model_name,
            total_size_gb=total_size_gb,
            resident_weights=resident,
        )

        # Sort streamed params by name to keep layers contiguous
        sorted_params = sorted(streamed.items())

        slot_size_bytes = int(slot_size_gb * 1024 ** 3)
        current_shard: Optional[Shard] = None
        shard_index = 0
        current_shard_bytes = 0

        def new_shard() -> Shard:
            nonlocal shard_index
            s = Shard(index=shard_index)
            shard_index += 1
            return s

        for name, tensor in sorted_params:
            tensor_size = tensor.element_size() * tensor.numel()

            # Extract layer index for block type
            layer_idx = self._extract_layer_index(name)
            block_type = "attention" if any(
                x in name.lower() for x in ["attn", "attention", "query", "key", "value"]
            ) else "mlp"

            if current_shard is None:
                current_shard = new_shard()
                result.shards.append(current_shard)

            # If this tensor doesn't fit in current shard, start a new one
            if current_shard_bytes + tensor_size > slot_size_bytes and current_shard.blocks:
                current_shard = new_shard()
                result.shards.append(current_shard)
                current_shard_bytes = 0

            # Add tensor to shard
            block = LayerBlock(
                name=name,
                layer_idx=layer_idx if layer_idx is not None else -1,
                block_type=block_type,
                num_params=tensor.numel(),
                size_bytes=tensor_size,
                size_gb=tensor_size / (1024 ** 3),
                state_dict_keys=[name],
            )
            current_shard.add_block(block)

            # Record param-to-shard mapping
            offset = current_shard_bytes
            result.param_to_shard[name] = (shard_index - 1, offset)
            current_shard_bytes += tensor_size

        return result

    @staticmethod
    def _extract_layer_index(key: str) -> Optional[int]:
        """Extract transformer layer index from a parameter key."""
        parts = key.split(".")
        for i, part in enumerate(parts):
            if part in ("layers", "h", "blocks", "layer") and i + 1 < len(parts):
                try:
                    return int(parts[i + 1])
                except ValueError:
                    continue
        return None
