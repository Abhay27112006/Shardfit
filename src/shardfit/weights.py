"""Model weight loading and shard preparation.

Supports loading weights from HuggingFace (safetensors/pytorch), GGUF
(llama.cpp format), and preparing them as contiguous buffers for streaming.
Each shard is one contiguous memory block so transfers are single large copies.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field
from typing import Optional

import torch

try:
    from safetensors.torch import load_file as load_safetensors
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

try:
    import gguf
    HAS_GGUF = True
except ImportError:
    HAS_GGUF = False

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
    # Format of the source file
    source_format: str = "unknown"

    def get_shard_for_param(self, param_name: str) -> Optional[tuple[int, int]]:
        """Return (shard_index, offset) for a parameter, or None if resident."""
        if param_name in self.resident_weights:
            return (-1, 0)  # resident
        return self.param_to_shard.get(param_name)

    def __repr__(self) -> str:
        return (
            f"ShardedWeights(model={self.model_name}, "
            f"total={self.total_size_gb:.2f} GB, "
            f"shards={len(self.shards)}, "
            f"format={self.source_format})"
        )


class WeightLoader:
    """Loads model weights and organizes them into shards.

    Supports:
    - safetensors (.safetensors)
    - PyTorch (.bin, .pt)
    - GGUF (.gguf) — llama.cpp format
    """

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
        self.source_format = "unknown"

    def load(
        self,
        slot_size_gb: float = 0.5,
    ) -> ShardedWeights:
        """Load weights from disk and organize into sharded format."""
        state_dict, source_format = self._load_state_dict()
        self.source_format = source_format
        return self._organize_into_shards(state_dict, slot_size_gb=slot_size_gb)

    def _load_state_dict(self) -> tuple[dict[str, torch.Tensor], str]:
        """Load state dict from file.

        Returns:
            (state_dict, source_format_string)
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}"
            )

        # Detect format by extension
        ext = os.path.splitext(self.model_path)[1].lower()

        if ext == ".gguf":
            return self._load_gguf(), "gguf"
        elif ext == ".safetensors" and HAS_SAFETENSORS:
            return load_safetensors(self.model_path, device=self.device), "safetensors"
        elif ext in (".bin", ".pt", ".pth"):
            sd = torch.load(self.model_path, map_location=self.device)
            if isinstance(sd, dict) and "state_dict" in sd:
                return sd["state_dict"], "pytorch"
            return sd, "pytorch"
        elif os.path.isdir(self.model_path):
            return self._load_from_directory(), "directory"
        else:
            raise FileNotFoundError(
                f"Could not find model weights in {self.model_path} "
                f"(supported: .gguf, .safetensors, .bin, .pt, .pth, or directory)"
            )

    def _load_from_directory(self) -> dict[str, torch.Tensor]:
        """Load weights from a directory (HuggingFace format)."""
        # Look for safetensors files first
        for fname in sorted(os.listdir(self.model_path)):
            if fname.endswith(".safetensors") and HAS_SAFETENSORS:
                fpath = os.path.join(self.model_path, fname)
                return load_safetensors(fpath, device=self.device)

        # Look for pytorch_model.bin
        for fname in sorted(os.listdir(self.model_path)):
            if fname.endswith(".bin"):
                fpath = os.path.join(self.model_path, fname)
                sd = torch.load(fpath, map_location=self.device)
                if isinstance(sd, dict) and "state_dict" in sd:
                    return sd["state_dict"]
                return sd

        raise FileNotFoundError(
            f"No weight files found in {self.model_path}"
        )

    def _load_gguf(self) -> dict[str, torch.Tensor]:
        """Load a GGUF format model file.

        GGUF is the llama.cpp serialization format. It contains:
        - A header with magic number, version, tensor count, metadata KV count
        - Key-value metadata (model architecture, hyperparameters)
        - Tensor metadata (name, type, offset, shape)
        - Tensor data (concatenated after metadata)

        We parse the GGUF header and metadata, then load each tensor into a
        PyTorch tensor mapped to the appropriate location.
        """
        if not HAS_GGUF:
            # Fall back to manual GGUF parsing
            return self._load_gguf_manual()

        # Use the gguf Python package if available
        import gguf

        reader = gguf.GGUFReader(self.model_path)
        state_dict = {}

        for tensor_info in reader.tensors:
            name = tensor_info.name
            # Read tensor data from the file
            data = reader.get_tensor(tensor_info)

            # Convert to PyTorch tensor
            # GGUF types map to PyTorch dtypes
            dtype = self._gguf_type_to_torch(tensor_info.tensor_type)
            tensor = torch.from_numpy(data).to(dtype=dtype)

            # Reshape if needed
            if tensor_info.shape:
                tensor = tensor.reshape(tensor_info.shape)

            state_dict[name] = tensor

        return state_dict

    def _load_gguf_manual(self) -> dict[str, torch.Tensor]:
        """Manual GGUF parser (fallback when gguf package not installed).

        Parses the GGUF binary format directly. Supports GGUF version 1-3.
        """
        with open(self.model_path, "rb") as f:
            # Read header
            magic = f.read(4)
            if magic != b"GGUF":
                raise ValueError(f"Not a GGUF file: bad magic {magic}")

            version = struct.unpack("<I", f.read(4))[0]
            if version not in (1, 2, 3):
                raise ValueError(f"Unsupported GGUF version: {version}")

            tensor_count = struct.unpack("<I", f.read(4))[0]
            metadata_kv_count = struct.unpack("<I", f.read(4))[0]

            # Read key-value metadata (skip for now, just advance position)
            for _ in range(metadata_kv_count):
                key_len = struct.unpack("<Q", f.read(8))[0]
                key = f.read(key_len).decode("utf-8", errors="replace")
                val_type = struct.unpack("<B", f.read(1))[0]
                # Read value based on type (GGUF value types 0-13)
                if val_type == 0:  # uint8
                    f.read(1)
                elif val_type == 1:  # int8
                    f.read(1)
                elif val_type == 2:  # uint16
                    f.read(2)
                elif val_type == 3:  # int16
                    f.read(2)
                elif val_type == 4:  # uint32
                    f.read(4)
                elif val_type == 5:  # int32
                    f.read(4)
                elif val_type == 6:  # uint64
                    f.read(8)
                elif val_type == 7:  # int64
                    f.read(8)
                elif val_type == 8:  # float32
                    f.read(4)
                elif val_type == 9:  # bool
                    f.read(1)
                elif val_type == 10:  # string
                    str_len = struct.unpack("<Q", f.read(8))[0]
                    f.read(str_len)
                elif val_type == 11:  # array
                    arr_type = struct.unpack("<B", f.read(1))[0]
                    arr_len = struct.unpack("<Q", f.read(8))[0]
                    elem_size = {0:1,1:1,2:2,3:2,4:4,5:4,6:8,7:8,8:4}.get(arr_type, 1)
                    f.read(arr_len * elem_size)
                elif val_type == 12:  # unknown
                    pass
                elif val_type == 13:  # another type
                    pass

            # Align to 32 bytes
            pos = f.tell()
            align = 32 - (pos % 32)
            if align < 32:
                f.read(align)

            # Read tensor metadata
            tensors = []
            for _ in range(tensor_count):
                name_len = struct.unpack("<Q", f.read(8))[0]
                name = f.read(name_len).decode("utf-8", errors="replace")
                n_dims = struct.unpack("<I", f.read(4))[0]
                dims = []
                for _ in range(n_dims):
                    dims.append(struct.unpack("<Q", f.read(8))[0])
                tensor_type = struct.unpack("<I", f.read(4))[0]
                offset = struct.unpack("<Q", f.read(8))[0]
                tensors.append({
                    "name": name,
                    "dims": dims,
                    "type": tensor_type,
                    "offset": offset,
                })

            # Align to 32 bytes before tensor data
            pos = f.tell()
            align = 32 - (pos % 32)
            if align < 32:
                f.read(align)

            # Load tensors
            state_dict = {}
            for t in tensors:
                f.seek(t["offset"])
                data = self._read_gguf_tensor(f, t)
                dtype = self._gguf_type_to_torch_manual(t["type"])
                tensor = torch.from_numpy(data.copy()).to(dtype=dtype)
                if t["dims"]:
                    tensor = tensor.reshape(tuple(t["dims"]))
                state_dict[t["name"]] = tensor

        return state_dict

    @staticmethod
    def _read_gguf_tensor(f, tensor_info: dict) -> bytes:
        """Read raw tensor data from GGUF file."""
        # Calculate number of elements
        num_elements = 1
        for dim in tensor_info["dims"]:
            num_elements *= dim

        # GGUF type sizes (approximate for quantization)
        type_sizes = {
            0: 4,   # F32
            1: 2,   # F16
            2: 2,   # Q4_0 (4 bits per element, 2 bytes per element in blocks)
            3: 2,   # Q4_1
            5: 1,   # Q8_0
            6: 1,   # Q5_0
            7: 1,   # Q5_1
            8: 2,   # Q2_K
            9: 2,   # Q3_K_S
            10: 2,  # Q3_K_M
            11: 2,  # Q3_K_L
            12: 2,  # Q4_K_S
            13: 2,  # Q4_K_M
            14: 1,  # Q5_K_S
            15: 1,  # Q5_K_M
            16: 1,  # Q6_K
            17: 1,  # Q2_K_S (not standard)
        }
        elem_size = type_sizes.get(tensor_info["type"], 2)
        data_size = num_elements * elem_size // 2 if tensor_info["type"] >= 2 else num_elements * elem_size
        return f.read(data_size)

    @staticmethod
    def _gguf_type_to_torch(gguf_type: int) -> torch.dtype:
        """Convert GGUF tensor type to PyTorch dtype."""
        mapping = {
            0: torch.float32,   # F32
            1: torch.float16,   # F16
            2: torch.float16,   # Q4_0 (dequantize to f16)
            3: torch.float16,   # Q4_1
            5: torch.float16,   # Q8_0
            6: torch.float16,   # Q5_0
            7: torch.float16,   # Q5_1
            8: torch.float16,   # Q2_K
            9: torch.float16,   # Q3_K_S
            10: torch.float16,  # Q3_K_M
            11: torch.float16,  # Q3_K_L
            12: torch.float16,  # Q4_K_S
            13: torch.float16,  # Q4_K_M
            14: torch.float16,  # Q5_K_S
            15: torch.float16,  # Q5_K_M
            16: torch.float16,  # Q6_K
            17: torch.float16,
        }
        return mapping.get(gguf_type, torch.float16)

    @staticmethod
    def _gguf_type_to_torch_manual(gguf_type: int) -> torch.dtype:
        """Convert GGUF tensor type to PyTorch dtype (manual parser)."""
        return WeightLoader._gguf_type_to_torch(gguf_type)

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
            source_format=self.source_format,
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
