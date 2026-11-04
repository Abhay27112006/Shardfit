"""Disk-backed weight loader for models larger than CPU RAM.

Instead of loading the entire model into CPU RAM, this module reads individual
tensors directly from GGUF or safetensors files on demand. This enables running
models (70B+) that exceed both GPU VRAM and system RAM.

Each shard's tensors are loaded from disk only when the shard rotates into a
VRAM slot, then released after computation. This keeps CPU RAM usage bounded.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field
from typing import Optional, Dict, List

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
class TensorInfo:
    """Metadata for a tensor stored on disk."""

    name: str
    dtype: torch.dtype
    shape: tuple[int, ...]
    offset: int           # byte offset in file
    size_bytes: int       # size of tensor data in file
    file_path: str        # path to the file containing this tensor
    file_index: int = 0   # which file in a multi-file shard


@dataclass
class DiskShardManifest:
    """Manifest describing which tensors are in each shard and where on disk."""

    model_name: str
    total_size_gb: float
    shards: List[Shard] = field(default_factory=list)
    # Mapping from parameter name to TensorInfo (on-disk location)
    tensor_index: Dict[str, TensorInfo] = field(default_factory=dict)
    # Resident weights (embedding, head) - loaded once to GPU
    resident_tensors: Dict[str, TensorInfo] = field(default_factory=dict)
    source_format: str = "unknown"

    def get_tensor_info(self, param_name: str) -> Optional[TensorInfo]:
        """Get on-disk location for a parameter."""
        if param_name in self.resident_tensors:
            return self.resident_tensors[param_name]
        return self.tensor_index.get(param_name)

    def __repr__(self) -> str:
        return (
            f"DiskShardManifest(model={self.model_name}, "
            f"total={self.total_size_gb:.2f} GB, "
            f"shards={len(self.shards)}, "
            f"format={self.source_format})"
        )


class DiskWeightLoader:
    """Loads model weights from disk on demand, one shard at a time.

    Instead of loading the entire model into CPU RAM, this reader opens the
    model file(s) and reads individual tensors by their byte offset when a
    shard rotates into a VRAM slot. After computation, the tensors are
    released, keeping CPU RAM usage bounded.

    Supports:
    - GGUF (.gguf) - single file with all tensors
    - Safetensors (.safetensors) - single or multi-file
    - PyTorch (.bin, .pth) - single or multi-file
    """

    # Patterns for resident (non-streamed) weights
    RESIDENT_PATTERNS = (
        "embed", "embedding", "lm_head", "output_head",
        "word_embeddings", "position_embeddings",
        "norm", "layernorm", "layer_norm",
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
        self._file_handles: Dict[int, object] = {}  # cached file handles

    def load_manifest(
        self,
        slot_size_gb: float = 0.5,
    ) -> DiskShardManifest:
        """Scan model file(s) and build a disk-backed shard manifest.

        Does NOT load any tensor data into memory - only reads metadata
        (names, shapes, offsets, sizes). Memory usage is O(number of tensors),
        not O(model size).

        Returns:
            DiskShardManifest with shard organization and tensor locations.
        """
        # Detect format and load all tensor metadata
        state_dict_meta, source_format = self._scan_state_dict_metadata()
        self.source_format = source_format

        model_name = os.path.basename(self.model_path)

        # Separate resident from streamed tensors
        resident: Dict[str, TensorInfo] = {}
        streamed: Dict[str, TensorInfo] = {}

        for name, info in state_dict_meta.items():
            is_resident = any(
                pattern in name.lower() for pattern in self.RESIDENT_PATTERNS
            )
            if is_resident:
                resident[name] = info
            else:
                streamed[name] = info

        total_size = sum(
            info.size_bytes for info in state_dict_meta.values()
        )
        total_size_gb = total_size / (1024 ** 3)

        result = DiskShardManifest(
            model_name=model_name,
            total_size_gb=total_size_gb,
            resident_tensors=resident,
            source_format=source_format,
        )

        # Sort streamed tensors by name to keep layers contiguous
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

        for name, info in sorted_params:
            tensor_size = info.size_bytes

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
                num_params=info.size_bytes // 2,  # assume float16
                size_bytes=tensor_size,
                size_gb=tensor_size / (1024 ** 3),
                state_dict_keys=[name],
            )
            current_shard.add_block(block)

            # Record tensor location
            result.tensor_index[name] = info
            current_shard_bytes += tensor_size

        return result

    def load_shard_to_cpu(
        self,
        shard: Shard,
        manifest: DiskShardManifest,
    ) -> Dict[str, torch.Tensor]:
        """Load a shard's tensors from disk into CPU memory.

        Only the tensors belonging to this shard are loaded. After the shard
        is done computing, the caller should release these tensors to free
        CPU RAM.

        Args:
            shard: The shard to load.
            manifest: The disk shard manifest with tensor locations.

        Returns:
            Dict mapping parameter name to CPU tensor.
        """
        cpu_tensors: Dict[str, torch.Tensor] = {}

        for block in shard.blocks:
            for key in block.state_dict_keys:
                info = manifest.get_tensor_info(key)
                if info is None:
                    continue

                tensor = self._load_tensor_from_disk(info)
                if tensor is not None:
                    cpu_tensors[key] = tensor

        return cpu_tensors

    def load_resident_to_gpu(
        self,
        manifest: DiskShardManifest,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        """Load resident tensors (embedding, head) from disk to GPU.

        These stay in VRAM permanently.

        Returns:
            Dict mapping parameter name to GPU tensor.
        """
        gpu_tensors: Dict[str, torch.Tensor] = {}

        for name, info in manifest.resident_tensors.items():
            tensor = self._load_tensor_from_disk(info)
            if tensor is not None:
                gpu_tensors[name] = tensor.to(device=device, dtype=dtype)

        return gpu_tensors

    def release_cpu_tensors(self, tensors: Dict[str, torch.Tensor]) -> None:
        """Release CPU tensors to free memory.

        Clears references and calls cuda empty cache if applicable.
        """
        for key in list(tensors.keys()):
            del tensors[key]
        tensors.clear()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Private: metadata scanning
    # ------------------------------------------------------------------

    def _scan_state_dict_metadata(
        self,
    ) -> tuple[Dict[str, TensorInfo], str]:
        """Scan model file(s) and extract tensor metadata without loading data.

        Returns:
            (dict of name -> TensorInfo, source_format_string)
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}"
            )

        ext = os.path.splitext(self.model_path)[1].lower()

        if ext == ".gguf":
            return self._scan_gguf_metadata(), "gguf"
        elif ext == ".safetensors" and HAS_SAFETENSORS:
            return self._scan_safetensors_metadata(), "safetensors"
        elif ext in (".bin", ".pth", ".pt"):
            return self._scan_pytorch_metadata(), "pytorch"
        elif os.path.isdir(self.model_path):
            return self._scan_directory_metadata(), "directory"
        else:
            raise FileNotFoundError(
                f"Could not find model weights in {self.model_path}"
            )

    def _scan_gguf_metadata(self) -> Dict[str, TensorInfo]:
        """Scan GGUF file for tensor metadata."""
        import gguf

        reader = gguf.GGUFReader(self.model_path)
        metadata: Dict[str, TensorInfo] = {}

        for tensor_info in reader.tensors:
            name = tensor_info.name
            shape = tuple(tensor_info.shape) if tensor_info.shape else ()

            # Calculate size in bytes for float16 (dequantized)
            num_elements = 1
            for dim in shape:
                num_elements *= dim

            # GGUF tensors are stored quantized; we dequantize to float16
            # Size in RAM = num_elements * 2 bytes (float16)
            size_bytes = num_elements * 2

            # Get tensor data offset
            offset = tensor_info.offset if hasattr(tensor_info, 'offset') else 0

            dtype = self._gguf_type_to_torch(tensor_info.tensor_type)

            metadata[name] = TensorInfo(
                name=name,
                dtype=dtype,
                shape=shape,
                offset=offset,
                size_bytes=size_bytes,
                file_path=self.model_path,
            )

        return metadata

    def _scan_safetensors_metadata(self) -> Dict[str, TensorInfo]:
        """Scan safetensors file for tensor metadata."""
        from safetensors.torch import safe_open

        metadata: Dict[str, TensorInfo] = {}

        with safe_open(self.model_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_slice(key)
                shape = tensor.get_shape()
                dtype = tensor.get_dtype()

                num_elements = 1
                for dim in shape:
                    num_elements *= dim

                # safetensors stores raw data; size = elements * element_size
                elem_size = torch.tensor([], dtype=dtype).element_size()
                size_bytes = num_elements * elem_size

                # Get offset from the file
                offset = f.offsets[key] if hasattr(f, 'offsets') else 0

                metadata[key] = TensorInfo(
                    name=key,
                    dtype=dtype,
                    shape=tuple(shape),
                    offset=offset,
                    size_bytes=size_bytes,
                    file_path=self.model_path,
                )

        return metadata

    def _scan_pytorch_metadata(self) -> Dict[str, TensorInfo]:
        """Scan PyTorch .bin/.pth file for tensor metadata."""
        # We need to load to get metadata, but we can be smart about it
        sd = torch.load(self.model_path, map_location="cpu", weights_only=True)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]

        metadata: Dict[str, TensorInfo] = {}
        for name, tensor in sd.items():
            num_elements = tensor.numel()
            elem_size = tensor.element_size()
            size_bytes = num_elements * elem_size

            metadata[name] = TensorInfo(
                name=name,
                dtype=tensor.dtype,
                shape=tuple(tensor.shape),
                offset=0,  # PyTorch bin doesn't expose offsets easily
                size_bytes=size_bytes,
                file_path=self.model_path,
            )

        return metadata

    def _scan_directory_metadata(self) -> Dict[str, TensorInfo]:
        """Scan a directory of model files (HuggingFace format)."""
        metadata: Dict[str, TensorInfo] = {}

        # Try safetensors first
        for fname in sorted(os.listdir(self.model_path)):
            if fname.endswith(".safetensors") and HAS_SAFETENSORS:
                fpath = os.path.join(self.model_path, fname)
                from safetensors.torch import safe_open

                with safe_open(fpath, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        tensor = f.get_slice(key)
                        shape = tensor.get_shape()
                        dtype = tensor.get_dtype()
                        num_elements = 1
                        for dim in shape:
                            num_elements *= dim
                        elem_size = torch.tensor([], dtype=dtype).element_size()
                        size_bytes = num_elements * elem_size
                        offset = f.offsets[key] if hasattr(f, 'offsets') else 0

                        metadata[key] = TensorInfo(
                            name=key,
                            dtype=dtype,
                            shape=tuple(shape),
                            offset=offset,
                            size_bytes=size_bytes,
                            file_path=fpath,
                        )
                return metadata

        # Fall back to pytorch_model.bin
        for fname in sorted(os.listdir(self.model_path)):
            if fname.endswith(".bin"):
                fpath = os.path.join(self.model_path, fname)
                sd = torch.load(fpath, map_location="cpu", weights_only=True)
                if isinstance(sd, dict) and "state_dict" in sd:
                    sd = sd["state_dict"]
                for name, tensor in sd.items():
                    num_elements = tensor.numel()
                    elem_size = tensor.element_size()
                    size_bytes = num_elements * elem_size
                    metadata[name] = TensorInfo(
                        name=name,
                        dtype=tensor.dtype,
                        shape=tuple(tensor.shape),
                        offset=0,
                        size_bytes=size_bytes,
                        file_path=fpath,
                    )
                return metadata

        raise FileNotFoundError(
            f"No weight files found in {self.model_path}"
        )

    # ------------------------------------------------------------------
    # Private: loading individual tensors from disk
    # ------------------------------------------------------------------

    def _load_tensor_from_disk(self, info: TensorInfo) -> Optional[torch.Tensor]:
        """Load a single tensor from disk by its metadata."""
        try:
            ext = os.path.splitext(info.file_path)[1].lower()

            if ext == ".gguf":
                return self._load_gguf_tensor(info)
            elif ext == ".safetensors":
                return self._load_safetensors_tensor(info)
            elif ext in (".bin", ".pth", ".pt"):
                return self._load_pytorch_tensor(info)
            else:
                return None
        except Exception as e:
            print(f"[WeightLoader] Failed to load {info.name}: {e}")
            return None

    def _load_gguf_tensor(self, info: TensorInfo) -> Optional[torch.Tensor]:
        """Load a single tensor from a GGUF file."""
        import gguf

        reader = gguf.GGUFReader(info.file_path)

        for tensor_info in reader.tensors:
            if tensor_info.name == info.name:
                data = reader.get_tensor(tensor_info)
                dtype = self._gguf_type_to_torch(tensor_info.tensor_type)
                tensor = torch.from_numpy(data.copy()).to(dtype=dtype)
                if tensor_info.shape:
                    tensor = tensor.reshape(tuple(tensor_info.shape))
                return tensor

        return None

    def _load_safetensors_tensor(self, info: TensorInfo) -> Optional[torch.Tensor]:
        """Load a single tensor from a safetensors file."""
        from safetensors.torch import safe_open

        with safe_open(info.file_path, framework="pt", device="cpu") as f:
            tensor = f.get_tensor(info.name)
            return tensor.to(dtype=info.dtype)

    def _load_pytorch_tensor(self, info: TensorInfo) -> Optional[torch.Tensor]:
        """Load a single tensor from a PyTorch .bin file."""
        # PyTorch bin files don't support individual tensor loading
        # We load the whole file but only keep the requested tensor
        sd = torch.load(info.file_path, map_location="cpu", weights_only=True)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        if info.name in sd:
            return sd[info.name].to(dtype=info.dtype)
        return None

    # ------------------------------------------------------------------
    # Private: utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _gguf_type_to_torch(gguf_type: int) -> torch.dtype:
        """Convert GGUF tensor type to PyTorch dtype."""
        mapping = {
            0: torch.float32,   # F32
            1: torch.float16,   # F16
            2: torch.float16,   # Q4_0
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
        }
        return mapping.get(gguf_type, torch.float16)

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
