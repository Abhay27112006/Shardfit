"""Hardware profile and VRAM budget configuration (section 5.2 of the brief)."""

from dataclasses import dataclass
from typing import Optional
import platform


@dataclass
class HardwareProfile:
    """Measured hardware characteristics."""

    # GPU
    gpu_name: str
    gpu_vram_total_gb: float          # V: total VRAM
    gpu_vram_bandwidth_gb_s: float    # v: VRAM bandwidth
    pcie_bandwidth_gb_s: float        # host-to-GPU effective bandwidth
    cuda_available: bool

    # System RAM
    ram_total_gb: float
    ram_available_gb: float           # H: host RAM available for pinning

    # Storage
    nvme_read_gb_s: float             # SSD sequential read bandwidth

    # Platform
    os_name: str
    architecture: str

    @property
    def is_macos(self) -> bool:
        return self.os_name == "Darwin"

    @property
    def is_windows(self) -> bool:
        return self.os_name == "Windows"

    @property
    def is_apple_silicon(self) -> bool:
        return self.is_macos and self.architecture == "arm64"

    @property
    def uses_unified_memory(self) -> bool:
        """Unified-memory machines (Apple Silicon) gain nothing from streaming within RAM."""
        return self.is_apple_silicon


@dataclass
class VRAMBudget:
    """VRAM budget allocation (section 5.2).

    Let V be total VRAM. Reserve R for KV cache, activations, CUDA context,
    workspace. Weights get W = V - R. Allocate S streaming slots of size C
    each. The remaining Pin = W - S*C holds resident shards.
    """

    vram_total_gb: float          # V
    reserve_gb: float             # R
    weights_budget_gb: float      # W = V - R
    num_slots: int                # S (at least 2 for double buffering)
    slot_size_gb: float           # C
    resident_budget_gb: float     # Pin = W - S*C

    @classmethod
    def calculate(
        cls,
        vram_total_gb: float,
        reserve_gb: float = 1.5,
        num_slots: int = 2,
        slot_size_gb: float = 0.5,
    ) -> "VRAMBudget":
        """Calculate VRAM budget from total VRAM and parameters."""
        weights_budget = vram_total_gb - reserve_gb
        resident_budget = weights_budget - num_slots * slot_size_gb
        return cls(
            vram_total_gb=vram_total_gb,
            reserve_gb=reserve_gb,
            weights_budget_gb=weights_budget,
            num_slots=num_slots,
            slot_size_gb=slot_size_gb,
            resident_budget_gb=max(0.0, resident_budget),
        )

    def validate(self) -> list[str]:
        """Return list of validation errors, or empty if valid."""
        errors: list[str] = []
        if self.num_slots < 2:
            errors.append("S (num_slots) must be at least 2 for double buffering")
        if self.slot_size_gb <= 0:
            errors.append("C (slot_size_gb) must be positive")
        if self.weights_budget_gb <= 0:
            errors.append("W (weights_budget_gb) must be positive")
        if self.resident_budget_gb < 0:
            errors.append("Pin (resident_budget_gb) is negative — reduce slots or slot size")
        if self.slot_size_gb > self.weights_budget_gb / self.num_slots:
            errors.append("C <= W/S required (shard must fit in a slot)")
        return errors


def detect_hardware() -> HardwareProfile:
    """Detect the current machine's hardware profile."""
    import psutil

    os_name = platform.system()
    architecture = platform.machine()

    # GPU detection
    cuda_available = False
    gpu_name = "None"
    gpu_vram_total_gb = 0.0
    gpu_vram_bandwidth_gb_s = 0.0
    pcie_bandwidth_gb_s = 12.0  # conservative default

    try:
        import torch

        if torch.cuda.is_available():
            cuda_available = True
            gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            gpu_vram_total_gb = props.total_memory / (1024 ** 3)
            # Estimate VRAM bandwidth (conservative)
            # For RTX 3050 6GB: ~168 GB/s (from brief assumptions)
            gpu_vram_bandwidth_gb_s = 168.0 if "3050" in gpu_name else 300.0
    except ImportError:
        pass

    # System RAM
    ram = psutil.virtual_memory()
    ram_total_gb = ram.total / (1024 ** 3)
    ram_available_gb = ram.available / (1024 ** 3)

    # Storage — conservative estimate
    nvme_read_gb_s = 3.0

    return HardwareProfile(
        gpu_name=gpu_name,
        gpu_vram_total_gb=gpu_vram_total_gb,
        gpu_vram_bandwidth_gb_s=gpu_vram_bandwidth_gb_s,
        pcie_bandwidth_gb_s=pcie_bandwidth_gb_s,
        cuda_available=cuda_available,
        ram_total_gb=ram_total_gb,
        ram_available_gb=ram_available_gb,
        nvme_read_gb_s=nvme_read_gb_s,
        os_name=os_name,
        architecture=architecture,
    )
