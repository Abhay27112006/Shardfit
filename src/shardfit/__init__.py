"""ShardFit: Stream language model shards through VRAM to run models larger than GPU memory."""

__version__ = "0.3.0"

from shardfit.config import HardwareProfile, VRAMBudget, detect_hardware
from shardfit.planner import Planner, ModelSpec, PlanResult
from shardfit.sharder import ModelSharder, Shard, LayerBlock
from shardfit.streamer import Streamer, Slot, StreamStats
from shardfit.engine import ShardFitEngine, GenerationConfig, GenerationResult
from shardfit.backends import Backend, CudaBackend, CpuBackend, create_backend
from shardfit.agent import AgentRunner, AgentBudget, AgentResult, AgentStep
from shardfit.kv_cache import KVCache, KVCacheConfig, KVCacheStats
from shardfit.weights import WeightLoader, ShardedWeights, WeightTensor
from shardfit.moe import ExpertCache, ExpertCacheConfig, ExpertCacheStats, MoERouter

__all__ = [
    "__version__",
    "HardwareProfile",
    "VRAMBudget",
    "detect_hardware",
    "Planner",
    "ModelSpec",
    "PlanResult",
    "ModelSharder",
    "Shard",
    "LayerBlock",
    "Streamer",
    "Slot",
    "StreamStats",
    "ShardFitEngine",
    "GenerationConfig",
    "GenerationResult",
    "Backend",
    "CudaBackend",
    "CpuBackend",
    "create_backend",
    "AgentRunner",
    "AgentBudget",
    "AgentResult",
    "AgentStep",
    "KVCache",
    "KVCacheConfig",
    "KVCacheStats",
    "WeightLoader",
    "ShardedWeights",
    "WeightTensor",
    "ExpertCache",
    "ExpertCacheConfig",
    "ExpertCacheStats",
    "MoERouter",
]
