"""Core tests for ShardFit modules.

These tests validate the sharder, planner, profiler, and backends
without requiring a downloaded model.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from shardfit.config import HardwareProfile, VRAMBudget, detect_hardware
from shardfit.planner import Planner, ModelSpec
from shardfit.sharder import ModelSharder, Shard, LayerBlock
from shardfit.backends import CpuBackend, create_backend
from shardfit.streamer import Streamer, Slot


def test_vram_budget():
    """Test VRAM budget calculation."""
    budget = VRAMBudget.calculate(
        vram_total_gb=6.0,
        reserve_gb=1.5,
        num_slots=2,
        slot_size_gb=0.5,
    )
    assert budget.weights_budget_gb == 4.5, f"Expected 4.5, got {budget.weights_budget_gb}"
    assert budget.resident_budget_gb == 3.5, f"Expected 3.5, got {budget.resident_budget_gb}"
    assert budget.num_slots == 2
    assert budget.slot_size_gb == 0.5
    errors = budget.validate()
    assert errors == [], f"Validation errors: {errors}"
    print("PASS: test_vram_budget")


def test_vram_budget_validation():
    """Test that invalid budgets are caught."""
    # S < 2 should fail
    budget = VRAMBudget.calculate(
        vram_total_gb=6.0,
        reserve_gb=1.5,
        num_slots=1,
        slot_size_gb=0.5,
    )
    errors = budget.validate()
    assert len(errors) > 0, "Should catch S < 2"

    # C > W/S should fail
    budget = VRAMBudget.calculate(
        vram_total_gb=6.0,
        reserve_gb=1.5,
        num_slots=2,
        slot_size_gb=3.0,  # W/S = 4.5/2 = 2.25, so 3.0 > 2.25
    )
    errors = budget.validate()
    assert len(errors) > 0, "Should catch C > W/S"
    print("PASS: test_vram_budget_validation")


def test_planner_dense():
    """Test the dense model planner."""
    hardware = HardwareProfile(
        gpu_name="RTX 3050",
        gpu_vram_total_gb=6.0,
        gpu_vram_bandwidth_gb_s=168.0,
        pcie_bandwidth_gb_s=12.0,
        cuda_available=True,
        ram_total_gb=16.0,
        ram_available_gb=8.0,
        nvme_read_gb_s=3.0,
        os_name="Windows",
        architecture="x86_64",
    )
    budget = VRAMBudget.calculate(6.0, 1.5, 2, 0.5)
    planner = Planner(hardware, budget)

    model = ModelSpec(name="test-8b", size_gb=4.9)
    result = planner.plan_dense(model, context_len=2048)

    assert result.tok_s > 0, f"Expected positive tok/s, got {result.tok_s}"
    assert result.go_no_go in ("go", "no-go")
    assert result.streamed_per_pass_gb == 4.9 - 3.5  # M - Pin
    print(f"PASS: test_planner_dense (tok/s={result.tok_s:.2f}, verdict={result.go_no_go})")


def test_planner_14b_no_fit():
    """Test that 14B model on 6GB VRAM is planned correctly."""
    hardware = HardwareProfile(
        gpu_name="RTX 3050",
        gpu_vram_total_gb=6.0,
        gpu_vram_bandwidth_gb_s=168.0,
        pcie_bandwidth_gb_s=12.0,
        cuda_available=True,
        ram_total_gb=16.0,
        ram_available_gb=8.0,
        nvme_read_gb_s=3.0,
        os_name="Windows",
        architecture="x86_64",
    )
    budget = VRAMBudget.calculate(6.0, 1.5, 2, 0.5)
    planner = Planner(hardware, budget)

    # 14B Q4_K_M is ~9 GB
    model = ModelSpec(name="test-14b", size_gb=9.0)
    result = planner.plan_dense(model, context_len=2048)

    assert result.streamed_per_pass_gb > 0
    print(f"PASS: test_planner_14b_no_fit (tok/s={result.tok_s:.3f}, streamed={result.streamed_per_pass_gb:.1f} GB)")


def test_sharder():
    """Test model sharding with a tiny model."""
    # Create a tiny model with known structure
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(100, 16)
            self.layers = torch.nn.ModuleList([
                torch.nn.Linear(16, 16) for _ in range(4)
            ])
            self.head = torch.nn.Linear(16, 100)

        def forward(self, x):
            return self.head(self.layers[-1](x))

    model = TinyModel()
    state_dict = model.state_dict()

    sharder = ModelSharder(slot_size_gb=0.001)  # tiny slot to force many shards
    shards = sharder.shard_state_dict(state_dict)

    assert len(shards) >= 1, "Should create at least one shard"
    total_size = sum(s.total_size_bytes for s in shards)
    expected_size = sum(t.element_size() * t.numel() for t in state_dict.values())
    assert total_size == expected_size, f"Shard sizes don't match: {total_size} vs {expected_size}"
    print(f"PASS: test_sharder ({len(shards)} shards, {total_size / 1024:.1f} KB)")


def test_cpu_backend():
    """Test the CPU backend."""
    backend = CpuBackend(num_threads=2)
    backend.initialize()
    assert backend.device is not None
    assert not backend.supports_cuda()

    # Test forward
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    logits, kv = backend.forward(input_ids)
    assert logits is not None
    print(f"PASS: test_cpu_backend (memory={backend.get_memory_used_mb()} MB)")


def test_create_backend_auto():
    """Test backend factory with auto selection."""
    backend = create_backend("auto")
    assert backend is not None
    cuda_available = backend.supports_cuda()
    print(f"PASS: test_create_backend_auto ({type(backend).__name__}, cuda={cuda_available})")


def test_streamer():
    """Test the streamer with dummy shards."""
    backend = CpuBackend()
    backend.initialize()

    streamer = Streamer(backend=backend, num_slots=2, slot_size_gb=1.0)

    # Create dummy shards
    shards = []
    for i in range(3):
        block = LayerBlock(
            name=f"layer_{i}.weight",
            layer_idx=i,
            block_type="layer",
            num_params=1000,
            size_bytes=4000,
            size_gb=4000 / (1024**3),
            state_dict_keys=[f"layer_{i}.weight"],
        )
        shard = Shard(index=i)
        shard.add_block(block)
        shards.append(shard)

    streamer.load_shards(shards)

    # Get shards one by one
    count = 0
    while True:
        shard = streamer.get_next_shard()
        if shard is None:
            break
        count += 1
        streamer.release_slot(shard)

    assert count == 3, f"Expected 3 shards, got {count}"
    print(f"PASS: test_streamer ({count} shards streamed)")


def test_kv_cache():
    """Test KV cache store and retrieve."""
    from shardfit.kv_cache import KVCache, KVCacheConfig
    import torch

    config = KVCacheConfig(
        num_layers=4,
        num_heads=8,
        head_dim=64,
        max_context_len=128,
        device="cpu",
    )
    cache = KVCache(config)

    # Store keys/values for layer 0
    batch, seq_len, heads, dim = 1, 16, 8, 64
    keys = torch.randn(batch, seq_len, heads, dim)
    values = torch.randn(batch, seq_len, heads, dim)

    cache.store(0, keys, values, start_pos=0)

    # Retrieve (batch dim is squeezed in store)
    k_out, v_out = cache.get(0, seq_len)
    assert k_out.shape == (seq_len, heads, dim)
    assert v_out.shape == (seq_len, heads, dim)
    assert cache.context_len == seq_len

    # Store more at offset
    keys2 = torch.randn(batch, 8, heads, dim)
    values2 = torch.randn(batch, 8, heads, dim)
    cache.store(0, keys2, values2, start_pos=seq_len)
    assert cache.context_len == 24

    print(f"PASS: test_kv_cache (memory={cache.get_memory_mb():.1f} MB, ctx={cache.context_len})")


def test_moe_expert_cache():
    """Test MoE expert cache hit/miss behavior."""
    from shardfit.moe import ExpertCache, ExpertCacheConfig
    import torch

    config = ExpertCacheConfig(
        num_experts=8,
        experts_per_token=2,
        expert_size_bytes=1024 * 1024,  # 1 MB per expert
        max_cached_experts=3,
        device="cpu",
    )
    cache = ExpertCache(config)

    # Add experts to backing store
    for i in range(8):
        tensor = torch.randn(1024, 1024, dtype=torch.float16)
        cache.add_expert_to_backing(i, tensor)

    # First access = miss
    t1 = cache.get_expert(0)
    assert cache.stats.misses == 1
    assert cache.stats.hits == 0

    # Second access = hit
    t2 = cache.get_expert(0)
    assert cache.stats.misses == 1
    assert cache.stats.hits == 1

    # Access different experts to fill cache
    cache.get_expert(1)
    cache.get_expert(2)
    assert len(cache.cached_experts) == 3

    # Access expert 3 -> evicts LRU (expert 0, which was least recently used)
    cache.get_expert(3)
    assert len(cache.cached_experts) == 3
    assert 0 not in cache.cached_experts  # evicted (LRU)
    assert 3 in cache.cached_experts  # newly added

    stats = cache.get_stats()
    assert stats["hit_rate"] > 0
    print(f"PASS: test_moe_expert_cache (hit_rate={stats['hit_rate']:.2%}, misses={stats['misses']})")


def test_weights_loader():
    """Test weight loader organization into shards."""
    from shardfit.weights import WeightLoader
    import torch
    import tempfile
    import os

    # Create a tiny state dict (float16 to match loader dtype)
    state_dict = {
        "model.embed_tokens.weight": torch.randn(100, 16, dtype=torch.float16),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(16, 16, dtype=torch.float16),
        "model.layers.0.self_attn.k_proj.weight": torch.randn(16, 16, dtype=torch.float16),
        "model.layers.0.self_attn.v_proj.weight": torch.randn(16, 16, dtype=torch.float16),
        "model.layers.0.self_attn.o_proj.weight": torch.randn(16, 16, dtype=torch.float16),
        "model.layers.0.mlp.gate_proj.weight": torch.randn(16, 16, dtype=torch.float16),
        "model.layers.0.mlp.down_proj.weight": torch.randn(16, 16, dtype=torch.float16),
        "model.layers.0.mlp.up_proj.weight": torch.randn(16, 16, dtype=torch.float16),
        "model.layers.1.self_attn.q_proj.weight": torch.randn(16, 16, dtype=torch.float16),
        "model.layers.1.self_attn.k_proj.weight": torch.randn(16, 16, dtype=torch.float16),
        "model.layers.1.self_attn.v_proj.weight": torch.randn(16, 16, dtype=torch.float16),
        "model.layers.1.self_attn.o_proj.weight": torch.randn(16, 16, dtype=torch.float16),
        "model.layers.1.mlp.gate_proj.weight": torch.randn(16, 16, dtype=torch.float16),
        "model.layers.1.mlp.down_proj.weight": torch.randn(16, 16, dtype=torch.float16),
        "model.layers.1.mlp.up_proj.weight": torch.randn(16, 16, dtype=torch.float16),
        "model.lm_head.weight": torch.randn(100, 16, dtype=torch.float16),
        "model.norm.weight": torch.randn(16, dtype=torch.float16),
    }

    # Save to temp file
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, "test_shardfit_weights.safetensors")
    from safetensors.torch import save_file as save_safetensors
    # Ensure tensors are contiguous for safetensors
    contiguous_sd = {k: v.contiguous() for k, v in state_dict.items()}
    save_safetensors(contiguous_sd, temp_path)

    try:
        loader = WeightLoader(temp_path, dtype=torch.float16, device="cpu")
        sharded = loader.load(slot_size_gb=0.001)  # Tiny slot to force sharding

        assert sharded.total_size_gb > 0
        assert len(sharded.shards) >= 1
        assert len(sharded.resident_weights) > 0  # embed, norm, lm_head

        # Verify all params accounted for
        total_in_shards = sum(s.total_size_bytes for s in sharded.shards)
        total_resident = sum(t.element_size() * t.numel() for t in sharded.resident_weights.values())
        expected = sum(t.element_size() * t.numel() for t in state_dict.values())
        assert total_in_shards + total_resident == expected

        print(f"PASS: test_weights_loader ({len(sharded.shards)} shards, "
              f"{len(sharded.resident_weights)} resident, {sharded.total_size_gb:.3f} GB)")
    finally:
        os.unlink(temp_path)


def test_moe_router():
    """Test MoE router expert selection."""
    from shardfit.moe import MoERouter
    import torch

    router = MoERouter(num_experts=8, experts_per_token=2)
    hidden = torch.randn(2, 4, 16)  # batch=2, seq=4, hidden=16

    expert_indices, weights = router.route(hidden)

    assert expert_indices.shape == (2, 4, 2)
    assert weights.shape == (2, 4, 2)
    assert expert_indices.max() < 8  # Valid expert IDs
    print(f"PASS: test_moe_router (experts={expert_indices.shape})")


def main():
    tests = [
        test_vram_budget,
        test_vram_budget_validation,
        test_planner_dense,
        test_planner_14b_no_fit,
        test_sharder,
        test_cpu_backend,
        test_create_backend_auto,
        test_streamer,
        test_kv_cache,
        test_moe_expert_cache,
        test_weights_loader,
        test_moe_router,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAIL: {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 40}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    print(f"{'=' * 40}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
