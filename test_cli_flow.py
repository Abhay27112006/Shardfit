"""Test the CLI chat flow end-to-end with a small model."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from shardfit.engine import ShardFitEngine, GenerationConfig

def main():
    print("=" * 60)
    print("ShardFit CLI Chat Flow Test")
    print("=" * 60)

    # Initialize engine with a tiny model (fast download)
    model_name = "sshleifer/tiny-gpt2"
    print(f"\nLoading model: {model_name}")
    print("Backend: auto (will use CUDA if available, else CPU)")

    try:
        engine = ShardFitEngine(
            model_path=model_name,
            backend_type="auto",
            vram_gb=6.0,
            reserve_gb=1.5,
            num_slots=2,
            slot_size_gb=0.5,
        )
    except Exception as e:
        print(f"[Error] Failed to initialize engine: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\nEngine initialized successfully!")
    print(f"Backend: {'CUDA' if engine.backend.supports_cuda() else 'CPU'}")
    print(f"Device: {engine.backend.device_name}")

    # Load model (triggers sharding, KV cache init, streamer setup)
    print("\nLoading model into engine...")
    engine.load_model()
    print(f"Shards: {len(engine.shards)}")
    print(f"KV cache: {engine.kv_cache.get_memory_mb():.1f} MB")
    print(f"Model format: {engine.sharded_weights.source_format}")

    # Generate a response
    prompt = "Hello, how are you?"
    print(f"\nPrompt: {prompt}")
    print("Generating...")

    config = GenerationConfig(
        max_new_tokens=64,
        temperature=0.7,
        do_sample=True,
    )

    try:
        result = engine.generate(prompt, config=config)
        print(f"\n--- Response ---")
        print(result.generated_text.encode("ascii", errors="replace").decode("ascii"))
        print(f"--- End ---")
        print(f"\nStats:")
        print(f"  Backend: {result.backend}")
        print(f"  Input tokens: {result.input_tokens}")
        print(f"  Output tokens: {result.output_tokens}")
        print(f"  Total time: {result.total_time_s:.2f}s")
        print(f"  Tok/s: {result.tok_s:.2f}")
        print(f"  TTFT: {result.ttft_s:.2f}s")
        print(f"  Lossless: {result.lossless}")

        # Memory stats
        mem = engine.get_memory_usage()
        print(f"\nMemory:")
        print(f"  Backend: {mem['backend_mb']} MB")
        print(f"  KV cache: {mem['kv_cache_mb']:.1f} MB")
        print(f"  CUDA: {mem['supports_cuda']}")

    except Exception as e:
        print(f"[Error] Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n" + "=" * 60)
    print("CLI chat flow test PASSED!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
