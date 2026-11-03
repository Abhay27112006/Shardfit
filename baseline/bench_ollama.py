"""
Phase 0 baseline benchmark for ShardFit.
Measures llama.cpp (via ollama) partial offload: tok/s, time-to-first-token,
prefill time, VRAM usage at three context lengths.

Runs against the ollama REST API (http://127.0.0.1:11434).

Usage:
    python bench_ollama.py --model gemma4:e4b --num-gpu 1
    python bench_ollama.py --model gemma4:e4b --num-gpu 0
"""

import argparse
import json
import time
import subprocess
import sys
import urllib.request
import urllib.error
import csv
import os

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

# Three context lengths per section 11 of the brief
CONTEXT_LENGTHS = [128, 512, 2048]

# Fixed prompt template - we pad to reach target context length
BASE_PROMPT = "The following is a detailed analysis of a software codebase. "


def pad_prompt(target_tokens: int) -> str:
    """Create a prompt of approximately target_tokens length."""
    # Rough estimate: ~0.75 tokens per word for English
    words_needed = int(target_tokens / 0.75)
    return (BASE_PROMPT * (words_needed // len(BASE_PROMPT.split()) + 1))[:target_tokens * 2]


def get_vram_mb() -> int:
    """Return current GPU VRAM usage in MB via nvidia-smi."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            return int(out.stdout.strip().splitlines()[0])
    except Exception:
        pass
    return -1


def benchmark_once(model: str, prompt: str, n_predict: int, num_gpu: int, seed: int = 42):
    """Run a single ollama generate request with streaming and measure metrics."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "num_predict": n_predict,
            "num_gpu": num_gpu,
            "seed": seed,
            "temperature": 0.0,
        },
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.perf_counter()
    first_token_time = None
    tokens_received = 0
    total_duration_ns = None
    eval_duration_ns = None
    eval_count = None
    prompt_eval_count = None
    prompt_eval_duration_ns = None

    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            for line in resp:
                line = line.strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue

                now = time.perf_counter()
                if first_token_time is None and chunk.get("response"):
                    first_token_time = now - t0

                if chunk.get("done"):
                    total_duration_ns = chunk.get("total_duration")
                    eval_duration_ns = chunk.get("eval_duration")
                    eval_count = chunk.get("eval_count")
                    prompt_eval_count = chunk.get("prompt_eval_count")
                    prompt_eval_duration_ns = chunk.get("prompt_eval_duration")
                    break
                else:
                    tokens_received += 1

    except urllib.error.URLError as e:
        print(f"  ERROR: {e}", file=sys.stderr)
        return None

    wall_time = time.perf_counter() - t0
    ttft = first_token_time if first_token_time is not None else wall_time

    # Decode tok/s from ollama's eval_duration (generation time only)
    decode_tok_s = None
    if eval_count and eval_duration_ns:
        decode_tok_s = eval_count / (eval_duration_ns / 1e9)

    # Prefill tok/s from prompt_eval_duration
    prefill_tok_s = None
    if prompt_eval_count and prompt_eval_duration_ns:
        prefill_tok_s = prompt_eval_count / (prompt_eval_duration_ns / 1e9)

    return {
        "ttft_s": round(ttft, 4),
        "wall_time_s": round(wall_time, 4),
        "decode_tok_s": round(decode_tok_s, 3) if decode_tok_s else None,
        "prefill_tok_s": round(prefill_tok_s, 3) if prefill_tok_s else None,
        "tokens_generated": eval_count,
        "prompt_tokens": prompt_eval_count,
        "vram_mb_before": get_vram_mb(),
        "vram_mb_after": get_vram_mb(),
    }


def run_benchmark(model: str, num_gpu: int, n_predict: int = 128, repetitions: int = 3):
    """Run benchmark at three context lengths, averaging over repetitions."""
    results = []

    for ctx in CONTEXT_LENGTHS:
        prompt = pad_prompt(ctx)
        run_results = []

        for rep in range(repetitions):
            print(f"  ctx={ctx} rep={rep + 1}/{repetitions} num_gpu={num_gpu} ...", end=" ")
            sys.stdout.flush()
            r = benchmark_once(model, prompt, n_predict, num_gpu)
            if r:
                r["context_len"] = ctx
                r["num_gpu"] = num_gpu
                r["model"] = model
                r["repetition"] = rep + 1
                run_results.append(r)
                print(f"TTFT={r['ttft_s']}s decode={r['decode_tok_s']}tok/s VRAM={r['vram_mb_after']}MB")
            else:
                print("FAILED")

        if run_results:
            # Average across repetitions
            avg = {
                "model": model,
                "num_gpu": num_gpu,
                "context_len": ctx,
                "repetitions": len(run_results),
                "ttft_s": round(sum(r["ttft_s"] for r in run_results) / len(run_results), 4),
                "decode_tok_s": round(sum(r["decode_tok_s"] for r in run_results if r["decode_tok_s"]) / max(1, sum(1 for r in run_results if r["decode_tok_s"])), 3),
                "prefill_tok_s": round(sum(r["prefill_tok_s"] for r in run_results if r["prefill_tok_s"]) / max(1, sum(1 for r in run_results if r["prefill_tok_s"])), 3),
                "vram_mb_after": round(sum(r["vram_mb_after"] for r in run_results) / len(run_results), 1),
                "tokens_generated": run_results[0]["tokens_generated"],
            }
            results.append(avg)

    return results


def main():
    parser = argparse.ArgumentParser(description="Phase 0 baseline benchmark")
    parser.add_argument("--model", required=True, help="Ollama model name (e.g. gemma4:e4b)")
    parser.add_argument("--num-gpu", type=int, default=1, help="Number of GPU layers (0=CPU only)")
    parser.add_argument("--n-predict", type=int, default=128, help="Tokens to generate")
    parser.add_argument("--repetitions", type=int, default=3, help="Repetitions per context length")
    parser.add_argument("--output", default="C:/Shardfit/baseline/results/phase0_baseline.csv",
                        help="Output CSV file")
    args = parser.parse_args()

    all_results = []

    for num_gpu in [args.num_gpu]:
        print(f"\n=== Benchmarking {args.model} num_gpu={num_gpu} ===")
        results = run_benchmark(args.model, num_gpu, args.n_predict, args.repetitions)
        all_results.extend(results)

    # Write CSV
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fieldnames = ["model", "num_gpu", "context_len", "repetitions",
                  "ttft_s", "decode_tok_s", "prefill_tok_s",
                  "vram_mb_after", "tokens_generated"]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nResults written to {args.output}")
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
