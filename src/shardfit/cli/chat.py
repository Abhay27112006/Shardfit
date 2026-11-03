"""ShardFit CLI — interactive chat interface.

Demonstrates the engine by chatting with a streamed model.
Supports both CUDA and CPU backends on Windows and Mac.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

from shardfit.engine import ShardFitEngine, GenerationConfig

console = Console() if HAS_RICH else None


def print_msg(text: str, style: str = "white") -> None:
    """Print a message to the console."""
    if HAS_RICH:
        console.print(text, style=style)
    else:
        print(text)


def print_panel(text: str, title: str = "", style: str = "blue") -> None:
    """Print text in a panel."""
    if HAS_RICH:
        console.print(Panel(text, title=title, border_style=style))
    else:
        if title:
            print(f"--- {title} ---")
        print(text)
        if title:
            print("")


def main() -> int:
    parser = argparse.ArgumentParser(description="ShardFit CLI chat")
    parser.add_argument(
        "--model",
        required=True,
        help="Path or HF name of the model (e.g., models/llama-3.1-8b-Q4_K_M)",
    )
    parser.add_argument(
        "--backend",
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="Compute backend (default: auto)",
    )
    parser.add_argument(
        "--vram-gb",
        type=float,
        default=6.0,
        help="Total VRAM in GB (default: 6.0)",
    )
    parser.add_argument(
        "--reserve-gb",
        type=float,
        default=1.5,
        help="Reserved VRAM in GB (default: 1.5)",
    )
    parser.add_argument(
        "--slots",
        type=int,
        default=2,
        help="Number of streaming slots (default: 2)",
    )
    parser.add_argument(
        "--slot-gb",
        type=float,
        default=0.5,
        help="Slot size in GB (default: 0.5)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Max new tokens per response (default: 128)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=None,
        help="CPU threads (CPU backend only)",
    )
    args = parser.parse_args()

    # Print banner
    print_panel(
        "ShardFit CLI v0.1.0\n"
        "Run bigger models than your GPU can hold, unattended.",
        title="ShardFit",
        style="cyan",
    )

    # Show configuration
    config_lines = [
        f"Model: {args.model}",
        f"Backend: {args.backend}",
        f"VRAM: {args.vram_gb} GB (reserve {args.reserve_gb} GB)",
        f"Slots: {args.slots} x {args.slot_gb} GB",
        f"Max tokens: {args.max_tokens}",
        f"Temperature: {args.temperature}",
    ]
    if args.backend == "cpu" or args.backend == "auto":
        config_lines.append(f"CPU threads: {args.num_threads or 'default'}")
    print_panel("\n".join(config_lines), title="Configuration", style="dim")

    print_msg("")
    print_msg("Type your message (or 'quit' / 'exit' to quit)", style="green")
    print_msg("")

    # Initialize engine
    try:
        engine = ShardFitEngine(
            model_path=args.model,
            backend_type=args.backend,
            vram_gb=args.vram_gb,
            reserve_gb=args.reserve_gb,
            num_slots=args.slots,
            slot_size_gb=args.slot_gb,
            num_threads=args.num_threads,
        )
    except Exception as e:
        print_msg(f"[Error] Failed to initialize engine: {e}", style="red")
        return 1

    # Generation config
    gen_config = GenerationConfig(
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        do_sample=True,
    )

    # Chat loop
    while True:
        try:
            if HAS_RICH:
                user_input = Prompt.ask("> ")
            else:
                user_input = input("> ")
        except (EOFError, KeyboardInterrupt):
            print_msg("\nGoodbye!", style="yellow")
            break

        user_input = user_input.strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print_msg("Goodbye!", style="yellow")
            break
        if user_input.lower() == "help":
            print_panel(
                "Commands:\n"
                "  quit / exit / q  - Exit the chat\n"
                "  help             - Show this help\n"
                "  stats            - Show engine memory usage\n"
                "  clear            - Clear screen",
                title="Help",
                style="yellow",
            )
            continue
        if user_input.lower() == "stats":
            mem = engine.get_memory_usage()
            print_panel(
                f"Backend memory: {mem['backend_mb']} MB\n"
                f"CUDA: {mem['supports_cuda']}",
                title="Engine Stats",
                style="blue",
            )
            continue
        if user_input.lower() == "clear":
            os.system("cls" if sys.platform == "win32" else "clear")
            continue

        # Generate response
        try:
            result = engine.generate(user_input, config=gen_config)
            print_panel(
                result.generated_text,
                title=f"ShardFit ({result.backend}, {result.tok_s:.1f} tok/s, "
                      f"TTFT={result.ttft_s:.1f}s)",
                style="green",
            )
        except Exception as e:
            print_msg(f"[Error] Generation failed: {e}", style="red")

    return 0


if __name__ == "__main__":
    import os
    sys.exit(main())
