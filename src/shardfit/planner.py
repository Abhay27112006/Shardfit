"""Phase 2: Planner.

Implements the speed prediction formulas from section 6 of the brief.

Given a model file, hardware profile, and context length, the planner predicts
tokens per second and gives a go/no-go before the run starts.

Formulas (section 6):
    t_pass = max(B / L, M / v)
    tok/s  = T / t_pass     (usually ~ T*L / B, since L << v)

    Largest model for speed target tok_min:
    M_max = (W - S*C) + L*T / tok_min

Conditions:
    1. C <= W / S, shard >= one layer or sub-block
    2. If B > H (host RAM pinable), rest streams from SSD
    3. M must fit in RAM + SSD capacity

MoE variant:
    M_act = A_sh + (k/N)*E
    B_tok = (1-h) * (k/N) * E
    tok/s = T / max(B_tok / L, M_act / v)
    E_max = L*T / (tok_min * (1-h) * (k/N))
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Optional

from shardfit.config import HardwareProfile, VRAMBudget


@dataclass
class ModelSpec:
    """Model specification for planning."""

    name: str
    size_gb: float              # M: model size as stored (quantized size)
    quantized: bool = True
    num_layers: int = 32
    hidden_size: int = 4096
    moe: bool = False
    # MoE fields
    shared_weights_gb: float = 0.0   # A_sh
    total_expert_bytes_gb: float = 0.0  # E
    experts_active: int = 0          # k
    experts_total: int = 0           # N


@dataclass
class PlanResult:
    """Planner output."""

    model: str
    hardware: str
    context_len: int

    # Budget
    vram_total_gb: float
    reserve_gb: float
    weights_budget_gb: float       # W
    resident_budget_gb: float      # Pin
    slot_size_gb: float
    num_slots: int

    # Streaming
    streamed_per_pass_gb: float    # B = M - Pin
    effective_bandwidth_gb_s: float  # L
    vram_bandwidth_gb_s: float     # v

    # Timing
    t_pass_s: float
    tokens_per_pass: int           # T (1 for plain decode)
    tok_s: float
    ttft_estimate_s: Optional[float] = None

    # Feasibility
    go_no_go: bool = "no-go"
    reasons: list = None

    def __post_init__(self):
        if self.reasons is None:
            self.reasons = []

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "hardware": self.hardware,
            "context_len": self.context_len,
            "vram_total_gb": self.vram_total_gb,
            "reserve_gb": self.reserve_gb,
            "weights_budget_gb": self.weights_budget_gb,
            "resident_budget_gb": self.resident_budget_gb,
            "slot_size_gb": self.slot_size_gb,
            "num_slots": self.num_slots,
            "streamed_per_pass_gb": self.streamed_per_pass_gb,
            "effective_bandwidth_gb_s": self.effective_bandwidth_gb_s,
            "vram_bandwidth_gb_s": self.vram_bandwidth_gb_s,
            "t_pass_s": self.t_pass_s,
            "tokens_per_pass": self.tokens_per_pass,
            "tok_s": round(self.tok_s, 4),
            "ttft_estimate_s": self.ttft_estimate_s,
            "go_no_go": self.go_no_go,
            "reasons": self.reasons,
        }


class Planner:
    """Predicts tokens per second for a model on a given hardware profile."""

    def __init__(
        self,
        hardware: HardwareProfile,
        budget: VRAMBudget,
    ):
        self.hardware = hardware
        self.budget = budget

    def plan_dense(
        self,
        model: ModelSpec,
        context_len: int = 2048,
        tokens_per_pass: int = 1,
        calibration: float = 0.7,
    ) -> PlanResult:
        """Plan for a dense model.

        calibration: real speed is typically 60-80% of the formula ceiling.
        """
        V = self.budget.vram_total_gb
        R = self.budget.reserve_gb
        W = self.budget.weights_budget_gb
        S = self.budget.num_slots
        C = self.budget.slot_size_gb
        Pin = self.budget.resident_budget_gb

        M = model.size_gb
        B = M - Pin  # bytes streamed per pass

        # Effective bandwidth L: host-to-GPU or VRAM, whichever is limiting
        L = self.hardware.pcie_bandwidth_gb_s   # host-to-GPU
        v = self.hardware.gpu_vram_bandwidth_gb_s  # VRAM bandwidth

        # If using unified memory (Mac), no PCIe copy penalty
        if self.hardware.uses_unified_memory:
            L = v  # streaming from RAM adds nothing

        # If B exceeds host RAM pinable (H), rest streams from SSD
        H = self.hardware.ram_available_gb
        if B > H:
            # t_load = H/L + (B-H)/SSD — effective bandwidth is lower
            ssd_bw = self.hardware.nvme_read_gb_s
            # Effective bandwidth for the streamed portion
            load_time = H / L + (B - H) / ssd_bw
            L_effective = B / load_time if load_time > 0 else L
        else:
            L_effective = L

        # t_pass = max(B / L, M / v)
        t_pass_stream = B / L_effective if L_effective > 0 else float("inf")
        t_pass_compute = M / v if v > 0 else float("inf")
        t_pass = max(t_pass_stream, t_pass_compute)

        # tok/s = T / t_pass (calibrated)
        tok_s_raw = tokens_per_pass / t_pass if t_pass > 0 else 0
        tok_s = tok_s_raw * calibration

        # Feasibility checks
        reasons = []
        go = True

        # Condition 1: C <= W / S
        if C > W / S:
            go = False
            reasons.append(f"Slot size {C} GB > W/S = {W/S:.2f} GB")

        # Condition 2: M must fit in RAM + SSD
        ram_plus_ssd = H + 2.0  # assume at least 2 GB SSD available
        if M > ram_plus_ssd:
            go = False
            reasons.append(
                f"Model {M:.1f} GB exceeds RAM+SSD capacity ({ram_plus_ssd:.1f} GB)"
            )

        # Condition 3: resident shard must fit
        if Pin < 0:
            go = False
            reasons.append("Resident budget is negative — reduce slots or slot size")

        # KV cache consideration (section 14 risk 1)
        # KV cache grows with context; shrink weight budget
        # Rough estimate: KV cache ~ 2 * context * hidden * num_layers * bytes_per_param
        # For now, just note it as a risk
        if context_len > 4096:
            reasons.append(
                f"Long context ({context_len}) increases KV cache, shrinking weight budget"
            )

        # Determine go/no-go
        go_no_go = "go" if go and tok_s > 0 else "no-go"

        # TTFT estimate: prefill time ~ context / prefill_tok_s
        # Prefill processes whole prompt in one pass
        prefill_tok_s = tok_s * 3  # prefill is typically 3x decode
        ttft = context_len / prefill_tok_s if prefill_tok_s > 0 else None

        return PlanResult(
            model=model.name,
            hardware=self.hardware.gpu_name or "CPU",
            context_len=context_len,
            vram_total_gb=V,
            reserve_gb=R,
            weights_budget_gb=W,
            resident_budget_gb=Pin,
            slot_size_gb=C,
            num_slots=S,
            streamed_per_pass_gb=B,
            effective_bandwidth_gb_s=L_effective,
            vram_bandwidth_gb_s=v,
            t_pass_s=t_pass,
            tokens_per_pass=tokens_per_pass,
            tok_s=tok_s,
            ttft_estimate_s=ttft,
            go_no_go=go_no_go,
            reasons=reasons,
        )

    def plan_moe(
        self,
        model: ModelSpec,
        context_len: int = 2048,
        tokens_per_pass: int = 1,
        expert_cache_hit_rate: float = 0.3,
        calibration: float = 0.7,
    ) -> PlanResult:
        """Plan for a MoE model (section 6 MoE variant)."""
        A_sh = model.shared_weights_gb
        E = model.total_expert_bytes_gb
        k = model.experts_active
        N = model.experts_total
        h = expert_cache_hit_rate

        if N == 0:
            raise ValueError("MoE model must have experts_total > 0")

        # M_act = A_sh + (k/N)*E
        M_act = A_sh + (k / N) * E

        # B_tok = (1-h) * (k/N) * E
        B_tok = (1 - h) * (k / N) * E

        L = self.hardware.pcie_bandwidth_gb_s
        v = self.hardware.gpu_vram_bandwidth_gb_s

        if self.hardware.uses_unified_memory:
            L = v

        # tok/s = T / max(B_tok / L, M_act / v)
        t_pass_stream = B_tok / L if L > 0 else float("inf")
        t_pass_compute = M_act / v if v > 0 else float("inf")
        t_pass = max(t_pass_stream, t_pass_compute)

        tok_s_raw = tokens_per_pass / t_pass if t_pass > 0 else 0
        tok_s = tok_s_raw * calibration

        reasons = []
        go = tok_s > 0

        go_no_go = "go" if go else "no-go"

        prefill_tok_s = tok_s * 3
        ttft = context_len / prefill_tok_s if prefill_tok_s > 0 else None

        return PlanResult(
            model=model.name,
            hardware=self.hardware.gpu_name or "CPU",
            context_len=context_len,
            vram_total_gb=self.budget.vram_total_gb,
            reserve_gb=self.budget.reserve_gb,
            weights_budget_gb=self.budget.weights_budget_gb,
            resident_budget_gb=self.budget.resident_budget_gb,
            slot_size_gb=self.budget.slot_size_gb,
            num_slots=self.budget.num_slots,
            streamed_per_pass_gb=B_tok,
            effective_bandwidth_gb_s=L,
            vram_bandwidth_gb_s=v,
            t_pass_s=t_pass,
            tokens_per_pass=tokens_per_pass,
            tok_s=tok_s,
            ttft_estimate_s=ttft,
            go_no_go=go_no_go,
            reasons=reasons,
        )

    def largest_model_for_speed(
        self,
        tok_min: float,
        tokens_per_pass: int = 1,
    ) -> float:
        """M_max = (W - S*C) + L*T / tok_min"""
        W = self.budget.weights_budget_gb
        S = self.budget.num_slots
        C = self.budget.slot_size_gb
        L = self.hardware.pcie_bandwidth_gb_s
        T = tokens_per_pass

        Pin = W - S * C
        M_max = Pin + L * T / tok_min
        return max(0.0, M_max)


def print_plan(result: PlanResult) -> None:
    """Pretty-print a plan result."""
    print("\n" + "=" * 60)
    print("ShardFit Phase 2: Planner")
    print("=" * 60)
    print(f"Model:        {result.model}")
    print(f"Hardware:     {result.hardware}")
    print(f"Context:      {result.context_len} tokens")
    print()
    print("VRAM Budget:")
    print(f"  Total VRAM:      {result.vram_total_gb:.2f} GB")
    print(f"  Reserved (R):    {result.reserve_gb:.2f} GB")
    print(f"  Weights (W):     {result.weights_budget_gb:.2f} GB")
    print(f"  Resident (Pin):  {result.resident_budget_gb:.2f} GB")
    print(f"  Slots (S):       {result.num_slots} x {result.slot_size_gb} GB")
    print()
    print("Streaming:")
    print(f"  Streamed/pass:   {result.streamed_per_pass_gb:.2f} GB")
    print(f"  Bandwidth (L):   {result.effective_bandwidth_gb_s:.2f} GB/s")
    print(f"  VRAM bw (v):     {result.vram_bandwidth_gb_s:.2f} GB/s")
    print()
    print("Performance:")
    print(f"  t_pass:          {result.t_pass_s:.4f} s")
    print(f"  tokens/pass:     {result.tokens_per_pass}")
    print(f"  tok/s:           {result.tok_s:.3f}")
    if result.ttft_estimate_s:
        print(f"  TTFT est:        {result.ttft_estimate_s:.2f} s")
    print()
    print(f"Verdict:         {result.go_no_go.upper()}")
    if result.reasons:
        print("  Issues:")
        for r in result.reasons:
            print(f"    - {r}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="ShardFit Phase 2: Planner")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--size-gb", type=float, required=True, help="Model size in GB")
    parser.add_argument("--gpu", default="rtx3050", help="GPU name (for display)")
    parser.add_argument(
        "--context", type=int, default=2048, help="Context length in tokens"
    )
    parser.add_argument(
        "--vram-gb", type=float, default=6.0, help="Total VRAM in GB"
    )
    parser.add_argument(
        "--reserve-gb", type=float, default=1.5, help="Reserved VRAM in GB"
    )
    parser.add_argument(
        "--slots", type=int, default=2, help="Number of streaming slots"
    )
    parser.add_argument(
        "--slot-gb", type=float, default=0.5, help="Slot size in GB"
    )
    parser.add_argument(
        "--calibration", type=float, default=0.7, help="Speed calibration factor"
    )
    parser.add_argument(
        "--output", help="Write plan as JSON to file"
    )
    args = parser.parse_args()

    from shardfit.config import HardwareProfile, VRAMBudget

    hardware = HardwareProfile(
        gpu_name=args.gpu,
        gpu_vram_total_gb=args.vram_gb,
        gpu_vram_bandwidth_gb_s=168.0,
        pcie_bandwidth_gb_s=12.0,
        cuda_available=True,
        ram_total_gb=16.0,
        ram_available_gb=8.0,
        nvme_read_gb_s=3.0,
        os_name="Windows",
        architecture="x86_64",
    )

    budget = VRAMBudget.calculate(
        vram_total_gb=args.vram_gb,
        reserve_gb=args.reserve_gb,
        num_slots=args.slots,
        slot_size_gb=args.slot_gb,
    )

    errors = budget.validate()
    if errors:
        print("Budget validation errors:")
        for e in errors:
            print(f"  - {e}")
        return 1

    model = ModelSpec(name=args.model, size_gb=args.size_gb)
    planner = Planner(hardware=hardware, budget=budget)

    result = planner.plan_dense(
        model=model,
        context_len=args.context,
        calibration=args.calibration,
    )

    print_plan(result)

    if args.output:
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"\nPlan written to {args.output}")

    return 0 if result.go_no_go == "go" else 1


if __name__ == "__main__":
    sys_exit = main()
    if sys_exit is not None:
        raise SystemExit(sys_exit)
