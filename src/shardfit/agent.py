"""Unattended agent runner (Phase 6 specification, section 13 of the brief).

A separate component that drives the engine through a task loop:
- Task input: goal, allowed tools, working directory, budgets
- Loop: plan -> act -> observe -> repeat (no human prompts during run)
- Resumable: checkpoint after every step
- Budgets: hard limits on steps, tokens, wall-clock time
- Sandboxed: container, no network by default, file access limited to working dir
- Logged: full transcript
- Result bundle: final output, artifacts, step log, run report
- Human review gate: proposes changes, doesn't push/deploy/send/delete without confirmation
- Failure handling: detect loops, stop rather than burn hours
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from shardfit.engine import ShardFitEngine, GenerationConfig, GenerationResult


@dataclass
class AgentStep:
    """A single step in the agent loop."""

    step_id: str
    step_number: int
    plan: str
    action: str
    observation: str
    complete: bool
    timestamp: str
    tokens_used: int = 0
    time_s: float = 0.0


@dataclass
class AgentBudget:
    """Hard limits for an agent run."""

    max_steps: int = 50
    max_tokens: int = 100000
    max_wall_clock_s: float = 3600.0  # 1 hour


@dataclass
class AgentResult:
    """Final result bundle from an agent run."""

    task_id: str
    goal: str
    status: str  # "complete", "budget_exceeded", "failed", "loop_detected"
    steps: list[AgentStep] = field(default_factory=list)
    total_tokens: int = 0
    total_time_s: float = 0.0
    final_output: Optional[str] = None
    artifacts: list[str] = field(default_factory=list)
    run_report: dict = field(default_factory=dict)
    checkpoint_path: Optional[str] = None


class AgentRunner:
    """Unattended agent runner that drives ShardFit through task loops."""

    def __init__(
        self,
        engine: ShardFitEngine,
        work_dir: str = "./agent_workspace",
        budget: Optional[AgentBudget] = None,
        checkpoint_interval: int = 1,  # checkpoint after every step
    ):
        self.engine = engine
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.budget = budget or AgentBudget()
        self.checkpoint_interval = checkpoint_interval

        self.task_id = str(uuid.uuid4())
        self.steps: list[AgentStep] = []
        self.total_tokens = 0
        self.start_time: Optional[float] = None
        self.status = "running"
        self.loop_detected = False

        # Sandbox: no network by default
        self.allow_network = False
        self.allow_file_outside_workdir = False

    def run(
        self,
        goal: str,
        allowed_tools: Optional[list[str]] = None,
        human_review: bool = True,
    ) -> AgentResult:
        """Run the agent loop until complete or budget exhausted.

        Loop: plan -> act -> observe -> repeat
        """
        self.start_time = time.perf_counter()
        allowed_tools = allowed_tools or ["read", "write", "grep", "edit"]

        print(f"[Agent] Task {self.task_id[:8]}: {goal}")
        print(f"[Agent] Budget: {self.budget.max_steps} steps, "
              f"{self.budget.max_tokens} tokens, {self.budget.max_wall_clock_s:.0f}s")
        print(f"[Agent] Human review: {'on' if human_review else 'off'}")
        print(f"[Agent] Sandbox: network={'on' if self.allow_network else 'off'}, "
              f"workdir-only={self.allow_file_outside_workdir}")

        step_number = 0

        while self.status == "running":
            step_number += 1

            # Budget check
            if step_number > self.budget.max_steps:
                self.status = "budget_exceeded"
                print(f"[Agent] Step budget exceeded ({self.budget.max_steps})")
                break

            elapsed = time.perf_counter() - self.start_time
            if elapsed > self.budget.max_wall_clock_s:
                self.status = "budget_exceeded"
                print(f"[Agent] Wall-clock budget exceeded ({self.budget.max_wall_clock_s:.0f}s)")
                break

            if self.total_tokens > self.budget.max_tokens:
                self.status = "budget_exceeded"
                print(f"[Agent] Token budget exceeded ({self.budget.max_tokens})")
                break

            # Loop detection: stop if same action repeats
            if self._detect_loop():
                self.status = "loop_detected"
                self.loop_detected = True
                print(f"[Agent] Loop detected — stopping to avoid burning hours")
                break

            # Plan
            plan = self._plan(goal, step_number)

            # Act
            action_result = self._act(plan, allowed_tools)

            # Observe
            observation = self._observe(action_result)
            complete = observation.get("complete", False)

            # Human review gate
            if human_review and action_result.get("modifies_files", False):
                print(f"[Agent] Human review required for step {step_number}")
                print(f"  Proposed: {action_result.get('description', '')}")
                # In a real implementation, this would pause for confirmation
                print(f"  [Auto-approving for demo]")

            step = AgentStep(
                step_id=str(uuid.uuid4()),
                step_number=step_number,
                plan=plan,
                action=action_result.get("description", ""),
                observation=observation.get("summary", ""),
                complete=complete,
                timestamp=datetime.now().isoformat(),
                tokens_used=action_result.get("tokens", 0),
                time_s=time.perf_counter() - self.start_time,
            )
            self.steps.append(step)
            self.total_tokens += step.tokens_used

            print(f"  Step {step_number}: {step.action[:60]}... "
                  f"({step.observation[:40]}...)")

            if complete:
                self.status = "complete"
                break

            # Checkpoint after every step (resumable)
            if step_number % self.checkpoint_interval == 0:
                self._checkpoint()

        # Build result bundle
        total_time = time.perf_counter() - self.start_time if self.start_time else 0
        result = AgentResult(
            task_id=self.task_id,
            goal=goal,
            status=self.status,
            steps=self.steps,
            total_tokens=self.total_tokens,
            total_time_s=total_time,
            final_output=self.steps[-1].observation if self.steps else None,
            run_report={
                "engine_backend": "cuda" if self.engine.backend.supports_cuda() else "cpu",
                "engine_model": self.engine.model_path,
                "steps_completed": len(self.steps),
                "loop_detected": self.loop_detected,
            },
        )

        # Save result bundle
        self._save_result_bundle(result)

        print(f"\n[Agent] Task {self.status} in {total_time:.1f}s, "
              f"{len(self.steps)} steps, {self.total_tokens} tokens")
        print(f"[Agent] Result bundle: {self.work_dir / f'result_{self.task_id}.json'}")

        return result

    def _plan(self, goal: str, step_number: int) -> str:
        """Plan the next action using the model."""
        prompt = (
            f"Goal: {goal}\n"
            f"Step {step_number}: What is the next action to take? "
            f"Be concise. Allowed tools: read, write, grep, edit.\n"
            f"Action:"
        )

        result = self.engine.generate(prompt, GenerationConfig(max_new_tokens=64, temperature=0.3))
        return result.generated_text.strip()

    def _act(self, plan: str, allowed_tools: list[str]) -> dict:
        """Execute the planned action (sandboxed)."""
        # In a real implementation, this would execute tool calls
        # For now, simulate an action
        return {
            "description": plan[:100],
            "tokens": 32,
            "modifies_files": "write" in plan.lower() or "edit" in plan.lower(),
            "success": True,
        }

    def _observe(self, action_result: dict) -> dict:
        """Observe the result of an action."""
        return {
            "summary": f"Action completed: {action_result.get('description', '')[:50]}",
            "complete": False,  # Would be determined by task evaluation
            "success": action_result.get("success", False),
        }

    def _detect_loop(self) -> bool:
        """Detect repeated failed steps (loop detection)."""
        if len(self.steps) < 4:
            return False

        # Check if last 3 actions are identical
        recent = [s.action for s in self.steps[-3:]]
        if len(set(recent)) == 1:
            return True

        # Check if last 3 observations are all failures
        recent_obs = [s.observation for s in self.steps[-3:]]
        if all("fail" in o.lower() for o in recent_obs):
            return True

        return False

    def _checkpoint(self) -> None:
        """Save checkpoint for resumability."""
        checkpoint = {
            "task_id": self.task_id,
            "status": self.status,
            "step_number": len(self.steps),
            "total_tokens": self.total_tokens,
            "steps": [asdict(s) for s in self.steps],
            "timestamp": datetime.now().isoformat(),
        }
        path = self.work_dir / f"checkpoint_{self.task_id}.json"
        with open(path, "w") as f:
            json.dump(checkpoint, f, indent=2)
        self.steps[-1].__dict__["checkpoint_path"] = str(path)

    def _save_result_bundle(self, result: AgentResult) -> None:
        """Save the final result bundle."""
        path = self.work_dir / f"result_{self.task_id}.json"
        with open(path, "w") as f:
            json.dump(asdict(result), f, indent=2)
        result.checkpoint_path = str(path)

    def resume(self, checkpoint_path: str) -> AgentResult:
        """Resume from a checkpoint (for power loss recovery)."""
        with open(checkpoint_path, "r") as f:
            data = json.load(f)

        self.task_id = data["task_id"]
        self.status = data["status"]
        self.total_tokens = data["total_tokens"]
        self.steps = [AgentStep(**s) for s in data["steps"]]

        print(f"[Agent] Resumed task {self.task_id[:8]} at step {len(self.steps)}")
        # Continue running...
        return self.run(goal=result.go if (result := None) else "resumed task")
