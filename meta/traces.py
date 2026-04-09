"""Trace collection, indexing, and differential analysis.

The #1 insight from Meta-Harness: full trace access yields 50.0% accuracy
vs 34.6% for scores-only. This module stores full execution traces and
provides indexed access + differential views for the proposer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TraceStep:
    """One step in an agent execution trace."""

    step_id: int
    timestamp: str
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    command: str | None = None
    output: str | None = None
    llm_analysis: str | None = None
    llm_plan: str | None = None
    error: str | None = None
    tokens_used: int = 0
    duration_sec: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TraceStep:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class TaskTrace:
    """Full execution trace for one task + one trial."""

    task_id: str
    trial: int
    variant_id: str
    passed: bool
    score: float | None = None
    steps: list[TraceStep] = field(default_factory=list)
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    total_duration_sec: float = 0.0
    error_summary: str | None = None
    failure_step: int | None = None  # Step where things went wrong

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    def last_commands(self, n: int = 5) -> list[str]:
        """Return the last N commands executed."""
        cmds = [s.command for s in self.steps if s.command]
        return cmds[-n:]

    def failure_context(self, window: int = 3) -> list[TraceStep]:
        """Return steps around the failure point."""
        if self.failure_step is None:
            return self.steps[-window:]
        start = max(0, self.failure_step - window)
        end = min(len(self.steps), self.failure_step + window + 1)
        return self.steps[start:end]

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "steps"}
        d["steps"] = [s.to_dict() for s in self.steps]
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskTrace:
        steps = [TraceStep.from_dict(s) for s in data.pop("steps", [])]
        return cls(steps=steps, **{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class TaskResult:
    """Aggregated result for one task across multiple trials."""

    task_id: str
    variant_id: str
    split: str  # "train" | "holdout"
    difficulty: str | None = None
    stratum: str | None = None
    trials: list[TaskTrace] = field(default_factory=list)

    @property
    def n_trials(self) -> int:
        return len(self.trials)

    @property
    def pass_rate(self) -> float:
        if not self.trials:
            return 0.0
        return sum(1 for t in self.trials if t.passed) / len(self.trials)

    @property
    def passed(self) -> bool:
        """Passed if majority of trials pass."""
        return self.pass_rate > 0.5

    @property
    def is_flaky(self) -> bool:
        """Task is flaky if it passes some trials but not all."""
        rates = [t.passed for t in self.trials]
        return len(set(rates)) > 1

    def passing_trace(self) -> TaskTrace | None:
        """Return a passing trace (for differential analysis)."""
        for t in self.trials:
            if t.passed:
                return t
        return None

    def failing_trace(self) -> TaskTrace | None:
        """Return a failing trace (for differential analysis)."""
        for t in self.trials:
            if not t.passed:
                return t
        return None


@dataclass
class EvalResult:
    """Full evaluation result for a variant across all tasks."""

    variant_id: str
    split: str
    task_results: list[TaskResult] = field(default_factory=list)

    @property
    def pass_count(self) -> int:
        return sum(1 for t in self.task_results if t.passed)

    @property
    def total_count(self) -> int:
        return len(self.task_results)

    @property
    def pass_rate(self) -> float:
        if not self.task_results:
            return 0.0
        return self.pass_count / self.total_count

    def by_difficulty(self) -> dict[str, float]:
        """Pass rate grouped by difficulty."""
        groups: dict[str, list[TaskResult]] = {}
        for tr in self.task_results:
            key = tr.difficulty or "unknown"
            groups.setdefault(key, []).append(tr)
        return {k: sum(1 for t in v if t.passed) / len(v) for k, v in groups.items()}

    def by_stratum(self) -> dict[str, float]:
        """Pass rate grouped by stratum/category."""
        groups: dict[str, list[TaskResult]] = {}
        for tr in self.task_results:
            key = tr.stratum or "unknown"
            groups.setdefault(key, []).append(tr)
        return {k: sum(1 for t in v if t.passed) / len(v) for k, v in groups.items()}

    def failing_tasks(self) -> list[TaskResult]:
        return [t for t in self.task_results if not t.passed]

    def regressions_vs(self, other: EvalResult) -> list[str]:
        """Tasks that pass in other but fail in self."""
        other_passing = {t.task_id for t in other.task_results if t.passed}
        self_failing = {t.task_id for t in self.task_results if not t.passed}
        return sorted(other_passing & self_failing)

    def improvements_vs(self, other: EvalResult) -> list[str]:
        """Tasks that fail in other but pass in self."""
        other_failing = {t.task_id for t in other.task_results if not t.passed}
        self_passing = {t.task_id for t in self.task_results if t.passed}
        return sorted(other_failing & self_passing)


class TraceStore:
    """Persistent store for execution traces with indexed access.

    Directory layout:
        store_root/
            index.json              # Task -> pass/fail summary
            variants/
                baseline/
                    train/
                        task_001_trial_0.json
                        task_001_trial_1.json
                    holdout/
                        ...
                iter_001/
                    ...
    """

    def __init__(self, store_root: Path):
        self.root = store_root
        self.root.mkdir(parents=True, exist_ok=True)
        self._index: dict[str, dict] = {}
        self._load_index()

    def _index_path(self) -> Path:
        return self.root / "index.json"

    def _load_index(self) -> None:
        path = self._index_path()
        if path.exists():
            self._index = json.loads(path.read_text())

    def _save_index(self) -> None:
        self._index_path().write_text(json.dumps(self._index, indent=2))

    def store_trace(self, trace: TaskTrace, split: str) -> Path:
        """Store a single task trace and update the index."""
        variant_dir = self.root / "variants" / trace.variant_id / split
        variant_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{trace.task_id}_trial_{trace.trial}.json"
        path = variant_dir / filename
        path.write_text(json.dumps(trace.to_dict(), indent=2))

        # Update index
        key = f"{trace.variant_id}/{split}/{trace.task_id}"
        if key not in self._index:
            self._index[key] = {
                "task_id": trace.task_id,
                "variant_id": trace.variant_id,
                "split": split,
                "trials": [],
            }
        self._index[key]["trials"].append({
            "trial": trace.trial,
            "passed": trace.passed,
            "n_steps": trace.n_steps,
            "error_summary": trace.error_summary,
            "failure_step": trace.failure_step,
        })
        self._save_index()
        return path

    def store_eval_result(self, result: EvalResult) -> None:
        """Store all traces from an evaluation result."""
        for task_result in result.task_results:
            for trace in task_result.trials:
                self.store_trace(trace, result.split)

    def load_trace(self, variant_id: str, split: str, task_id: str, trial: int) -> TaskTrace | None:
        """Load a specific trace."""
        path = self.root / "variants" / variant_id / split / f"{task_id}_trial_{trial}.json"
        if not path.exists():
            return None
        return TaskTrace.from_dict(json.loads(path.read_text()))

    def get_index(self, variant_id: str | None = None, split: str | None = None) -> dict:
        """Get the trace index, optionally filtered."""
        if variant_id is None and split is None:
            return self._index
        return {
            k: v for k, v in self._index.items()
            if (variant_id is None or v["variant_id"] == variant_id)
            and (split is None or v["split"] == split)
        }

    def get_failing_tasks(self, variant_id: str, split: str) -> list[dict]:
        """Get summary of failing tasks for a variant."""
        failing = []
        for key, entry in self._index.items():
            if entry["variant_id"] != variant_id or entry["split"] != split:
                continue
            trials = entry["trials"]
            pass_rate = sum(1 for t in trials if t["passed"]) / len(trials) if trials else 0
            if pass_rate < 0.5:
                failing.append({
                    "task_id": entry["task_id"],
                    "pass_rate": pass_rate,
                    "n_trials": len(trials),
                    "error_summaries": [t.get("error_summary") for t in trials if not t["passed"]],
                    "failure_steps": [t.get("failure_step") for t in trials if not t["passed"]],
                })
        return failing

    def get_differential(self, task_id: str, variant_id: str, split: str) -> dict | None:
        """Get a differential view: passing vs failing trace for same task.

        This is the key diagnostic tool — shows the proposer exactly where
        a flaky task diverges between success and failure.
        """
        key = f"{variant_id}/{split}/{task_id}"
        entry = self._index.get(key)
        if not entry:
            return None

        passing_trial = None
        failing_trial = None
        for t in entry["trials"]:
            if t["passed"] and passing_trial is None:
                passing_trial = t["trial"]
            if not t["passed"] and failing_trial is None:
                failing_trial = t["trial"]

        result: dict[str, Any] = {"task_id": task_id}

        if passing_trial is not None:
            trace = self.load_trace(variant_id, split, task_id, passing_trial)
            if trace:
                result["passing"] = {
                    "n_steps": trace.n_steps,
                    "last_commands": trace.last_commands(),
                    "total_tokens": trace.total_tokens,
                }

        if failing_trial is not None:
            trace = self.load_trace(variant_id, split, task_id, failing_trial)
            if trace:
                result["failing"] = {
                    "n_steps": trace.n_steps,
                    "last_commands": trace.last_commands(),
                    "failure_step": trace.failure_step,
                    "error_summary": trace.error_summary,
                    "failure_context": [s.to_dict() for s in trace.failure_context()],
                    "total_tokens": trace.total_tokens,
                }

        return result


def parse_harbor_trajectory(trajectory_path: Path, task_id: str, trial: int, variant_id: str) -> TaskTrace:
    """Parse a Harbor trajectory JSON file into a TaskTrace.

    Harbor trajectories follow the format used by the KIRA agent,
    with Step objects containing tool_calls and observations.
    """
    data = json.loads(trajectory_path.read_text())

    steps = []
    for step_data in data.get("steps", []):
        tool_name = None
        tool_args = None
        command = None

        tool_calls = step_data.get("tool_calls", [])
        if tool_calls:
            tc = tool_calls[0]
            tool_name = tc.get("function_name", "")
            tool_args = tc.get("arguments", {})
            if tool_name == "bash_command":
                command = tool_args.get("keystrokes", "")

        observation = ""
        obs = step_data.get("observation", {})
        if obs:
            results = obs.get("results", [])
            if results:
                observation = results[0].get("content", "")

        metrics = step_data.get("metrics", {})

        steps.append(TraceStep(
            step_id=step_data.get("step_id", len(steps) + 1),
            timestamp=step_data.get("timestamp", ""),
            tool_name=tool_name,
            tool_args=tool_args,
            command=command,
            output=observation[:5000] if observation else None,  # Limit stored output
            llm_analysis=None,  # Could parse from message
            llm_plan=None,
            error=observation[:2000] if observation and "error" in observation.lower() else None,
            tokens_used=metrics.get("prompt_tokens", 0) + metrics.get("completion_tokens", 0),
        ))

    # Determine pass/fail from result
    passed = data.get("passed", False)
    score = data.get("score")

    # Find failure step (last step with error, or last step)
    failure_step = None
    if not passed and steps:
        for i, s in enumerate(reversed(steps)):
            if s.error:
                failure_step = len(steps) - 1 - i
                break
        if failure_step is None:
            failure_step = len(steps) - 1

    # Error summary
    error_summary = None
    if not passed and steps:
        last_errors = [s.error for s in steps if s.error]
        if last_errors:
            error_summary = last_errors[-1][:500]
        else:
            error_summary = f"Failed after {len(steps)} steps (no explicit error)"

    total_tokens = sum(s.tokens_used for s in steps)

    return TaskTrace(
        task_id=task_id,
        trial=trial,
        variant_id=variant_id,
        passed=passed,
        score=score,
        steps=steps,
        total_tokens=total_tokens,
        error_summary=error_summary,
        failure_step=failure_step,
    )
