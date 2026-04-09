"""Runner integration for evaluating harness variants.

Executes the inner agent against tasks via Harbor and collects
full execution traces. Supports incremental evaluation (only re-run
failing tasks + canaries) for efficient search iterations.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .surfaces import Surface, Variant, apply_variant_to_harness
from .traces import (
    EvalResult,
    TaskResult,
    TaskTrace,
    TraceStep,
    TraceStore,
)


@dataclass
class TaskCase:
    """One task assignment for evaluation."""

    task_id: str
    split: str  # "train" | "holdout" | "all"
    difficulty: str | None = None
    stratum: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__


@dataclass
class RunConfig:
    """Configuration for a single evaluation run."""

    variant: Variant
    tasks: list[TaskCase]
    n_trials: int = 1
    model: str = "anthropic/claude-opus-4-6"
    environment: str = "runloop"
    max_episodes: int = 20
    concurrency: int = 20
    timeout_multiplier: float = 1.0


def _parse_trial_result(trial_dir: Path, task_id: str, trial: int, variant_id: str) -> TaskTrace:
    """Parse a Harbor trial directory into a TaskTrace.

    Harbor trial layout:
        trial_dir/
            result.json         # TrialResult with rewards, timing, errors
            agent/
                trajectory.json # ATIF v1.6 execution trace
            verifier/
                reward.txt      # Simple float reward
                test-stdout.txt
                test-stderr.txt
            exception.txt       # If error occurred
    """
    result_path = trial_dir / "result.json"
    trajectory_path = trial_dir / "agent" / "trajectory.json"
    reward_path = trial_dir / "verifier" / "reward.txt"
    exception_path = trial_dir / "exception.txt"

    # Determine pass/fail from result.json
    passed = False
    score = None
    error_summary = None
    total_tokens = 0
    total_cost = 0.0

    if result_path.exists():
        try:
            result_data = json.loads(result_path.read_text())

            # Extract reward from verifier_result
            vr = result_data.get("verifier_result") or {}
            rewards = vr.get("rewards") or {}
            if rewards:
                # TerminalBench uses "reward" key, value 1.0 = pass
                score = rewards.get("reward", rewards.get(next(iter(rewards))))
                passed = score is not None and float(score) >= 1.0

            # Extract token usage from agent_result
            ar = result_data.get("agent_result") or {}
            total_tokens = (ar.get("n_input_tokens") or 0) + (ar.get("n_output_tokens") or 0)
            total_cost = ar.get("cost_usd") or 0.0

            # Check for exceptions
            ei = result_data.get("exception_info")
            if ei:
                error_summary = f"{ei.get('exception_type', 'Error')}: {ei.get('exception_message', '')[:500]}"
        except (json.JSONDecodeError, KeyError) as e:
            error_summary = f"Failed to parse result.json: {e}"

    elif reward_path.exists():
        # Fallback: read reward.txt directly
        try:
            score = float(reward_path.read_text().strip())
            passed = score >= 1.0
        except ValueError:
            error_summary = "Invalid reward.txt"

    # Read exception file if exists
    if exception_path.exists() and not error_summary:
        error_summary = exception_path.read_text()[:500]

    # Parse full trajectory if available
    steps = []
    failure_step = None
    if trajectory_path.exists():
        try:
            traj_data = json.loads(trajectory_path.read_text())
            for i, step_data in enumerate(traj_data.get("steps", [])):
                tool_name = None
                command = None
                output = None
                error = None

                # Extract tool call info
                tool_calls = step_data.get("tool_calls") or []
                if tool_calls:
                    tc = tool_calls[0]
                    tool_name = tc.get("function_name", "")
                    args = tc.get("arguments") or {}
                    if tool_name in ("bash_command", "execute_commands"):
                        command = args.get("keystrokes", args.get("command", ""))

                # Extract observation
                obs = step_data.get("observation") or {}
                results = obs.get("results") or []
                if results:
                    output = results[0].get("content", "")[:5000]
                    if output and "error" in output.lower():
                        error = output[:2000]

                # Metrics
                metrics = step_data.get("metrics") or {}
                tokens = (metrics.get("prompt_tokens") or 0) + (metrics.get("completion_tokens") or 0)

                steps.append(TraceStep(
                    step_id=step_data.get("step_id", i + 1),
                    timestamp=step_data.get("timestamp", ""),
                    tool_name=tool_name,
                    tool_args=tc.get("arguments") if tool_calls else None,
                    command=command,
                    output=output,
                    llm_analysis=step_data.get("message", "")[:2000] if step_data.get("message") else None,
                    error=error,
                    tokens_used=tokens,
                ))
        except (json.JSONDecodeError, KeyError):
            pass

    # Find failure step
    if not passed and steps:
        for i in range(len(steps) - 1, -1, -1):
            if steps[i].error:
                failure_step = i
                break
        if failure_step is None:
            failure_step = len(steps) - 1

    if not passed and not error_summary and steps:
        last_errors = [s.error for s in steps if s.error]
        error_summary = last_errors[-1][:500] if last_errors else f"Failed after {len(steps)} steps"

    return TaskTrace(
        task_id=task_id,
        trial=trial,
        variant_id=variant_id,
        passed=passed,
        score=score,
        steps=steps,
        total_tokens=total_tokens,
        total_cost_usd=total_cost,
        error_summary=error_summary,
        failure_step=failure_step,
    )


def _parse_job_results(job_dir: Path, variant_id: str, task_map: dict[str, TaskCase]) -> EvalResult:
    """Parse all trial results from a Harbor job directory.

    Harbor job layout:
        job_dir/
            config.json
            result.json     # Aggregated job result
            {trial_name}/   # One dir per trial
                result.json
                agent/trajectory.json
                ...
    """
    task_results: dict[str, TaskResult] = {}

    # Each subdirectory is a trial
    for trial_dir in sorted(job_dir.iterdir()):
        if not trial_dir.is_dir():
            continue
        result_path = trial_dir / "result.json"
        if not result_path.exists():
            continue

        try:
            result_data = json.loads(result_path.read_text())
            task_name = result_data.get("task_name", trial_dir.name)
        except (json.JSONDecodeError, KeyError):
            continue

        # Determine trial number from trial name or order
        if task_name not in task_results:
            task_case = task_map.get(task_name)
            task_results[task_name] = TaskResult(
                task_id=task_name,
                variant_id=variant_id,
                split=task_case.split if task_case else "all",
                difficulty=task_case.difficulty if task_case else None,
                stratum=task_case.stratum if task_case else None,
            )

        trial_num = len(task_results[task_name].trials)
        trace = _parse_trial_result(trial_dir, task_name, trial_num, variant_id)
        task_results[task_name].trials.append(trace)

    return EvalResult(
        variant_id=variant_id,
        split="all",
        task_results=list(task_results.values()),
    )


class HarborRunner:
    """Runs the inner agent via Harbor CLI and collects traces.

    Uses `harbor run` for batch execution with:
    - `-i` flags to select specific tasks
    - `-k` for number of trials
    - `-n` for concurrency
    - `--agent-import-path` for the modified harness
    """

    def __init__(
        self,
        harness_root: Path,
        surface_defs: list[Surface],
        trace_store: TraceStore,
        work_dir: Path,
        dataset: str = "terminal-bench@2.0",
        env_file: Path | None = None,
    ):
        self.harness_root = harness_root
        self.surface_defs = surface_defs
        self.trace_store = trace_store
        self.work_dir = work_dir
        self.dataset = dataset
        self.env_file = env_file
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def _prepare_harness(self, variant: Variant) -> Path:
        """Apply variant and return path to modified harness."""
        output_root = self.work_dir / "harnesses" / variant.variant_id
        return apply_variant_to_harness(
            variant=variant,
            surface_defs=self.surface_defs,
            harness_root=self.harness_root,
            output_root=output_root,
        )

    async def run_harbor_job(
        self,
        config: RunConfig,
        task_ids: list[str] | None = None,
        job_name: str | None = None,
    ) -> Path:
        """Run a Harbor job and return the job directory.

        Args:
            config: Run configuration with variant, model, etc.
            task_ids: If set, only run these tasks (via -i flags).
                      If None, run all tasks in the dataset.
            job_name: Custom job name. Defaults to variant_id.
        """
        harness_path = self._prepare_harness(config.variant)
        job_name = job_name or config.variant.variant_id
        jobs_dir = self.work_dir / "jobs"

        cmd = [
            "harbor", "run",
            "--agent-import-path", "agent:AgentHarness",
            "-d", self.dataset,
            "-m", config.model,
            "-e", config.environment,
            "-k", str(config.n_trials),
            "-n", str(config.concurrency),
            "-o", str(jobs_dir),
            "--job-name", job_name,
            "-y",  # auto-confirm env vars
        ]

        # Always pass env file if it exists (agent needs API keys inside container)
        env_file = self.env_file or (self.harness_root / ".env")
        if env_file.exists():
            cmd.extend(["--env-file", str(env_file)])

        # Filter to specific tasks
        if task_ids:
            for tid in task_ids:
                cmd.extend(["-i", tid])

        print(f"[harbor] Running: {' '.join(cmd[:10])}... ({len(task_ids or [])} tasks, {config.n_trials} trials)")

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(harness_path),
        )
        stdout, stderr = await proc.communicate()

        job_dir = jobs_dir / job_name

        # Save logs
        log_dir = self.work_dir / "logs" / job_name
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "stdout.txt").write_text(stdout.decode())
        (log_dir / "stderr.txt").write_text(stderr.decode())
        (log_dir / "command.txt").write_text(" ".join(cmd))
        (log_dir / "returncode.txt").write_text(str(proc.returncode))

        if proc.returncode != 0:
            print(f"[harbor] WARNING: exit code {proc.returncode}")
            print(f"[harbor] stderr: {stderr.decode()[:500]}")

        return job_dir

    async def run_eval(
        self,
        config: RunConfig,
        split: str | None = None,
        concurrency: int = 20,
    ) -> EvalResult:
        """Run full evaluation for a variant.

        Runs all tasks (or tasks in a split) via a single `harbor run` call.
        """
        # Filter tasks by split
        tasks = config.tasks
        if split:
            tasks = [t for t in tasks if t.split == split]

        task_ids = [t.task_id for t in tasks]
        task_map = {t.task_id: t for t in tasks}

        job_name = f"{config.variant.variant_id}_{split or 'all'}"

        config_with_concurrency = RunConfig(
            variant=config.variant,
            tasks=config.tasks,
            n_trials=config.n_trials,
            model=config.model,
            environment=config.environment,
            max_episodes=config.max_episodes,
            concurrency=concurrency,
        )

        job_dir = await self.run_harbor_job(
            config=config_with_concurrency,
            task_ids=task_ids,
            job_name=job_name,
        )

        # Parse results
        eval_result = _parse_job_results(job_dir, config.variant.variant_id, task_map)
        eval_result.split = split or "all"

        # Store traces
        self.trace_store.store_eval_result(eval_result)

        return eval_result

    async def run_incremental(
        self,
        config: RunConfig,
        previous_result: EvalResult,
        n_canaries: int = 5,
    ) -> EvalResult:
        """Run incremental evaluation: only failing tasks + random canaries.

        This is the key optimization for search iterations:
        - Re-run all tasks that FAILED in previous_result
        - Plus n_canaries random PASSING tasks (regression check)
        - Assume all other passing tasks still pass
        - Returns a complete EvalResult combining new + assumed results
        """
        import random

        failing_ids = {t.task_id for t in previous_result.failing_tasks()}
        passing_ids = {t.task_id for t in previous_result.task_results if t.passed}

        # Pick canaries from passing tasks
        canary_ids = set(random.sample(
            sorted(passing_ids),
            min(n_canaries, len(passing_ids)),
        ))

        run_ids = failing_ids | canary_ids
        task_map = {t.task_id: t for t in config.tasks}

        print(f"[harbor] Incremental: {len(failing_ids)} failing + {len(canary_ids)} canaries = {len(run_ids)} tasks")

        job_name = f"{config.variant.variant_id}_incr"

        job_dir = await self.run_harbor_job(
            config=config,
            task_ids=sorted(run_ids),
            job_name=job_name,
        )

        # Parse new results
        new_result = _parse_job_results(job_dir, config.variant.variant_id, task_map)

        # Merge: new results for tasks we ran, previous results for tasks we didn't
        new_task_ids = {t.task_id for t in new_result.task_results}
        merged_results = list(new_result.task_results)

        for prev_task in previous_result.task_results:
            if prev_task.task_id not in new_task_ids:
                # Carry forward with corrected variant_id
                carried_trials = [
                    TaskTrace(
                        task_id=t.task_id,
                        trial=t.trial,
                        variant_id=config.variant.variant_id,
                        passed=t.passed,
                        score=t.score,
                        steps=t.steps,
                        total_tokens=t.total_tokens,
                        total_cost_usd=t.total_cost_usd,
                        total_duration_sec=t.total_duration_sec,
                        error_summary=t.error_summary,
                        failure_step=t.failure_step,
                    )
                    for t in prev_task.trials
                ]
                merged_results.append(TaskResult(
                    task_id=prev_task.task_id,
                    variant_id=config.variant.variant_id,
                    split=prev_task.split,
                    difficulty=prev_task.difficulty,
                    stratum=prev_task.stratum,
                    trials=carried_trials,
                ))

        merged = EvalResult(
            variant_id=config.variant.variant_id,
            split=previous_result.split,
            task_results=merged_results,
        )

        self.trace_store.store_eval_result(merged)
        return merged


class MockRunner(HarborRunner):
    """Mock runner for testing the framework without Harbor.

    Simulates task execution with configurable pass rates.
    """

    def __init__(
        self,
        harness_root: Path,
        surface_defs: list[Surface],
        trace_store: TraceStore,
        work_dir: Path,
        base_pass_rates: dict[str, float] | None = None,
    ):
        super().__init__(harness_root, surface_defs, trace_store, work_dir)
        self.base_pass_rates = base_pass_rates or {}

    async def run_eval(
        self,
        config: RunConfig,
        split: str | None = None,
        concurrency: int = 5,
    ) -> EvalResult:
        """Simulate evaluation with random pass/fail based on base rates."""
        import random

        tasks = config.tasks
        if split:
            tasks = [t for t in tasks if t.split == split]

        task_results = []
        for task in tasks:
            base_rate = self.base_pass_rates.get(task.task_id, 0.5)
            trials = []
            for trial_num in range(config.n_trials):
                passed = random.random() < base_rate
                n_steps = random.randint(3, 15)
                steps = []
                for i in range(n_steps):
                    steps.append(TraceStep(
                        step_id=i + 1,
                        timestamp=f"2026-04-09T10:{i:02d}:00Z",
                        tool_name="execute_commands" if i < n_steps - 1 else "task_complete",
                        command=f"cmd_{i}" if i < n_steps - 1 else None,
                        output="ok" if passed or i < n_steps - 2 else "ERROR: failed",
                        error=None if passed or i < n_steps - 2 else "Simulated failure",
                    ))
                trials.append(TaskTrace(
                    task_id=task.task_id,
                    trial=trial_num,
                    variant_id=config.variant.variant_id,
                    passed=passed,
                    steps=steps,
                    total_tokens=random.randint(5000, 50000),
                    error_summary=None if passed else "Simulated failure",
                    failure_step=n_steps - 2 if not passed else None,
                ))

            task_results.append(TaskResult(
                task_id=task.task_id,
                variant_id=config.variant.variant_id,
                split=task.split,
                difficulty=task.difficulty,
                stratum=task.stratum,
                trials=trials,
            ))

        result = EvalResult(
            variant_id=config.variant.variant_id,
            split=split or "all",
            task_results=task_results,
        )
        self.trace_store.store_eval_result(result)
        return result

    async def run_incremental(
        self,
        config: RunConfig,
        previous_result: EvalResult,
        n_canaries: int = 5,
    ) -> EvalResult:
        """Simulate incremental eval: re-roll failing + canaries, carry forward rest."""
        import random

        failing_ids = {t.task_id for t in previous_result.failing_tasks()}
        passing_results = [t for t in previous_result.task_results if t.passed]
        passing_ids = {t.task_id for t in passing_results}

        canary_ids = set(random.sample(
            sorted(passing_ids),
            min(n_canaries, len(passing_ids)),
        ))
        rerun_ids = failing_ids | canary_ids

        # Build a temporary config with only the tasks to re-run
        rerun_tasks = [t for t in config.tasks if t.task_id in rerun_ids]
        rerun_config = RunConfig(
            variant=config.variant,
            tasks=rerun_tasks,
            n_trials=config.n_trials,
            model=config.model,
            environment=config.environment,
            max_episodes=config.max_episodes,
            concurrency=config.concurrency,
        )
        new_result = await self.run_eval(rerun_config)

        # Merge: new results for re-run tasks, carry forward for the rest
        new_task_ids = {t.task_id for t in new_result.task_results}
        merged = list(new_result.task_results)

        for prev_task in previous_result.task_results:
            if prev_task.task_id not in new_task_ids:
                merged.append(TaskResult(
                    task_id=prev_task.task_id,
                    variant_id=config.variant.variant_id,
                    split=prev_task.split,
                    difficulty=prev_task.difficulty,
                    stratum=prev_task.stratum,
                    trials=[
                        TaskTrace(
                            task_id=t.task_id, trial=t.trial,
                            variant_id=config.variant.variant_id,
                            passed=t.passed, score=t.score, steps=t.steps,
                            total_tokens=t.total_tokens,
                            total_cost_usd=t.total_cost_usd,
                            total_duration_sec=t.total_duration_sec,
                            error_summary=t.error_summary,
                            failure_step=t.failure_step,
                        ) for t in prev_task.trials
                    ],
                ))

        result = EvalResult(
            variant_id=config.variant.variant_id,
            split=previous_result.split,
            task_results=merged,
        )
        self.trace_store.store_eval_result(result)
        return result
