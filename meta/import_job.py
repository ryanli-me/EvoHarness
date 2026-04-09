"""Import Harbor job results into the meta framework's TraceStore.

Parses a Harbor job directory and creates:
- TraceStore with indexed traces for all tasks
- Baseline EvalResult for the tree search to start from
- Task manifest with difficulty/category metadata

Usage:
    python -m meta.import_job jobs/baseline-modal/bm3 --output runs/experiment
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .runner import _parse_trial_result, TaskCase
from .traces import EvalResult, TaskResult, TraceStore


# Task metadata from TerminalBench-2
TASK_METADATA: dict[str, dict[str, str]] = {}


def _load_task_metadata() -> dict[str, dict[str, str]]:
    """Load difficulty/category metadata from cached task.toml files."""
    global TASK_METADATA
    if TASK_METADATA:
        return TASK_METADATA

    cache_dir = Path.home() / ".cache" / "harbor" / "tasks"
    if not cache_dir.exists():
        return {}

    for task_dir in cache_dir.iterdir():
        if not task_dir.is_dir():
            continue
        # Task name is the subdirectory inside the hash dir
        for sub in task_dir.iterdir():
            if not sub.is_dir():
                continue
            toml_path = sub / "task.toml"
            if toml_path.exists():
                text = toml_path.read_text()
                difficulty = ""
                category = ""
                for line in text.split("\n"):
                    if line.startswith("difficulty"):
                        difficulty = line.split('"')[1] if '"' in line else ""
                    elif line.startswith("category"):
                        category = line.split('"')[1] if '"' in line else ""
                TASK_METADATA[sub.name] = {
                    "difficulty": difficulty,
                    "category": category,
                }

    return TASK_METADATA


def import_harbor_job(
    job_dir: Path,
    output_dir: Path,
    variant_id: str = "baseline",
) -> tuple[EvalResult, list[TaskCase]]:
    """Import a Harbor job directory into our framework.

    Args:
        job_dir: Path to Harbor job directory (contains trial subdirs)
        output_dir: Path to output directory for trace store + manifests
        variant_id: Name for this variant (default: "baseline")

    Returns:
        (EvalResult, list of TaskCases)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_store = TraceStore(output_dir / "traces")
    metadata = _load_task_metadata()

    task_results: dict[str, TaskResult] = {}
    task_cases: list[TaskCase] = []

    for trial_dir in sorted(job_dir.iterdir()):
        if not trial_dir.is_dir():
            continue

        result_path = trial_dir / "result.json"
        if not result_path.exists():
            continue

        try:
            result_data = json.loads(result_path.read_text())
            task_name = result_data.get("task_name", "")
        except (json.JSONDecodeError, KeyError):
            continue

        if not task_name:
            continue

        # Parse the trial
        trial_num = 0
        if task_name in task_results:
            trial_num = len(task_results[task_name].trials)

        trace = _parse_trial_result(trial_dir, task_name, trial_num, variant_id)

        # Get metadata
        meta = metadata.get(task_name, {})
        difficulty = meta.get("difficulty", "unknown")
        category = meta.get("category", "unknown")

        if task_name not in task_results:
            task_results[task_name] = TaskResult(
                task_id=task_name,
                variant_id=variant_id,
                split="all",
                difficulty=difficulty,
                stratum=category,
            )
            task_cases.append(TaskCase(
                task_id=task_name,
                split="all",
                difficulty=difficulty,
                stratum=category,
            ))

        task_results[task_name].trials.append(trace)
        trace_store.store_trace(trace, "all")

    eval_result = EvalResult(
        variant_id=variant_id,
        split="all",
        task_results=list(task_results.values()),
    )

    # Save eval result summary
    summary = {
        "variant_id": variant_id,
        "pass_count": eval_result.pass_count,
        "total_count": eval_result.total_count,
        "pass_rate": eval_result.pass_rate,
        "by_difficulty": eval_result.by_difficulty(),
        "by_stratum": eval_result.by_stratum(),
        "failing_tasks": [t.task_id for t in eval_result.failing_tasks()],
        "passing_tasks": [t.task_id for t in eval_result.task_results if t.passed],
    }
    (output_dir / "baseline_summary.json").write_text(json.dumps(summary, indent=2))

    # Save task cases
    (output_dir / "task_cases.json").write_text(json.dumps(
        [t.to_dict() for t in task_cases], indent=2
    ))

    # Print summary
    print(f"[import] Imported {eval_result.total_count} tasks from {job_dir}")
    print(f"[import] Pass rate: {eval_result.pass_count}/{eval_result.total_count} ({eval_result.pass_rate:.1%})")

    by_diff = eval_result.by_difficulty()
    if by_diff:
        print(f"[import] By difficulty: {', '.join(f'{k}: {v:.0%}' for k, v in sorted(by_diff.items()))}")

    by_strat = eval_result.by_stratum()
    if by_strat:
        print(f"[import] By category: {', '.join(f'{k}: {v:.0%}' for k, v in sorted(by_strat.items()))}")

    failing = eval_result.failing_tasks()
    if failing:
        print(f"[import] Failing ({len(failing)}):")
        for t in sorted(failing, key=lambda x: x.task_id):
            ft = t.failing_trace()
            err = ft.error_summary[:80] if ft and ft.error_summary else "unknown"
            print(f"  {t.task_id} ({t.difficulty}/{t.stratum}): {err}")

    print(f"[import] Traces saved to {output_dir / 'traces'}")
    return eval_result, task_cases


def main():
    parser = argparse.ArgumentParser(description="Import Harbor job into meta framework")
    parser.add_argument("job_dir", type=Path, help="Path to Harbor job directory")
    parser.add_argument("--output", "-o", type=Path, default=Path("runs/experiment"), help="Output directory")
    parser.add_argument("--variant-id", default="baseline", help="Variant name")
    args = parser.parse_args()

    import_harbor_job(args.job_dir, args.output, args.variant_id)


if __name__ == "__main__":
    main()
