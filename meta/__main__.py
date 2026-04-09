#!/usr/bin/env python3
"""CLI entry point for the meta framework.

Usage:
    # Run full optimization experiment
    python -m meta run meta/configs/kira_tbench2.toml

    # Run with mock runner (for development/demo)
    python -m meta demo

    # Show current surfaces
    python -m meta surfaces
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path


def cmd_run(args: argparse.Namespace) -> None:
    """Run a full optimization experiment."""
    from meta.config import load_config
    from meta.core import run_experiment
    from meta.proposer import ClaudeCodeProposer
    from meta.runner import HarborRunner
    from meta.surfaces import load_surfaces_from_harness
    from meta.traces import TraceStore

    config = load_config(Path(args.config))
    trace_store = TraceStore(config.output_dir / "traces")
    surfaces = load_surfaces_from_harness(config.harness_root)

    runner = HarborRunner(
        harness_root=config.harness_root,
        surface_defs=surfaces,
        trace_store=trace_store,
        work_dir=config.output_dir / "work",
        dataset=args.dataset or "terminal-bench@2.0",
    )

    proposer = ClaudeCodeProposer(model=args.proposer_model or "sonnet")

    report = asyncio.run(run_experiment(config, runner, proposer))
    print(f"\nBest variant: {report.best_variant_id}")
    print(f"Report saved to: {config.output_dir / 'report.md'}")


def cmd_demo(args: argparse.Namespace) -> None:
    """Run a demo with mock runner to show the framework in action."""
    from meta.core import ExperimentConfig, run_experiment
    from meta.proposer import ClaudeCodeProposer, LiteLLMProposer, MockProposer
    from meta.runner import MockRunner, TaskCase
    from meta.surfaces import load_surfaces_from_harness
    from meta.traces import TraceStore

    harness_root = Path(__file__).resolve().parent.parent
    output_dir = Path(args.output or "runs/demo")

    # Create task split
    tasks = []
    train_failing = [f"task_train_fail_{i}" for i in range(5)]
    train_passing = [f"task_train_pass_{i}" for i in range(8)]
    holdout_failing = [f"task_holdout_fail_{i}" for i in range(3)]
    holdout_passing = [f"task_holdout_pass_{i}" for i in range(5)]

    for tid in train_failing:
        tasks.append(TaskCase(task_id=tid, split="train", difficulty="hard", stratum="file_ops"))
    for tid in train_passing:
        tasks.append(TaskCase(task_id=tid, split="train", difficulty="medium", stratum="compilation"))
    for tid in holdout_failing:
        tasks.append(TaskCase(task_id=tid, split="holdout", difficulty="hard", stratum="file_ops"))
    for tid in holdout_passing:
        tasks.append(TaskCase(task_id=tid, split="holdout", difficulty="medium", stratum="compilation"))

    base_pass_rates = {}
    for tid in train_passing + holdout_passing:
        base_pass_rates[tid] = 0.9
    for tid in train_failing + holdout_failing:
        base_pass_rates[tid] = 0.2

    config = ExperimentConfig(
        name="demo",
        harness_root=harness_root,
        output_dir=output_dir,
        tasks=tasks,
        max_iterations=args.iterations or 5,
        n_trials_search=args.trials or 2,
        n_trials_final=args.trials or 2,
        model="anthropic/claude-opus-4-6",
        proposer_model=args.proposer_model or "anthropic/claude-sonnet-4-6",
    )

    surfaces = load_surfaces_from_harness(harness_root)
    trace_store = TraceStore(output_dir / "traces")

    runner = MockRunner(
        harness_root=harness_root,
        surface_defs=surfaces,
        trace_store=trace_store,
        work_dir=output_dir / "work",
        base_pass_rates=base_pass_rates,
    )

    if args.mock_proposer:
        proposer = MockProposer()
    elif args.lite:
        proposer = LiteLLMProposer(model=args.proposer_model or "anthropic/claude-sonnet-4-6")
    else:
        proposer = ClaudeCodeProposer(model=args.proposer_model or "sonnet")

    report = asyncio.run(run_experiment(config, runner, proposer))
    print(f"\nReport saved to: {output_dir / 'report.md'}")


def cmd_import(args: argparse.Namespace) -> None:
    """Import Harbor job results into trace store."""
    from meta.import_job import import_harbor_job
    import_harbor_job(Path(args.job_dir), Path(args.output), args.variant_id)


def cmd_surfaces(args: argparse.Namespace) -> None:
    """Show the editable surfaces of the harness."""
    from meta.surfaces import load_surfaces_from_harness

    harness_root = Path(args.harness or ".")
    surfaces = load_surfaces_from_harness(harness_root)

    for s in surfaces:
        risk = "HIGH" if s.fragility > 0.5 else "MEDIUM" if s.fragility > 0.2 else "LOW"
        print(f"\n{'='*60}")
        print(f"Surface: {s.name} [{risk} risk]")
        print(f"Kind: {s.kind} | Target: {s.target}")
        print(f"Description: {s.description}")
        print(f"Value length: {len(s.base_value)} chars")
        preview = s.base_value[:200].replace('\n', '\\n')
        print(f"Preview: {preview}...")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Meta-Harness: Automated harness optimization framework",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run
    p_run = subparsers.add_parser("run", help="Run optimization experiment")
    p_run.add_argument("config", help="Path to TOML config file")
    p_run.add_argument("--dataset", help="Dataset override")
    p_run.add_argument("--proposer-model", help="Model for proposer (default: sonnet)")
    p_run.set_defaults(func=cmd_run)

    # demo
    p_demo = subparsers.add_parser("demo", help="Run demo with mock runner")
    p_demo.add_argument("--output", "-o", help="Output directory")
    p_demo.add_argument("--iterations", "-i", type=int, help="Max iterations")
    p_demo.add_argument("--trials", "-t", type=int, help="Trials per task")
    p_demo.add_argument("--proposer-model", help="Model for the proposer agent")
    p_demo.add_argument("--mock-proposer", action="store_true", help="Use mock proposer (no API key needed)")
    p_demo.add_argument("--lite", action="store_true", help="Use single LLM call proposer instead of Claude Code agent")
    p_demo.set_defaults(func=cmd_demo)

    # import
    p_import = subparsers.add_parser("import", help="Import Harbor job results")
    p_import.add_argument("job_dir", help="Path to Harbor job directory")
    p_import.add_argument("--output", "-o", default="runs/experiment", help="Output directory")
    p_import.add_argument("--variant-id", default="baseline", help="Variant name")
    p_import.set_defaults(func=cmd_import)

    # surfaces
    p_surf = subparsers.add_parser("surfaces", help="Show harness surfaces")
    p_surf.add_argument("--harness", help="Path to harness root")
    p_surf.set_defaults(func=cmd_surfaces)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
