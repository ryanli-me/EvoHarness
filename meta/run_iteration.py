"""Run a single iteration: propose → apply → eval.

Usage:
    # Run proposer, apply variant, eval on failing tasks
    python -m meta.run_iteration \
        --experiment-dir runs/experiment \
        --iteration 2 \
        --parent-variant iter_001 \
        --proposer-model sonnet \
        --eval-env modal \
        --focus-surface env_bootstrap

    # Just apply a proposal and run eval (skip proposer)
    python -m meta.run_iteration \
        --experiment-dir runs/experiment \
        --iteration 2 \
        --proposal runs/experiment/iter_002/proposal.json \
        --eval-env modal
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
from pathlib import Path

from .surfaces import load_surfaces_from_harness, Variant, apply_variant_to_harness
from .proposer import ClaudeCodeProposer
from .traces import TraceStore, EvalResult, TaskResult, TaskTrace
from .notebook import ResearchNotebook


def load_eval_result(experiment_dir: Path) -> EvalResult:
    """Rebuild EvalResult from imported traces."""
    task_cases = json.loads((experiment_dir / "task_cases.json").read_text())
    trace_dir = experiment_dir / "traces" / "variants" / "baseline" / "all"

    task_results = []
    for tc in task_cases:
        tid = tc["task_id"]
        trials = []
        for tf in sorted(trace_dir.glob(f"{tid}_trial_*.json")):
            data = json.loads(tf.read_text())
            trials.append(TaskTrace(
                **{k: v for k, v in data.items()
                   if k in TaskTrace.__dataclass_fields__ and k != "steps"}
            ))
        if trials:
            task_results.append(TaskResult(
                task_id=tid, variant_id="baseline", split="all",
                difficulty=tc.get("difficulty"), stratum=tc.get("stratum"),
                trials=trials,
            ))

    return EvalResult(variant_id="baseline", split="all", task_results=task_results)


def run_proposer(
    experiment_dir: Path,
    iteration: int,
    parent_variant: Variant,
    surfaces,
    eval_result: EvalResult,
    model: str = "sonnet",
    budget: float = 3.0,
    focus_surface: str | None = None,
) -> dict | None:
    """Run the Claude Code proposer and return the proposal."""
    trace_store = TraceStore(experiment_dir / "traces")
    notebook = ResearchNotebook(experiment_dir / "notebook")

    proposer = ClaudeCodeProposer(model=model, max_budget_usd=budget)

    # Build workspace
    workspace_dir = experiment_dir / f"iter_{iteration:03d}" / "proposer_workspace"
    proposer._build_workspace(
        workspace_dir, surfaces, parent_variant,
        eval_result, trace_store, notebook, iteration,
    )

    # Build prompt
    focus_hint = ""
    if focus_surface:
        focus_hint = f" Focus on the {focus_surface} surface specifically."

    prompt = (
        "Read TASK.md for your instructions. Investigate the failing traces in traces/. "
        "Check notebook/ for prior learnings — do NOT repeat dead ends. "
        f"Grep across traces for common patterns, then write your proposal to proposal.json.{focus_hint}"
    )

    cmd = [
        "claude", "-p", prompt,
        "--model", model,
        "--output-format", "text",
        "--max-budget-usd", str(budget),
        "--permission-mode", "auto",
        "--allowed-tools",
        "Read", "Glob", "Grep", "Write",
        "Bash(grep:*,cat:*,head:*,tail:*,wc:*,sort:*,uniq:*,jq:*,find:*,ls:*)",
    ]

    print(f"[iter {iteration}] Launching proposer (model={model}, budget=${budget})...")
    result = subprocess.run(
        cmd, cwd=str(workspace_dir),
        capture_output=True, text=True, timeout=600,
    )

    # Save output
    (workspace_dir / "agent_stdout.txt").write_text(result.stdout)
    (workspace_dir / "agent_stderr.txt").write_text(result.stderr)

    # Read proposal
    proposal_path = workspace_dir / "proposal.json"
    if not proposal_path.exists():
        print(f"[iter {iteration}] Proposer did not write proposal.json")
        print(f"[iter {iteration}] stdout: {result.stdout[-500:]}")
        return None

    proposal = json.loads(proposal_path.read_text())
    if proposal.get("surface_name") is None:
        print(f"[iter {iteration}] Proposer decided no change needed")
        return None

    print(f"[iter {iteration}] Proposal: {proposal.get('change_summary', '')}")
    print(f"[iter {iteration}] Surface: {proposal['surface_name']}")

    # Copy proposal to iteration dir
    iter_dir = experiment_dir / f"iter_{iteration:03d}"
    (iter_dir / "proposal.json").write_text(json.dumps(proposal, indent=2))

    return proposal


def apply_and_build_harness(
    experiment_dir: Path,
    iteration: int,
    parent_variant: Variant,
    proposal: dict,
    surfaces,
    harness_root: Path,
) -> tuple[Variant, Path]:
    """Apply proposal to create a variant and build the harness directory."""
    iter_dir = experiment_dir / f"iter_{iteration:03d}"
    harness_dir = iter_dir / "harness"

    # Create variant
    variant = parent_variant.derive(
        variant_id=f"iter_{iteration:03d}",
        changes={proposal["surface_name"]: proposal["new_value"]},
        hypothesis=proposal.get("hypothesis", ""),
    )
    variant.save(iter_dir / "variant.json")

    # Apply variant to harness
    apply_variant_to_harness(variant, surfaces, harness_root, harness_dir)

    print(f"[iter {iteration}] Harness built at {harness_dir}")
    print(f"[iter {iteration}] Changed surface: {proposal['surface_name']}")

    return variant, harness_dir


def launch_eval(
    harness_dir: Path,
    task_ids: list[str],
    job_name: str,
    output_dir: Path,
    env: str = "modal",
    model: str = "anthropic/claude-opus-4-6",
    env_file: Path | None = None,
) -> None:
    """Launch Harbor eval on specified tasks."""
    cmd = [
        "harbor", "run",
        "--agent-import-path", "agent:AgentHarness",
        "-d", "terminal-bench@2.0",
        "-m", model,
        "-e", env,
        "-k", "1",
        "-n", str(len(task_ids)),
        "-o", str(output_dir),
        "--job-name", job_name,
        "-y",
    ]

    if env_file and env_file.exists():
        cmd.extend(["--env-file", str(env_file)])

    for tid in task_ids:
        cmd.extend(["-i", tid])

    print(f"[eval] Launching {len(task_ids)} tasks on {env} (concurrency={len(task_ids)})...")
    print(f"[eval] Command: {' '.join(cmd[:15])}...")

    subprocess.Popen(
        cmd, cwd=str(harness_dir),
        stdout=open(output_dir / f"{job_name}.log", "w"),
        stderr=subprocess.STDOUT,
    )
    print(f"[eval] Launched in background. Results at {output_dir}/{job_name}/")


def main():
    parser = argparse.ArgumentParser(description="Run a single optimization iteration")
    parser.add_argument("--experiment-dir", type=Path, default=Path("runs/experiment"))
    parser.add_argument("--harness-root", type=Path, default=Path("."))
    parser.add_argument("--iteration", "-i", type=int, required=True)
    parser.add_argument("--parent-variant", default="baseline", help="Parent variant ID or 'baseline'")
    parser.add_argument("--proposer-model", default="sonnet")
    parser.add_argument("--proposer-budget", type=float, default=3.0)
    parser.add_argument("--focus-surface", help="Hint proposer to focus on this surface")
    parser.add_argument("--proposal", type=Path, help="Skip proposer, use existing proposal.json")
    parser.add_argument("--eval-env", default="modal", choices=["modal", "docker"])
    parser.add_argument("--eval-model", default="anthropic/claude-opus-4-6")
    parser.add_argument("--skip-eval", action="store_true", help="Only propose, don't eval")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    args = parser.parse_args()

    harness_root = args.harness_root.resolve()
    experiment_dir = args.experiment_dir.resolve()
    surfaces = load_surfaces_from_harness(harness_root)

    # Load parent variant
    if args.parent_variant == "baseline":
        parent = Variant.baseline(surfaces)
    else:
        parent = Variant.load(experiment_dir / f"{args.parent_variant}" / "variant.json")

    # Load eval result for proposer context
    eval_result = load_eval_result(experiment_dir)
    summary = json.loads((experiment_dir / "baseline_summary.json").read_text())
    failing_tasks = summary["failing_tasks"]

    print(f"[iter {args.iteration}] Parent: {args.parent_variant}")
    print(f"[iter {args.iteration}] Failing tasks: {len(failing_tasks)}")

    # Step 1: Propose
    if args.proposal:
        proposal = json.loads(args.proposal.read_text())
        print(f"[iter {args.iteration}] Using existing proposal: {proposal.get('change_summary')}")
        # Copy to iter dir
        iter_dir = experiment_dir / f"iter_{args.iteration:03d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        (iter_dir / "proposal.json").write_text(json.dumps(proposal, indent=2))
    else:
        proposal = run_proposer(
            experiment_dir, args.iteration, parent, surfaces, eval_result,
            model=args.proposer_model, budget=args.proposer_budget,
            focus_surface=args.focus_surface,
        )
        if proposal is None:
            print(f"[iter {args.iteration}] No proposal. Exiting.")
            return

    # Step 2: Apply variant to harness
    variant, harness_dir = apply_and_build_harness(
        experiment_dir, args.iteration, parent, proposal, surfaces, harness_root,
    )

    if args.skip_eval:
        print(f"[iter {args.iteration}] Skipping eval. Harness ready at {harness_dir}")
        return

    # Step 3: Launch eval on failing tasks
    launch_eval(
        harness_dir=harness_dir,
        task_ids=failing_tasks,
        job_name=f"iter_{args.iteration:03d}",
        output_dir=experiment_dir / f"iter_{args.iteration:03d}" / "harbor_jobs",
        env=args.eval_env,
        model=args.eval_model,
        env_file=args.env_file,
    )

    print(f"\n[iter {args.iteration}] Done! Monitor progress with:")
    print(f"  for d in {experiment_dir}/iter_{args.iteration:03d}/harbor_jobs/iter_{args.iteration:03d}/*/; do")
    print(f'    [ -f "$d/verifier/reward.txt" ] && echo "$(basename $d | sed s/__.*//): $(cat $d/verifier/reward.txt)"')
    print(f"  done")


if __name__ == "__main__":
    main()
