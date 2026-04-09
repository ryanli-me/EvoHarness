"""Core optimization loop with tree search.

Instead of Meta-Harness's linear chain (8 iterations, 7 wasted),
we maintain a tree of variant branches:

    baseline
    ├── branch A: modify system_prompt → prescreen → eval → accept/prune
    ├── branch B: modify env_bootstrap → prescreen → eval → accept/prune
    └── merge(A, B): combine accepted changes → eval

Each iteration:
    1. Select a branch to extend (Thompson sampling)
    2. Proposer investigates that branch's failing traces
    3. Prescreen on 3 failing tasks (cheap filter)
    4. If promising, full incremental eval
    5. Update branch scores, notebook
    6. Periodically try merging top branches
"""

from __future__ import annotations

import asyncio
import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .notebook import IterationRecord, ResearchNotebook
from .proposer import ClaudeCodeProposer
from .runner import HarborRunner, RunConfig, TaskCase
from .surfaces import Surface, Variant
from .traces import EvalResult, TraceStore


@dataclass
class ExperimentConfig:
    """Configuration for a full optimization experiment."""

    name: str
    harness_root: Path
    output_dir: Path
    tasks: list[TaskCase]
    max_iterations: int = 10
    n_trials_search: int = 1
    n_trials_final: int = 5
    n_canaries: int = 5
    n_prescreen_tasks: int = 3      # Tasks for cheap pre-screening
    n_parallel_proposals: int = 3   # Parallel proposer branches per iteration
    regression_tolerance: int = 1
    merge_interval: int = 3         # Try merging every N iterations
    model: str = "anthropic/claude-opus-4-6"
    proposer_model: str = "anthropic/claude-opus-4-6"
    environment: str = "runloop"
    max_episodes: int = 20
    concurrency: int = 20


@dataclass
class Branch:
    """A branch in the search tree."""

    variant: Variant
    result: EvalResult
    score: float                    # pass_count
    parent_branch_id: str | None
    iteration_created: int
    status: str = "active"          # active | pruned | merged
    prescreen_scores: list[float] = field(default_factory=list)

    @property
    def branch_id(self) -> str:
        return self.variant.variant_id


class SearchTree:
    """Tree of variant branches with Thompson sampling selection.

    Maintains multiple active branches. Uses Thompson sampling to
    balance exploration (try uncertain branches) vs exploitation
    (extend high-scoring branches).
    """

    def __init__(self):
        self.branches: dict[str, Branch] = {}
        self.baseline_score: float = 0.0

    def add_branch(self, branch: Branch) -> None:
        self.branches[branch.branch_id] = branch

    @property
    def active_branches(self) -> list[Branch]:
        return [b for b in self.branches.values() if b.status == "active"]

    @property
    def best_branch(self) -> Branch | None:
        """Best branch by score across ALL branches (not just active)."""
        if not self.branches:
            return None
        return max(self.branches.values(), key=lambda b: b.score)

    def select_branch(self) -> Branch:
        """Select branch to extend using Thompson sampling.

        Each branch has a Beta distribution parameterized by its
        (pass_count, fail_count). We sample from each and pick the highest.
        """
        active = self.active_branches
        if not active:
            raise ValueError("No active branches")

        if len(active) == 1:
            return active[0]

        best_sample = -1.0
        best_branch = active[0]

        for branch in active:
            # Beta(successes + 1, failures + 1) — +1 for uninformative prior
            total = branch.result.total_count
            passed = branch.result.pass_count
            failed = total - passed
            sample = np.random.beta(passed + 1, failed + 1)

            if sample > best_sample:
                best_sample = sample
                best_branch = branch

        return best_branch

    def prune_branch(self, branch_id: str) -> None:
        if branch_id in self.branches:
            self.branches[branch_id].status = "pruned"

    def summary(self) -> str:
        lines = ["## Search Tree"]
        for b in sorted(self.branches.values(), key=lambda x: -x.score):
            status_icon = {"active": "+", "pruned": "x", "merged": "~"}[b.status]
            parent = f" (from {b.parent_branch_id})" if b.parent_branch_id else ""
            lines.append(
                f"  [{status_icon}] {b.branch_id}: "
                f"{b.score:.0f}/{b.result.total_count} "
                f"({b.result.pass_rate:.1%}){parent}"
            )
        return "\n".join(lines)


class Gatekeeper:
    """Acceptance gate: pass count must improve, regressions within tolerance."""

    def __init__(self, regression_tolerance: int = 1):
        self.regression_tolerance = regression_tolerance

    def evaluate(
        self,
        candidate: EvalResult,
        parent: EvalResult,
    ) -> tuple[bool, str]:
        regressions = candidate.regressions_vs(parent)
        improvements = candidate.improvements_vs(parent)

        if candidate.pass_count < parent.pass_count:
            return False, (
                f"Regressed: {candidate.pass_count} < {parent.pass_count} "
                f"(+{len(improvements)}, -{len(regressions)})"
            )

        if candidate.pass_count == parent.pass_count:
            return False, (
                f"Unchanged: {candidate.pass_count} == {parent.pass_count} "
                f"(+{len(improvements)}, -{len(regressions)})"
            )

        if len(regressions) > self.regression_tolerance:
            return False, (
                f"Too many regressions ({len(regressions)} > "
                f"{self.regression_tolerance}): {regressions[:5]}"
            )

        return True, (
            f"Improved: {parent.pass_count} -> {candidate.pass_count} "
            f"(+{len(improvements)}, -{len(regressions)})"
        )

    def check_prescreen(
        self,
        prescreen_passed: int,
        prescreen_total: int,
        parent_passed: int,
    ) -> tuple[bool, str]:
        """Quick check: did prescreen show any improvement over parent?"""
        if prescreen_passed > parent_passed:
            return True, f"Prescreen promising: {prescreen_passed}/{prescreen_total} (was {parent_passed})"
        return False, f"Prescreen not promising: {prescreen_passed}/{prescreen_total} (was {parent_passed})"


@dataclass
class ExperimentReport:
    """Final report from an optimization experiment."""

    name: str
    baseline_score: float
    final_score: float
    iterations_run: int
    branches_created: int
    branches_accepted: int
    branches_pruned: int
    merges_attempted: int
    merges_accepted: int
    best_variant_id: str
    improvements: list[str]
    regressions: list[str]
    total_duration_sec: float
    tree_summary: str
    iteration_summaries: list[str]

    def summary(self) -> str:
        lines = [
            f"# Experiment Report: {self.name}",
            f"",
            f"## Results",
            f"- Baseline: {self.baseline_score:.1%}",
            f"- Final:    {self.final_score:.1%}",
            f"- Delta:    {self.final_score - self.baseline_score:+.1%}",
            f"",
            f"## Tree Search",
            f"- Iterations: {self.iterations_run}",
            f"- Branches: {self.branches_created} created, {self.branches_accepted} accepted, {self.branches_pruned} pruned",
            f"- Merges: {self.merges_attempted} attempted, {self.merges_accepted} accepted",
            f"- Best variant: {self.best_variant_id}",
            f"- Duration: {self.total_duration_sec:.0f}s",
            f"",
            self.tree_summary,
            f"",
            f"## Changes",
        ]
        if self.improvements:
            lines.append(f"- Improvements ({len(self.improvements)}): {', '.join(self.improvements[:10])}")
        if self.regressions:
            lines.append(f"- Regressions ({len(self.regressions)}): {', '.join(self.regressions[:10])}")

        lines.append(f"\n## Iteration Log")
        for s in self.iteration_summaries:
            lines.append(s)

        return "\n".join(lines)


async def run_experiment(
    config: ExperimentConfig,
    runner: HarborRunner,
    proposer=None,
) -> ExperimentReport:
    """Run a tree-search optimization experiment.

    Phase 0: Baseline eval
    Phase 1: Tree search — select branch, propose, prescreen, eval, accept/prune
    Phase 2: Merge top branches
    """
    start_time = time.time()

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    trace_store = TraceStore(output_dir / "traces")
    notebook = ResearchNotebook(output_dir / "notebook")
    gatekeeper = Gatekeeper(config.regression_tolerance)
    tree = SearchTree()

    if proposer is None:
        proposer = ClaudeCodeProposer()

    from .surfaces import load_surfaces_from_harness
    surfaces = load_surfaces_from_harness(config.harness_root)

    (output_dir / "config.json").write_text(json.dumps({
        "name": config.name,
        "max_iterations": config.max_iterations,
        "n_trials_search": config.n_trials_search,
        "n_prescreen_tasks": config.n_prescreen_tasks,
        "merge_interval": config.merge_interval,
        "model": config.model,
        "proposer_model": config.proposer_model,
        "n_tasks": len(config.tasks),
    }, indent=2))

    (output_dir / "surface_manifest.json").write_text(json.dumps(
        [s.to_dict() for s in surfaces], indent=2
    ))

    print(f"[meta] Starting tree-search experiment: {config.name}")
    print(f"[meta] Surfaces: {[s.name for s in surfaces]}")
    print(f"[meta] Tasks: {len(config.tasks)}, Prescreen: {config.n_prescreen_tasks}")
    print(f"[meta] Max iterations: {config.max_iterations}, Merge every: {config.merge_interval}")

    # --- Phase 0: Baseline ---
    print(f"\n[meta] Phase 0: Evaluating baseline...")
    baseline = Variant.baseline(surfaces)
    baseline.save(output_dir / "variants" / "baseline.json")

    baseline_config = RunConfig(
        variant=baseline,
        tasks=config.tasks,
        n_trials=config.n_trials_search,
        model=config.model,
        environment=config.environment,
        max_episodes=config.max_episodes,
        concurrency=config.concurrency,
    )

    baseline_result = await runner.run_eval(baseline_config, concurrency=config.concurrency)

    tree.baseline_score = baseline_result.pass_count
    tree.add_branch(Branch(
        variant=baseline,
        result=baseline_result,
        score=baseline_result.pass_count,
        parent_branch_id=None,
        iteration_created=0,
    ))

    print(f"[meta] Baseline: {baseline_result.pass_count}/{baseline_result.total_count} ({baseline_result.pass_rate:.1%})")
    by_diff = baseline_result.by_difficulty()
    if by_diff:
        print(f"[meta] By difficulty: {', '.join(f'{k}: {v:.0%}' for k, v in sorted(by_diff.items()))}")

    for surface in surfaces:
        notebook.update_surface_risk(surface.name, surface.fragility, "Initial")

    # --- Phase 1: Tree search with parallel proposals ---
    iterations_run = 0
    branches_accepted = 0
    branches_pruned = 0
    merges_attempted = 0
    merges_accepted = 0
    iteration_summaries = []
    n_parallel = config.n_parallel_proposals

    for iteration in range(1, config.max_iterations + 1):
        iterations_run = iteration

        # --- Try merge periodically ---
        if iteration > 1 and iteration % config.merge_interval == 0:
            merge_result = await _try_merge(
                tree, config, runner, gatekeeper, notebook,
                output_dir, iteration, iteration_summaries,
            )
            if merge_result:
                merges_attempted += 1
                if merge_result == "accepted":
                    merges_accepted += 1

        # --- Select N branches to extend in parallel ---
        # Use Thompson sampling N times (may pick same branch multiple times)
        parent_branches = []
        for _ in range(n_parallel):
            if not tree.active_branches:
                break
            b = tree.select_branch()
            if b.result.pass_rate < 1.0:
                parent_branches.append(b)
        # Deduplicate — if same branch picked multiple times, still only propose once from it
        seen = set()
        unique_parents = []
        for b in parent_branches:
            if b.branch_id not in seen:
                seen.add(b.branch_id)
                unique_parents.append(b)

        if not unique_parents:
            print("[meta] No branches to explore. Stopping.")
            break

        print(f"\n[meta] === Iteration {iteration}/{config.max_iterations} — {len(unique_parents)} parallel proposals ===")
        for b in unique_parents:
            print(f"[meta]   extending: {b.branch_id} ({b.result.pass_rate:.1%})")

        # --- Propose in parallel (local Claude Code processes) ---
        async def propose_one(parent_branch, slot):
            return await proposer.propose(
                surfaces=surfaces,
                current_variant=parent_branch.variant,
                train_result=parent_branch.result,
                trace_store=trace_store,
                notebook=notebook,
                iteration=iteration * 100 + slot,  # unique iteration id per slot
            )

        print(f"[meta] Spawning {len(unique_parents)} proposers in parallel...")
        proposal_tasks = [
            propose_one(parent, slot)
            for slot, parent in enumerate(unique_parents)
        ]
        proposals = await asyncio.gather(*proposal_tasks, return_exceptions=True)

        # --- Filter valid proposals ---
        candidates = []  # list of (parent_branch, proposal, variant)
        for parent, result in zip(unique_parents, proposals):
            if isinstance(result, Exception):
                print(f"[meta]   {parent.branch_id}: proposer error: {result}")
                continue
            if result is None:
                print(f"[meta]   {parent.branch_id}: no proposal")
                continue
            proposal = result
            variant = parent.variant.derive(
                variant_id=f"branch_{iteration:03d}_{len(candidates)}",
                changes={proposal.surface_name: proposal.new_value},
                hypothesis=proposal.hypothesis,
            )
            variant.save(output_dir / "variants" / f"{variant.variant_id}.json")
            candidates.append((parent, proposal, variant))
            print(f"[meta]   {parent.branch_id} -> {proposal.surface_name}: {proposal.change_summary}")

        if not candidates:
            print("[meta] No valid proposals. Stopping.")
            break

        # --- Prescreen all candidates in parallel ---
        async def prescreen_one(parent_branch, candidate_variant):
            failing_ids = [t.task_id for t in parent_branch.result.failing_tasks()]
            prescreen_ids = random.sample(failing_ids, min(config.n_prescreen_tasks, len(failing_ids)))
            prescreen_tasks = [t for t in config.tasks if t.task_id in prescreen_ids]
            prescreen_config = RunConfig(
                variant=candidate_variant, tasks=prescreen_tasks,
                n_trials=config.n_trials_search, model=config.model,
                environment=config.environment, max_episodes=config.max_episodes,
                concurrency=config.concurrency,
            )
            return await runner.run_eval(prescreen_config, concurrency=config.concurrency)

        print(f"[meta] Pre-screening {len(candidates)} candidates...")
        prescreen_tasks_list = [
            prescreen_one(parent, variant)
            for parent, proposal, variant in candidates
        ]
        prescreen_results = await asyncio.gather(*prescreen_tasks_list, return_exceptions=True)

        # --- Pick best prescreen result for full eval ---
        best_candidate = None
        best_prescreen_score = -1
        for (parent, proposal, variant), ps_result in zip(candidates, prescreen_results):
            if isinstance(ps_result, Exception):
                print(f"[meta]   {variant.variant_id}: prescreen error: {ps_result}")
                branches_pruned += 1
                continue

            ps_ok, ps_reason = gatekeeper.check_prescreen(ps_result.pass_count, ps_result.total_count, 0)
            if not ps_ok:
                print(f"[meta]   {variant.variant_id}: PRUNED ({ps_reason})")
                branches_pruned += 1
                record = IterationRecord(
                    iteration=iteration, hypothesis=proposal.hypothesis,
                    surfaces_changed=[proposal.surface_name],
                    change_summary=proposal.change_summary,
                    variant_id=variant.variant_id, parent_id=parent.branch_id,
                    train_pass_rate=ps_result.pass_rate,
                    accepted=False, rejection_reason=f"Prescreen: {ps_reason}",
                    learnings=f"Pre-screen failed.",
                )
                _update_notebook(surfaces, notebook, proposal, record, iteration, ps_reason)
                iteration_summaries.append(record.summary())
                continue

            print(f"[meta]   {variant.variant_id}: prescreen passed ({ps_result.pass_count}/{ps_result.total_count})")
            if ps_result.pass_count > best_prescreen_score:
                best_prescreen_score = ps_result.pass_count
                best_candidate = (parent, proposal, variant)

        if best_candidate is None:
            print("[meta] All candidates pruned at prescreen.")
            continue

        parent, proposal, candidate = best_candidate
        print(f"[meta] Best prescreen: {candidate.variant_id}. Full eval...")

        # --- Full incremental eval on best candidate ---
        full_config = RunConfig(
            variant=candidate, tasks=config.tasks,
            n_trials=config.n_trials_search, model=config.model,
            environment=config.environment, max_episodes=config.max_episodes,
            concurrency=config.concurrency,
        )

        if hasattr(runner, 'run_incremental'):
            candidate_result = await runner.run_incremental(
                full_config, parent.result, n_canaries=config.n_canaries,
            )
        else:
            candidate_result = await runner.run_eval(full_config, concurrency=config.concurrency)

        print(f"[meta] Result: {candidate_result.pass_count}/{candidate_result.total_count} ({candidate_result.pass_rate:.1%})")

        # --- Gate ---
        accepted, reason = gatekeeper.evaluate(candidate_result, parent.result)
        improvements = candidate_result.improvements_vs(parent.result)
        regressions = candidate_result.regressions_vs(parent.result)

        record = IterationRecord(
            iteration=iteration, hypothesis=proposal.hypothesis,
            surfaces_changed=[proposal.surface_name],
            change_summary=proposal.change_summary,
            variant_id=candidate.variant_id, parent_id=parent.branch_id,
            train_pass_rate=candidate_result.pass_rate,
            train_improvements=improvements, train_regressions=regressions,
            accepted=accepted,
            rejection_reason=None if accepted else reason,
        )

        if accepted:
            print(f"[meta] ACCEPTED: {reason}")
            branches_accepted += 1
            tree.add_branch(Branch(
                variant=candidate, result=candidate_result,
                score=candidate_result.pass_count,
                parent_branch_id=parent.branch_id,
                iteration_created=iteration,
            ))
            record.learnings = f"New branch from {parent.branch_id}. {reason}"
            notebook.add_finding(f"Iter {iteration}: {proposal.change_summary} — {reason}")
        else:
            print(f"[meta] REJECTED: {reason}")
            branches_pruned += 1
            record.learnings = f"Change rejected: {reason}"
            notebook.add_dead_end(f"Iter {iteration}: {proposal.change_summary} — {reason}")

        _update_notebook(surfaces, notebook, proposal, record, iteration, reason)
        iteration_summaries.append(record.summary())

    # --- Report ---
    total_duration = time.time() - start_time
    best = tree.best_branch

    report = ExperimentReport(
        name=config.name,
        baseline_score=baseline_result.pass_rate,
        final_score=best.result.pass_rate if best else baseline_result.pass_rate,
        iterations_run=iterations_run,
        branches_created=len(tree.branches) - 1,  # exclude baseline
        branches_accepted=branches_accepted,
        branches_pruned=branches_pruned,
        merges_attempted=merges_attempted,
        merges_accepted=merges_accepted,
        best_variant_id=best.branch_id if best else "baseline",
        improvements=best.result.improvements_vs(baseline_result) if best else [],
        regressions=best.result.regressions_vs(baseline_result) if best else [],
        total_duration_sec=total_duration,
        tree_summary=tree.summary(),
        iteration_summaries=iteration_summaries,
    )

    (output_dir / "report.md").write_text(report.summary())
    print(f"\n{report.summary()}")

    return report


async def _try_merge(
    tree: SearchTree,
    config: ExperimentConfig,
    runner: HarborRunner,
    gatekeeper: Gatekeeper,
    notebook: ResearchNotebook,
    output_dir: Path,
    iteration: int,
    iteration_summaries: list[str],
) -> str | None:
    """Try merging top 2 accepted branches. Returns 'accepted', 'rejected', or None."""
    accepted_branches = [
        b for b in tree.active_branches
        if b.branch_id != "baseline" and b.score > tree.baseline_score
    ]
    if len(accepted_branches) < 2:
        return None

    b1, b2 = sorted(accepted_branches, key=lambda b: -b.score)[:2]
    if set(b1.variant.changed_surface_names()) == set(b2.variant.changed_surface_names()):
        return None  # Same surfaces changed — merge wouldn't be meaningful

    print(f"\n[meta] === Merge attempt: {b1.branch_id} + {b2.branch_id} ===")
    merged_variant = b1.variant.merge(b2.variant, f"merge_{iteration:03d}")
    merged_variant.save(output_dir / "variants" / f"{merged_variant.variant_id}.json")

    merged_config = RunConfig(
        variant=merged_variant, tasks=config.tasks,
        n_trials=config.n_trials_search, model=config.model,
        environment=config.environment, max_episodes=config.max_episodes,
        concurrency=config.concurrency,
    )
    merged_result = await runner.run_eval(merged_config, concurrency=config.concurrency)

    better_parent = b1 if b1.score >= b2.score else b2
    accepted, reason = gatekeeper.evaluate(merged_result, better_parent.result)

    if accepted:
        tree.add_branch(Branch(
            variant=merged_variant, result=merged_result,
            score=merged_result.pass_count,
            parent_branch_id=f"{b1.branch_id}+{b2.branch_id}",
            iteration_created=iteration,
        ))
        print(f"[meta] MERGE ACCEPTED: {reason}")
        notebook.add_finding(f"Merge {b1.branch_id}+{b2.branch_id}: {reason}")
        iteration_summaries.append(f"Merge [{b1.branch_id}+{b2.branch_id}] ACCEPTED: {reason}")
        return "accepted"
    else:
        print(f"[meta] MERGE REJECTED: {reason}")
        iteration_summaries.append(f"Merge [{b1.branch_id}+{b2.branch_id}] REJECTED: {reason}")
        return "rejected"


def _update_notebook(
    surfaces: list[Surface],
    notebook: ResearchNotebook,
    proposal,
    record: IterationRecord,
    iteration: int,
    reason: str,
) -> None:
    surface = next((s for s in surfaces if s.name == proposal.surface_name), None)
    if surface:
        surface.record_edit(iteration, record.accepted, proposal.change_summary)
        notebook.update_surface_risk(
            surface.name, surface.fragility,
            f"{'Accepted' if record.accepted else 'Rejected'} at iter {iteration}: {reason}"
        )
    notebook.record_iteration(record)
