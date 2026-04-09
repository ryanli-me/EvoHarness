# The Process: How an Iteration Should Work

## The Correct Flow

```
Step 1: BASELINE (once)
  Run all 89 tasks × 1 trial, DEFAULT resources
  → Get pass/fail map + full traces
  → This is our ground truth

Step 2: PROPOSE
  Claude Code agent reads failing traces + notebook
  → Proposes ONE change to ONE surface
  → Writes proposal.json

Step 3: APPLY
  apply_variant_to_harness() creates a complete harness directory
  → Self-contained, runnable with `harbor run`

Step 4: PRESCREEN (cheap filter)
  Run proposal on 3 randomly-selected FAILING tasks, DEFAULT resources
  → If 0/3 flip → PRUNE. Cost: $9. Done.
  → If 1+ flip → proceed to Step 5

Step 5: EVAL FAILING TASKS
  Run proposal on ALL failing tasks (~20), DEFAULT resources
  → Count how many flip

Step 6: EVAL CANARIES (regression check)
  Run proposal on 5 randomly-selected PASSING tasks, DEFAULT resources
  → If any canary fails → likely regression

Step 7: GATEKEEPER
  Compare: flips vs regressions
  → ACCEPT if: pass_count improves AND regressions ≤ tolerance (1)
  → REJECT otherwise

Step 8: UPDATE
  If ACCEPTED:
    → notebook/findings.md: what worked
    → surface fragility stays low
    → This variant becomes the new parent for next iteration

  If REJECTED:
    → notebook/dead_ends.md: what failed and WHY
    → surface fragility increases
    → Parent variant unchanged, try different approach

Step 9: REPEAT from Step 2
```

## What We Did Wrong in Our Experiment

```
✗ Baseline ran on OVERRIDDEN resources (8 CPU, 16GB)
  → Should have been DEFAULT (1 CPU, 2GB)

✗ Branch evals only ran on failing tasks, no canaries
  → Missed 8 regressions that showed up later

✗ Both proposals auto-accepted without regression check
  → Should have run canaries before accepting

✗ Ran proposer + eval manually instead of through the loop
  → Bypassed gatekeeper, notebook updates, fragility tracking
```

## What Would Have Happened With the Correct Process

```
Step 1: Baseline (all 89, default resources)
  → 61 passed, 28 failed

Step 2-3: Proposer → verification protocol for system_prompt

Step 4: Prescreen (3 failing tasks, default resources)
  → 1/3 flip → proceed

Step 5: Eval failing (28 tasks, default resources)
  → Some flip (maybe 5-7 on default resources, not 9)

Step 6: Canaries (5 passing tasks, default resources)
  → Maybe 1 fails → regression detected

Step 7: Gatekeeper
  → If 6 flips, 1 regression → net +5 → ACCEPT
  → If 3 flips, 2 regressions → net +1 but over tolerance → REJECT
     → notebook records: "verification protocol too aggressive,
        causes timeout on simple tasks"
     → proposer next iteration: make it conditional

Step 8: If rejected, proposer reads dead_ends.md
  → Proposes softer version: "only verify if task provides test command"
  → This targeted version might flip 5 tasks with 0 regressions
```

## Key Principle

**Every eval must be on DEFAULT resources.** Otherwise you're optimizing
for a phantom benchmark that doesn't match the leaderboard. Our 9 flips
on big resources became 7 flips + 8 regressions on default — the change
was net negative.

## The Loop in Code

This is what `python -m meta run` does automatically:

```python
for iteration in range(max_iterations):
    # Select branch (Thompson sampling)
    parent = tree.select_branch()

    # N proposers in parallel
    proposals = await gather([propose(parent) for _ in range(N)])

    # Prescreen all (3 tasks each, parallel)
    for proposal in proposals:
        prescreen = await eval(proposal, 3_failing_tasks)
        if prescreen.pass_count == 0:
            prune(proposal)  # $9 wasted, not $267

    # Full eval on best surviving proposal
    best = pick_best_prescreen(proposals)
    result = await eval(best, all_failing + 5_canaries)

    # Gatekeeper
    if result.pass_count > parent.pass_count and regressions <= 1:
        accept(best)  # new branch in tree
        notebook.add_finding(...)
    else:
        reject(best)  # dead end
        notebook.add_dead_end(...)
        surface.fragility += 1

    # Periodically merge top branches
    if iteration % 3 == 0:
        merge(branch_a, branch_b)
```

## For the Final Submission

```bash
# Run best harness on official conditions
cd runs/experiment/best_variant/harness
harbor run -d terminal-bench@2.0 -m anthropic/claude-opus-4-6 -k 5
# No overrides. This is the real score.
```
