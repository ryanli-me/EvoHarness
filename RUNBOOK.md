# Meta-Harness Runbook

## Goal
Automatically discover harness improvements over KIRA+env_bootstrap (76.4%) on TerminalBench-2.

## Steps

### Step 1: Baseline Eval
Run all 89 tasks × 1 trial to get pass/fail map + traces.

```bash
source .env && harbor run \
  --agent-import-path agent:AgentHarness \
  -d terminal-bench@2.0 \
  -m anthropic/claude-opus-4-6 \
  -e modal \
  -k 1 -n 89 \
  --override-cpus 8 --override-memory-mb 16384 \
  --timeout-multiplier 3.0 \
  -o jobs/baseline-modal --job-name baseline \
  -y --env-file .env
```

**Status**: RUNNING (PID 79270)
**Expected**: ~76% pass rate, ~20 failing tasks, ~30 min on Modal

### Step 2: Import Traces
Parse Harbor job output into our TraceStore so the proposer can read them.

```bash
python -m meta import jobs/baseline-modal/baseline --output runs/experiment
```

**TODO**: Write the import script (`meta/import_job.py`)

### Step 3: Tree Search Loop
Run the optimization loop with parallel proposals.

```bash
python -m meta run meta/configs/kira_tbench2.toml
```

What happens per iteration:
1. Thompson sampling selects N branches to extend
2. N Claude Code proposers spawn in parallel, each reading traces
3. Each proposes ONE surface change with hypothesis
4. All candidates pre-screened on 3 failing tasks (parallel, on Modal)
5. Best candidate gets full incremental eval (failing + 5 canaries)
6. Gatekeeper accepts/rejects
7. Notebook updated with findings/dead ends
8. Every 3 iterations: try merging top 2 accepted branches

**Config**: 10 iterations, 3 parallel proposals, 3 prescreen tasks

### Step 4: Final Eval
Run best variant on all 89 tasks × 5 trials for official score.

```bash
source .env && harbor run \
  --agent-import-path agent:AgentHarness \
  -d terminal-bench@2.0 \
  -m anthropic/claude-opus-4-6 \
  -e modal \
  -k 5 -n 89 \
  --override-cpus 8 --override-memory-mb 16384 \
  --timeout-multiplier 3.0 \
  -o jobs/final --job-name final \
  -y --env-file .env
```

Uses the modified harness from Step 3 (best variant applied).

### Step 5: Compare
- Baseline score (Step 1) vs Final score (Step 4)
- What surfaces changed, what hypotheses were confirmed
- Full report in `runs/experiment/report.md`

## Cost Estimate

| Step | Tasks | Trials | Est. Cost |
|------|-------|--------|-----------|
| Baseline | 89 | 1 | ~$250 |
| Per search iteration | ~25 (failing+canaries) | 1 | ~$75 |
| 10 iterations | ~250 total | 1 | ~$750 |
| Final eval | 89 | 5 | ~$1,250 |
| Proposer (10 iter × 3 parallel) | — | — | ~$30 |
| **Total** | | | **~$2,280** |

## Immediate TODOs

- [ ] Wait for baseline to finish (Step 1)
- [ ] Write import script to parse Harbor → TraceStore
- [ ] Update config to use Modal + correct concurrency
- [ ] Run the loop
