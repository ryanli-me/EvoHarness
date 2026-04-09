# Experiment Log: Automated Harness Optimization on TerminalBench-2

## Timeline

```
12:30  Baseline launched (89 tasks × 1 trial, Modal, 8CPU/16GB)
13:00  Baseline ~done: 61/84 = 72.6% (5 tasks still running)
13:05  Proposer 1 launched (Claude Code sonnet, $3 budget, focus: system_prompt)
13:10  Proposer 1 done — proposed verification protocol for system_prompt
13:15  Iter 1 eval launched (23 failing tasks on Modal, all parallel)
13:30  Iter 1 partial: 7 flipped
13:35  Proposer 2 launched (Claude Code sonnet, $3 budget, focus: env_bootstrap)
13:40  Proposer 2 done — proposed expanded environment detection
13:42  Iter 2 eval launched (23 failing tasks on Modal, all parallel)
13:45  Iter 1: 8 flipped, 4 running. Iter 2: 1 flipped, 22 running
```

## Environment

- **Inner agent**: KIRA + env_bootstrap (agent.py, the Meta-Harness artifact)
- **Model**: Claude Opus 4.6
- **Benchmark**: TerminalBench-2 (89 tasks)
- **Search eval**: Modal (8 CPU, 16GB RAM, 20GB storage, 3x timeout)
- **Official submission**: Default resources (1 CPU, 2GB, default timeout, -k 5)
- **Proposer**: Claude Code (sonnet), local process, $3 budget per proposal

## Editable Surfaces

| Surface | Risk | Description |
|---------|------|-------------|
| system_prompt | LOW | 12-line prompt template. Most room for improvement. |
| tool_definitions | LOW | 3 tool schemas + descriptions. Descriptions are the lever. |
| env_bootstrap | LOW | Environment snapshot before agent loop. |
| completion_logic | HIGH | Double-confirmation checklist. Meta-Harness failed 3/4 times. |

## Phase 0: Baseline

**Command**:
```bash
harbor run --agent-import-path agent:AgentHarness \
  -d terminal-bench@2.0 -m anthropic/claude-opus-4-6 \
  -e modal -k 1 -n 89 \
  --override-cpus 8 --override-memory-mb 16384 --override-storage-mb 20480 \
  --timeout-multiplier 3.0 \
  -o jobs/baseline-modal --job-name bm3 -y --env-file .env
```

**Results**: 88/89 done (1 timeout: train-fasttext)
- 61 passed, 27 failed
- Easy: 100%, Medium: 77%, Hard: 64%
- Worst categories: scientific-computing (38%), file-operations (50%)

**Data saved**: `jobs/baseline-modal/bm3/` (Harbor output), `runs/experiment/traces/` (imported)

---

## Phase 1: Tree Search

### Iteration 1: System Prompt — Verification Protocol

**Branch**: iter_001 (from baseline)
**Surface changed**: system_prompt
**Proposer**: Claude Code sonnet, ~$1.50, ~5 min

**What the proposer found**:
- Read traces of 23 failing tasks
- Pattern: agents call task_complete without running the test command mentioned in the task
- Evidence: `build-pmars` had explicit test command `pmars -b -r 50 -f flashpaper.red rave.red | tail -n 1`, agent never ran it
- `sqlite-with-gcov` agent deleted test-generated `.gcda` files before completing

**Proposed change**: Added mandatory pre-completion verification:
1. Run the task's test command if one is provided
2. Verify required outputs exist
3. Don't delete solution files

**Harness**: `runs/experiment/iter_001/harness/` (only prompt-templates/terminus-kira.txt changed)
**Proposal**: `runs/experiment/iter_001/proposal.json`
**Proposer workspace**: `runs/experiment/proposer_workspaces/iter_001/`

**Eval command**: 23 failing tasks, Modal, all parallel
**Harbor output**: `jobs/iter1/iter1/`

**Results** (19/23 done, 4 hard tasks still running):
```
FLIPPED TO PASS (8):
  adaptive-rejection-sampler  (medium/scientific-computing)  ✓
  cancel-async-tasks          (hard/software-engineering)    ✓
  extract-elf                 (medium/file-operations)       ✓
  make-mips-interpreter       (hard/software-engineering)    ✓
  protein-assembly            (hard/scientific-computing)    ✓
  raman-fitting               (medium/scientific-computing)  ✓
  sanitize-git-repo           (medium/security)              ✓
  torch-pipeline-parallelism  (hard/software-engineering)    ✓

STILL FAILING (11):
  build-pmars, caffe-cifar-10, db-wal-recovery, dna-assembly, dna-insert,
  filter-js-from-html, model-extraction-relu-logits, mteb-retrieve,
  sam-cell-seg, sqlite-with-gcov, video-processing

STILL RUNNING (4):
  gpt2-codegolf, install-windows-3.11, make-doom-for-mips, mteb-leaderboard
```

**Projected score**: 61 + 8 = 69/84 = 82.1% (up from 72.6%)

---

### Iteration 2: Env Bootstrap — Expanded Environment Detection

**Branch**: iter_002 (from iter_001 — includes the prompt change)
**Surface changed**: env_bootstrap
**Proposer**: Claude Code sonnet, ~$1.50, ~5 min

**What the proposer found**:
- Many failing tasks involve specialized tools/libraries the agent wastes turns discovering
- Agent tries `import numpy` and fails, tries `pip install numpy`, etc.
- Current bootstrap only checks basic language versions

**Proposed change**: Expanded `_gather_env_snapshot()` to also detect:
- GPU availability (nvidia-smi)
- Disk space (df -h /app)
- 20 common Python packages (numpy, scipy, torch, biopython, cv2, PIL, ffmpeg, etc.)
- Extra tools (ffmpeg, git, make, mips-linux-gnu-gcc, curl, wget)

**Harness**: `runs/experiment/iter_002/harness/` (agent.py modified — env_bootstrap code block replaced)
**Proposal**: `runs/experiment/iter_002/proposer_workspace/proposal.json`

**Eval**: 23 failing tasks, Modal, all parallel
**Harbor output**: `runs/experiment/iter_002/harbor_jobs/iter_002/`

**Results** (1/23 done, 22 running):
- 1 flipped so far, too early to tell

---

## Search Tree

```
baseline (61/88 = 69.3%)
│
├── iter_001: system_prompt + verification protocol
│   Status: 8 flipped (19/23 done), LIKELY ACCEPT
│   Projected: 69/88 = 78.4%
│   │
│   └── iter_002: env_bootstrap + expanded detection (branched FROM iter_001)
│       Status: 1 flipped (1/23 done), RUNNING
│
├── (iter_003: tool_definitions?) — NOT YET STARTED
│
└── merge(001 + 002 if both accepted) — LATER
```

## Next Steps

1. Wait for iter 1 + iter 2 to finish
2. Accept/reject each branch
3. If iter 2 accepted: merge iter_001 + iter_002 changes
4. Consider iter_003 on tool_definitions
5. **Final eval on default resources** (1 CPU, 2GB, -k 5) — required for leaderboard

## Important Notes

- Search evals use overridden resources (8 CPU, 16GB) for speed
- **Leaderboard submissions must use default resources**: `harbor run -d terminal-bench@2.0 -a "agent" -m "model" -k 5`
- Results need validation on default resources before claiming improvement
- Each task's timeout varies (15 min to 3.3 hours)

## Cost Tracking

| Item | Cost |
|------|------|
| Baseline (89 tasks × 1 trial) | ~$250 |
| Proposer 1 | ~$1.50 |
| Iter 1 eval (23 tasks × 1 trial) | ~$69 |
| Proposer 2 | ~$1.50 |
| Iter 2 eval (23 tasks × 1 trial) | ~$69 |
| **Running total** | **~$391** |
