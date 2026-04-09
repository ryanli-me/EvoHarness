# Meta-Harness: Tree Search for Automated Harness Optimization

## The Problem

Agent harnesses (prompts, tools, bootstrapping code) are manually engineered.
Meta-Harness showed an LLM can optimize these, but used linear hill climbing:
8 iterations, 7 wasted, 1 useful change found.

## Our Approach: Tree Search with Parallel Proposals

```
                         ┌──────────────┐
                         │   BASELINE   │
                         │  75% pass    │
                         │  20 failing  │
                         └──────┬───────┘
                                │
                    ┌───────────┼───────────┐
                    ▼           ▼           ▼
              ┌──────────┐ ┌──────────┐ ┌──────────┐
              │Proposer A│ │Proposer B│ │Proposer C│   ← 3 Claude Code
              │(sonnet)  │ │(sonnet)  │ │(sonnet)  │     agents in parallel
              │reads     │ │reads     │ │reads     │     each reads traces,
              │traces,   │ │traces,   │ │traces,   │     greps for patterns,
              │diagnoses │ │diagnoses │ │diagnoses │     proposes 1 change
              └────┬─────┘ └────┬─────┘ └────┬─────┘
                   │            │            │
                   ▼            ▼            ▼
              ┌──────────┐ ┌──────────┐ ┌──────────┐
              │Prescreen │ │Prescreen │ │Prescreen │   ← Run each proposal
              │3 failing │ │3 failing │ │3 failing │     on 3 failing tasks
              │tasks     │ │tasks     │ │tasks     │     (cheap filter)
              └────┬─────┘ └────┬─────┘ └────┬─────┘
                   │            │            │
                   ▼            ▼            ▼
              1/3 pass     0/3 pass     2/3 pass
              ┌──────┐     ┌──────┐     ┌──────┐
              │MAYBE │     │PRUNE │     │ BEST │        ← Pick best prescreen
              └──────┘     └──────┘     └──┬───┘
                                           │
                                           ▼
                                    ┌──────────────┐
                                    │  Full eval    │   ← Run on ALL 20
                                    │  20 failing   │     failing tasks
                                    │  + 5 canaries │     + 5 passing canaries
                                    └──────┬───────┘
                                           │
                                           ▼
                                    ┌──────────────┐
                                    │  Gatekeeper   │   ← Accept if:
                                    │  pass↑ reg≤1  │     more tasks pass AND
                                    └──────┬───────┘     regressions ≤ 1
                                           │
                                    ┌──────┴──────┐
                                    ▼             ▼
                              ┌──────────┐  ┌──────────┐
                              │ ACCEPT   │  │ REJECT   │
                              │ new      │  │ dead end │
                              │ branch   │  │ logged   │
                              └──────────┘  └──────────┘
```

## After Multiple Iterations: The Search Tree

```
baseline (75%)
├── branch_001: system_prompt + error recovery (77%) ✓ ACCEPTED
│   └── branch_004: + budget awareness (78%) ✓ ACCEPTED
├── branch_002: env_bootstrap + read README (76%) ✓ ACCEPTED
├── branch_003: tool_definitions change → PRUNED at prescreen
└── merge(001, 002): error recovery + read README (79%) ✓ MERGED

Best variant: merge_006 (79%)
```

## Key Innovation: Why Tree > Linear

```
Meta-Harness (linear):
  baseline → reject → reject → reject → reject → reject → reject → ACCEPT → neutral
  Cost: 8 full evals. Found: 1 improvement.

Our framework (tree):
  baseline ─┬─ branch A (prescreen 3 tasks) → PRUNE (cost: 3 tasks)
            ├─ branch B (prescreen 3 tasks) → promising → full eval → ACCEPT
            ├─ branch C (prescreen 3 tasks) → PRUNE (cost: 3 tasks)
            └─ branch D (prescreen 3 tasks) → promising → full eval → ACCEPT
            └─ merge(B,D) → full eval → ACCEPT
  Cost: 4×3 + 3×20 = 72 task evals. Found: 3 improvements.
```

## Components

### 1. Surfaces (what we can edit)

```
┌─────────────────────────────────────────────────────────┐
│ system_prompt          [LOW risk]                        │
│ "You are an AI assistant tasked with solving..."         │
│ 12 lines, very minimal. Most room for improvement.      │
├─────────────────────────────────────────────────────────┤
│ tool_definitions       [LOW risk]                        │
│ execute_commands, task_complete, image_read               │
│ 3 tool schemas + description strings.                    │
├─────────────────────────────────────────────────────────┤
│ env_bootstrap          [LOW risk]                        │
│ _gather_env_snapshot() — pwd, ls, language versions       │
│ Could detect project type, read README, etc.             │
├─────────────────────────────────────────────────────────┤
│ completion_logic       [HIGH risk — 3/4 regressions]     │
│ Double-confirmation checklist before task_complete.       │
│ Meta-Harness tried 3 times, regressed each time.         │
└─────────────────────────────────────────────────────────┘
```

### 2. Proposer (Claude Code agent)

Each proposer gets a workspace:
```
proposer_workspace/
  TASK.md              ← Instructions: investigate, hypothesize, propose
  surfaces/            ← Current surface values + risk ratings
  results/             ← Pass/fail summary, failing task list
  traces/              ← FULL execution traces for every failing task
    task_name/
      trial_0.json     ← Every command, output, error, LLM reasoning
      differential.json ← Passing vs failing comparison
  notebook/            ← Accumulated knowledge from prior iterations
    findings.md        ← What works
    dead_ends.md       ← What doesn't (DO NOT REPEAT)
    surface_risk.md    ← Which surfaces are safe to edit
```

The proposer can grep, cat, read any file — full investigation before proposing.

### 3. Research Notebook (memory across iterations)

```
Iteration 1: Tried adding error recovery to prompt → ACCEPTED (+2 tasks)
Iteration 2: Tried verification protocol → PRUNED (prescreen 0/3)
Iteration 3: Tried modifying completion logic → REJECTED (2 regressions)
             → completion_logic fragility: 0.75 → HIGH risk

findings.md:
  - Error recovery guidance in prompt helps (iter 1)

dead_ends.md:
  - Verification protocol doesn't help failing tasks (iter 2)
  - Modifying completion logic causes regressions (iter 3)

surface_risk.md:
  - system_prompt [LOW risk] — 1 accepted / 1 rejected
  - completion_logic [HIGH risk] — 0 accepted / 1 rejected
```

### 4. Incremental Eval (cost optimization)

```
Full eval:     89 tasks × $3/task = $267 per iteration
Our approach:  20 failing + 5 canaries × $3/task = $75 per iteration
                                                    ↓
                                               3.6x cheaper
```

Only re-run failing tasks. Carry forward passing results.
If a canary (previously passing task) fails → regression detected.

### 5. Thompson Sampling (branch selection)

Each branch has a Beta(passed, failed) distribution.
Sample from each, pick the highest → balances exploration vs exploitation.

```
baseline:    Beta(60, 20) → sample ~0.74
branch_001:  Beta(62, 18) → sample ~0.78  ← likely picked
branch_002:  Beta(61, 19) → sample ~0.71
```

New branches with few evals have high variance → get explored.
Strong branches get exploited. Weak branches naturally die out.

## Prior Art Comparison

| | Meta-Harness | better-harness | AutoHarness | Ours |
|--|-------------|----------------|-------------|------|
| Search | Linear hill climb | Linear hill climb | Thompson tree | Thompson tree + parallel |
| Proposer | Claude Code agent | LangChain Deep Agent | Gemini + critic | Claude Code agent |
| Pre-screening | No | No | 10 envs × 1000 steps | 3 tasks (cheap) |
| Branch merging | No | No | No | Yes |
| Fragility tracking | No | No | No | Yes |
| Dead ends memory | No | Accept/reject only | No | Full notebook |
| Incremental eval | No (full 89 tasks) | No | N/A | Yes (failing + canaries) |
| Cost per iteration | ~$267 | ~$267 | N/A | ~$75 |
