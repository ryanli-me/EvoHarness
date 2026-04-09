"""Proposer agent: Claude Code as a researcher that investigates traces.

Unlike a single LLM call, we spawn Claude Code as a real agent with
filesystem access to traces, surfaces, and the research notebook.
It can grep, cat, and investigate interactively — just like Meta-Harness.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .notebook import ResearchNotebook
from .surfaces import Surface, Variant
from .traces import EvalResult, TraceStore


PROPOSAL_SCHEMA = {
    "type": "object",
    "properties": {
        "observation": {"type": "string", "description": "What you found in the traces"},
        "hypothesis": {"type": "string", "description": "Your testable hypothesis"},
        "surface_name": {"type": ["string", "null"], "description": "Which surface to change, or null to stop"},
        "new_value": {"type": "string", "description": "Complete new value for the surface"},
        "change_summary": {"type": "string", "description": "Brief description of the change"},
        "predicted_improvements": {"type": "array", "items": {"type": "string"}},
        "predicted_safe": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number"},
    },
    "required": ["observation", "hypothesis", "surface_name"],
}


@dataclass
class Proposal:
    """A proposed change from the proposer agent."""

    observation: str
    hypothesis: str
    surface_name: str
    new_value: str
    change_summary: str
    predicted_improvements: list[str]
    predicted_safe: list[str]
    confidence: float
    raw_response: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Proposal:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class ClaudeCodeProposer:
    """Proposer that spawns Claude Code as a real agent.

    Builds a workspace with traces, surfaces, and notebook, then
    lets Claude Code investigate freely with grep, cat, etc.
    """

    def __init__(
        self,
        model: str = "sonnet",
        max_turns: int = 30,
        max_budget_usd: float = 2.0,
    ):
        self.model = model
        self.max_turns = max_turns
        self.max_budget_usd = max_budget_usd

    def _build_workspace(
        self,
        workspace_dir: Path,
        surfaces: list[Surface],
        current_variant: Variant,
        eval_result: EvalResult,
        trace_store: TraceStore,
        notebook: ResearchNotebook,
        iteration: int,
    ) -> None:
        """Build the proposer workspace with all investigation materials."""
        if workspace_dir.exists():
            shutil.rmtree(workspace_dir)
        workspace_dir.mkdir(parents=True)

        # 1. Current surfaces with metadata
        surfaces_dir = workspace_dir / "surfaces"
        surfaces_dir.mkdir()
        manifest = {}
        for surface in surfaces:
            value = current_variant.surfaces.get(surface.name, surface.base_value)
            # Write surface value
            ext = ".txt" if surface.kind == "file" else ".py"
            filename = f"{surface.name}{ext}"
            (surfaces_dir / filename).write_text(value)
            # Write metadata
            risk = "HIGH" if surface.fragility > 0.5 else "MEDIUM" if surface.fragility > 0.2 else "LOW"
            manifest[surface.name] = {
                "file": filename,
                "kind": surface.kind,
                "target": surface.target,
                "description": surface.description,
                "risk": risk,
                "fragility": surface.fragility,
            }
        (surfaces_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

        # 2. Evaluation results summary
        results_dir = workspace_dir / "results"
        results_dir.mkdir()

        summary = {
            "pass_count": eval_result.pass_count,
            "total_count": eval_result.total_count,
            "pass_rate": eval_result.pass_rate,
            "by_difficulty": eval_result.by_difficulty(),
            "by_stratum": eval_result.by_stratum(),
        }
        (results_dir / "summary.json").write_text(json.dumps(summary, indent=2))

        # Failing tasks list
        failing = []
        for task in eval_result.failing_tasks():
            fail_trace = task.failing_trace()
            entry = {
                "task_id": task.task_id,
                "pass_rate": task.pass_rate,
                "difficulty": task.difficulty,
                "stratum": task.stratum,
                "n_trials": task.n_trials,
                "is_flaky": task.is_flaky,
            }
            if fail_trace:
                entry["n_steps"] = fail_trace.n_steps
                entry["error_summary"] = fail_trace.error_summary
                entry["last_commands"] = fail_trace.last_commands()
                entry["failure_step"] = fail_trace.failure_step
            failing.append(entry)
        (results_dir / "failing_tasks.json").write_text(json.dumps(failing, indent=2))

        # Passing tasks list (brief)
        passing = [
            {"task_id": t.task_id, "pass_rate": t.pass_rate, "difficulty": t.difficulty}
            for t in eval_result.task_results if t.passed
        ]
        (results_dir / "passing_tasks.json").write_text(json.dumps(passing, indent=2))

        # 3. Full traces for failing tasks (the key diagnostic data)
        traces_dir = workspace_dir / "traces"
        traces_dir.mkdir()
        for task in eval_result.failing_tasks():
            task_dir = traces_dir / task.task_id
            task_dir.mkdir()
            for trace in task.trials:
                trace_file = task_dir / f"trial_{trace.trial}.json"
                trace_file.write_text(json.dumps(trace.to_dict(), indent=2))

            # Also write differential if available
            diff = trace_store.get_differential(
                task.task_id, current_variant.variant_id, eval_result.split,
            )
            if diff:
                (task_dir / "differential.json").write_text(json.dumps(diff, indent=2))

        # 4. Research notebook
        notebook_dir = workspace_dir / "notebook"
        shutil.copytree(notebook.root, notebook_dir, dirs_exist_ok=True)

        # 5. TASK.md — the instructions for Claude Code
        task_md = f"""# Harness Optimization — Iteration {iteration}

You are a researcher optimizing an AI agent's harness for TerminalBench-2.
Your workspace contains everything you need to investigate and propose a change.

## Your workspace

- `surfaces/` — Current harness surfaces (prompt, tools, bootstrap, completion logic)
  - `surfaces/manifest.json` — Metadata, risk ratings, descriptions for each surface
- `results/` — Current evaluation results
  - `results/summary.json` — Overall pass rate, by difficulty, by category
  - `results/failing_tasks.json` — List of failing tasks with error summaries
  - `results/passing_tasks.json` — List of passing tasks
- `traces/` — Full execution traces for FAILING tasks
  - `traces/{{task_id}}/trial_0.json` — Complete step-by-step trace
  - Each trace has: tool calls, commands, outputs, errors
  - `traces/{{task_id}}/differential.json` — Passing vs failing comparison (if available)
- `notebook/` — Research notebook from prior iterations
  - `notebook/findings.md` — Confirmed improvements
  - `notebook/dead_ends.md` — Failed approaches (DO NOT repeat these)
  - `notebook/surface_risk.md` — Which surfaces are safe vs fragile
  - `notebook/iterations/` — Full records of each prior iteration

## Your job

1. **Investigate**: Read the failing traces. Use grep to find patterns across tasks.
   Look at what commands failed, what errors occurred, where the agent got stuck.
2. **Hypothesize**: Form a specific hypothesis about why tasks are failing.
3. **Propose**: Choose ONE surface to change and write the complete new value.

## Rules

- Make ONE change to ONE surface per iteration
- Prefer ADDITIVE changes over modifications to existing logic
- Check `notebook/dead_ends.md` — do NOT repeat failed approaches
- Check `surfaces/manifest.json` — HIGH risk surfaces have caused regressions before
- Write real, working code/prompts. Not pseudocode.
- The new value must be a complete DROP-IN REPLACEMENT for the surface file

## Output

When you're done investigating, write your proposal to `proposal.json` with this structure:
```json
{{
    "observation": "What you found in the traces",
    "hypothesis": "Your specific, testable hypothesis",
    "surface_name": "name of the surface to change (from manifest.json)",
    "new_value": "the COMPLETE new value for the surface",
    "change_summary": "brief description of what you changed",
    "predicted_improvements": ["task_ids that should improve"],
    "predicted_safe": ["task_ids that should NOT regress"],
    "confidence": 0.7
}}
```

If you believe no useful change can be made, write `{{"surface_name": null}}`.
"""
        (workspace_dir / "TASK.md").write_text(task_md)

    async def propose(
        self,
        surfaces: list[Surface],
        current_variant: Variant,
        train_result: EvalResult,
        trace_store: TraceStore,
        notebook: ResearchNotebook,
        iteration: int,
    ) -> Proposal | None:
        """Spawn Claude Code to investigate traces and propose a change."""
        workspace_dir = trace_store.root.parent / "proposer_workspaces" / f"iter_{iteration:03d}"
        self._build_workspace(
            workspace_dir, surfaces, current_variant,
            train_result, trace_store, notebook, iteration,
        )

        prompt = (
            "Read TASK.md for your instructions. Investigate the failing traces in traces/, "
            "check the notebook for prior learnings, then write your proposal to proposal.json. "
            "Be thorough — grep across traces for common patterns before proposing."
        )

        cmd = [
            "claude",
            "-p", prompt,
            "--model", self.model,
            "--output-format", "text",
            "--max-budget-usd", str(self.max_budget_usd),
            "--permission-mode", "auto",
            "--allowed-tools", "Read", "Glob", "Grep", "Write", "Bash(grep:*,cat:*,head:*,tail:*,wc:*,sort:*,uniq:*,jq:*,find:*,ls:*)",
        ]

        print(f"[proposer] Spawning Claude Code (model={self.model}, budget=${self.max_budget_usd})...")

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(workspace_dir),
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=600,  # 10 min max
            )

            raw_output = stdout.decode()

            # Save agent output for debugging
            (workspace_dir / "agent_stdout.txt").write_text(raw_output)
            (workspace_dir / "agent_stderr.txt").write_text(stderr.decode())

            # Read the proposal file the agent wrote
            proposal_path = workspace_dir / "proposal.json"
            if not proposal_path.exists():
                print("[proposer] Agent did not write proposal.json")
                # Try to parse from stdout as fallback
                return self._parse_from_stdout(raw_output)

            data = json.loads(proposal_path.read_text())

            if data.get("surface_name") is None:
                print("[proposer] Agent decided no change is needed")
                return None

            return Proposal(
                observation=data.get("observation", ""),
                hypothesis=data.get("hypothesis", ""),
                surface_name=data["surface_name"],
                new_value=data["new_value"],
                change_summary=data.get("change_summary", ""),
                predicted_improvements=data.get("predicted_improvements", []),
                predicted_safe=data.get("predicted_safe", []),
                confidence=data.get("confidence", 0.5),
                raw_response=raw_output[:5000],
            )

        except asyncio.TimeoutError:
            print("[proposer] Claude Code timed out after 600s")
            return None
        except Exception as e:
            print(f"[proposer] Error: {e}")
            return None

    def _parse_from_stdout(self, stdout: str) -> Proposal | None:
        """Fallback: try to extract proposal JSON from stdout."""
        try:
            # Look for JSON block in output
            if "```json" in stdout:
                json_str = stdout.split("```json")[1].split("```")[0]
            elif "{" in stdout and "surface_name" in stdout:
                start = stdout.index("{")
                depth = 0
                for i in range(start, len(stdout)):
                    if stdout[i] == "{": depth += 1
                    elif stdout[i] == "}": depth -= 1
                    if depth == 0:
                        json_str = stdout[start:i+1]
                        break
                else:
                    return None
            else:
                return None

            data = json.loads(json_str)
            if data.get("surface_name") is None:
                return None

            return Proposal(
                observation=data.get("observation", ""),
                hypothesis=data.get("hypothesis", ""),
                surface_name=data["surface_name"],
                new_value=data["new_value"],
                change_summary=data.get("change_summary", ""),
                predicted_improvements=data.get("predicted_improvements", []),
                predicted_safe=data.get("predicted_safe", []),
                confidence=data.get("confidence", 0.5),
                raw_response=stdout[:5000],
            )
        except (json.JSONDecodeError, ValueError):
            return None


# Keep the simple proposer as a lightweight alternative
class LiteLLMProposer:
    """Lightweight proposer using a single LLM call (no agent)."""

    def __init__(self, model: str = "anthropic/claude-sonnet-4-6", max_tokens: int = 16000):
        self.model = model
        self.max_tokens = max_tokens

    async def propose(
        self,
        surfaces: list[Surface],
        current_variant: Variant,
        train_result: EvalResult,
        trace_store: TraceStore,
        notebook: ResearchNotebook,
        iteration: int,
    ) -> Proposal | None:
        import litellm

        # Build compact context
        parts = ["# Surfaces\n"]
        for s in surfaces:
            risk = "HIGH" if s.fragility > 0.5 else "MEDIUM" if s.fragility > 0.2 else "LOW"
            value = current_variant.surfaces.get(s.name, s.base_value)
            parts.append(f"## {s.name} [{risk}]\n{s.description}\n```\n{value[:2000]}\n```\n")

        parts.append(f"# Results: {train_result.pass_count}/{train_result.total_count}\n")
        for t in train_result.failing_tasks()[:8]:
            ft = t.failing_trace()
            parts.append(f"FAIL: {t.task_id} ({t.difficulty}) — {ft.error_summary[:200] if ft and ft.error_summary else 'unknown'}")

        parts.append(f"\n{notebook.build_proposer_context()}")

        system = (
            "You optimize agent harnesses. Propose ONE change to ONE surface. "
            "Respond with JSON: {observation, hypothesis, surface_name, new_value, change_summary, "
            "predicted_improvements, predicted_safe, confidence}"
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": "\n".join(parts)},
        ]

        for attempt in range(3):
            try:
                resp = await litellm.acompletion(
                    model=self.model, messages=messages,
                    max_tokens=self.max_tokens, temperature=0.7 if attempt == 0 else 0.3,
                    timeout=300,
                )
                content = resp.choices[0].message.content or ""
                json_str = content
                if "```json" in content:
                    json_str = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    json_str = content.split("```")[1].split("```")[0]

                data = json.loads(json_str.strip())
                if data.get("surface_name") is None:
                    return None

                return Proposal(
                    observation=data.get("observation", ""),
                    hypothesis=data.get("hypothesis", ""),
                    surface_name=data["surface_name"],
                    new_value=data["new_value"],
                    change_summary=data.get("change_summary", ""),
                    predicted_improvements=data.get("predicted_improvements", []),
                    predicted_safe=data.get("predicted_safe", []),
                    confidence=data.get("confidence", 0.5),
                    raw_response=content,
                )
            except json.JSONDecodeError:
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": "Invalid JSON. Respond with ONLY a JSON object."})
            except Exception as e:
                print(f"[proposer] Error: {e}")
                if attempt == 2:
                    return None
        return None


class MockProposer:
    """Mock proposer for offline demos."""

    def __init__(self):
        self._proposals = [
            Proposal(
                observation="Tasks fail because agent explores before acting.",
                hypothesis="Adding task-type strategies to the prompt will reduce wasted turns.",
                surface_name="system_prompt",
                new_value=(
                    "You are an AI assistant tasked with solving command-line tasks.\n\n"
                    "## Strategy Guide\n"
                    "- FILE tasks: check paths exist, use absolute paths, verify after\n"
                    "- COMPILATION: read errors carefully, fix one at a time\n"
                    "- CONFIG: back up before editing, verify syntax\n"
                    "- NEVER repeat a failing command\n\n"
                    "Task Description:\n{instruction}\n\nCurrent terminal state:\n{terminal_state}\n"
                ),
                change_summary="Added task-type strategy guide to system prompt",
                predicted_improvements=[], predicted_safe=[], confidence=0.7,
            ),
            Proposal(
                observation="Agent declares complete without verifying.",
                hypothesis="Requiring a verification command will reduce premature completions.",
                surface_name="system_prompt",
                new_value=(
                    "You are an AI assistant tasked with solving command-line tasks.\n\n"
                    "## Strategy Guide\n"
                    "- FILE tasks: check paths exist, use absolute paths, verify after\n"
                    "- COMPILATION: read errors carefully, fix one at a time\n"
                    "- CONFIG: back up before editing, verify syntax\n"
                    "- NEVER repeat a failing command\n\n"
                    "## Before task_complete\n"
                    "1. Re-read the task description\n"
                    "2. Run a verification command proving the task is done\n"
                    "3. Check no extra files were created\n\n"
                    "Task Description:\n{instruction}\n\nCurrent terminal state:\n{terminal_state}\n"
                ),
                change_summary="Added verification protocol before task_complete",
                predicted_improvements=[], predicted_safe=[], confidence=0.6,
            ),
        ]
        self._index = 0

    async def propose(self, surfaces, current_variant, train_result, trace_store, notebook, iteration):
        if self._index >= len(self._proposals):
            return None
        proposal = self._proposals[self._index]
        self._index += 1
        return proposal
