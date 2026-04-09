"""Surface abstraction for editable harness components.

A Surface is one discrete, editable piece of the agent harness — a prompt,
a tool definition, a code module, etc. A Variant is a concrete set of
surface values that can be applied to produce a modified harness.

Inspired by better-harness (LangChain) but extended with fragility tracking.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Surface:
    """One editable component of the agent harness.

    Attributes:
        name: Human-readable identifier (e.g., "system_prompt").
        kind: Either "file" (a file to replace) or "code_block" (a named
              section within a file, delimited by markers).
        target: For "file" kind: relative path from harness root.
                For "code_block": "filepath::block_name".
        description: What this surface controls, shown to the proposer.
        base_value: The original content of this surface.
        fragility: 0.0-1.0 score tracking how often edits cause regressions.
                   Updated automatically by the gatekeeper.
        edit_history: List of (iteration, accepted, summary) tuples.
    """

    name: str
    kind: str  # "file" | "code_block"
    target: str
    description: str
    base_value: str
    fragility: float = 0.0
    edit_history: list[tuple[int, bool, str]] = field(default_factory=list)

    def record_edit(self, iteration: int, accepted: bool, summary: str) -> None:
        """Record an edit attempt and update fragility score."""
        self.edit_history.append((iteration, accepted, summary))
        # Fragility = fraction of rejected edits (exponentially weighted recent)
        if len(self.edit_history) >= 2:
            recent = self.edit_history[-5:]  # last 5 edits
            rejections = sum(1 for _, acc, _ in recent if not acc)
            self.fragility = rejections / len(recent)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "target": self.target,
            "description": self.description,
            "base_value": self.base_value,
            "fragility": self.fragility,
            "edit_history": self.edit_history,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Surface:
        return cls(**data)


@dataclass
class Variant:
    """A concrete set of surface values — one candidate harness configuration.

    Attributes:
        variant_id: Unique identifier (e.g., "baseline", "iter_003").
        surfaces: Mapping of surface name -> current value.
        changes: Mapping of surface name -> value, only for surfaces that
                 differ from baseline. Empty for the baseline variant.
        hypothesis: The proposer's hypothesis for why this variant is better.
        parent_id: The variant this was derived from.
    """

    variant_id: str
    surfaces: dict[str, str]
    changes: dict[str, str] = field(default_factory=dict)
    hypothesis: str = ""
    parent_id: str | None = None

    @classmethod
    def baseline(cls, surface_defs: list[Surface]) -> Variant:
        """Create the baseline variant from surface definitions."""
        return cls(
            variant_id="baseline",
            surfaces={s.name: s.base_value for s in surface_defs},
        )

    def derive(
        self,
        variant_id: str,
        changes: dict[str, str],
        hypothesis: str = "",
    ) -> Variant:
        """Create a child variant with specific surface changes."""
        new_surfaces = copy.deepcopy(self.surfaces)
        new_surfaces.update(changes)
        return Variant(
            variant_id=variant_id,
            surfaces=new_surfaces,
            changes=changes,
            hypothesis=hypothesis,
            parent_id=self.variant_id,
        )

    def changed_surface_names(self) -> list[str]:
        """Return names of surfaces that differ from parent."""
        return list(self.changes.keys())

    def merge(self, other: Variant, variant_id: str) -> Variant:
        """Merge changes from another variant into this one.

        Combines all surface changes from both variants.
        If both change the same surface, `other` wins.
        """
        merged_changes = {**self.changes, **other.changes}
        merged_surfaces = copy.deepcopy(self.surfaces)
        merged_surfaces.update(other.changes)
        return Variant(
            variant_id=variant_id,
            surfaces=merged_surfaces,
            changes=merged_changes,
            hypothesis=f"Merge of {self.variant_id} + {other.variant_id}",
            parent_id=self.variant_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "surfaces": self.surfaces,
            "changes": self.changes,
            "hypothesis": self.hypothesis,
            "parent_id": self.parent_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Variant:
        return cls(**data)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> Variant:
        return cls.from_dict(json.loads(path.read_text()))


def load_surfaces_from_harness(harness_root: Path) -> list[Surface]:
    """Load surface definitions for the KIRA agent.

    This extracts the editable surfaces from the existing harness code:
    - system_prompt: The prompt template
    - tool_definitions: The 3 tool schemas
    - env_bootstrap: The _gather_env_snapshot() method
    - completion_logic: The double-confirmation checklist
    - output_config: Output limiting and duration defaults
    """
    surfaces = []

    # 1. System prompt
    prompt_path = harness_root / "prompt-templates" / "terminus-kira.txt"
    if prompt_path.exists():
        surfaces.append(Surface(
            name="system_prompt",
            kind="file",
            target="prompt-templates/terminus-kira.txt",
            description=(
                "The system prompt template injected before each task. "
                "Contains task description placeholder {instruction} and "
                "terminal state placeholder {terminal_state}. Currently 12 lines, "
                "very minimal. High potential for task-adaptive strategies."
            ),
            base_value=prompt_path.read_text(),
        ))

    # 2. Tool definitions — extracted from agent.py
    agent_path = harness_root / "agent.py"
    if agent_path.exists():
        agent_code = agent_path.read_text()

        # Extract the TOOLS list and associated description strings
        # Find from _EXECUTE_COMMANDS_DESC to end of TOOLS list
        tools_start = agent_code.find("# Tool description strings")
        tools_end = agent_code.find("\n\n\nclass AgentHarness")
        if tools_start >= 0 and tools_end >= 0:
            tools_block = agent_code[tools_start:tools_end]
            surfaces.append(Surface(
                name="tool_definitions",
                kind="code_block",
                target="agent.py::tool_definitions",
                description=(
                    "The 3 tool schemas (execute_commands, task_complete, image_read) "
                    "and their description strings. Controls what tools the LLM can "
                    "call and how they're described. Adding new tools or improving "
                    "descriptions could help."
                ),
                base_value=tools_block,
            ))

        # 3. Env bootstrap — the _gather_env_snapshot method
        bootstrap_start = agent_code.find("    async def _gather_env_snapshot")
        bootstrap_end = agent_code.find("    async def _run_agent_loop")
        if bootstrap_start >= 0 and bootstrap_end >= 0:
            bootstrap_block = agent_code[bootstrap_start:bootstrap_end].rstrip()
            surfaces.append(Surface(
                name="env_bootstrap",
                kind="code_block",
                target="agent.py::env_bootstrap",
                description=(
                    "Gathers environment snapshot before agent loop: working dir, "
                    "file listing, available languages, package managers, memory. "
                    "Injected into initial prompt. Could detect task type, check "
                    "more tools, or do deeper analysis."
                ),
                base_value=bootstrap_block,
            ))

        # 4. Completion confirmation message
        confirm_start = agent_code.find("    def _get_completion_confirmation_message")
        confirm_end = agent_code.find("    def _limit_output_length")
        if confirm_start >= 0 and confirm_end >= 0:
            confirm_block = agent_code[confirm_start:confirm_end].rstrip()
            surfaces.append(Surface(
                name="completion_logic",
                kind="code_block",
                target="agent.py::completion_logic",
                description=(
                    "The double-confirmation checklist shown when agent calls "
                    "task_complete. Currently a generic checklist. Could be made "
                    "task-specific or include actual verification commands. "
                    "WARNING: Meta-Harness found this surface fragile — edits "
                    "caused regressions in 3/4 attempts."
                ),
                base_value=confirm_block,
                fragility=0.75,  # Pre-seeded from Meta-Harness experience
            ))

    return surfaces


_HARNESS_FILES = [
    "agent.py",
    "anthropic_caching.py",
    "prompt-templates/terminus-kira.txt",
    "pyproject.toml",
]


def apply_variant_to_harness(
    variant: Variant,
    surface_defs: list[Surface],
    harness_root: Path,
    output_root: Path,
) -> Path:
    """Apply a variant's surface values to produce a modified harness.

    Copies only the essential harness files (not .venv, runs, etc.)
    to output_root and applies all surface changes.
    Returns the path to the modified harness.
    """
    import shutil

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Copy only essential harness files
    for rel in _HARNESS_FILES:
        src = harness_root / rel
        dst = output_root / rel
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    surface_map = {s.name: s for s in surface_defs}

    for surface_name, value in variant.surfaces.items():
        surface = surface_map.get(surface_name)
        if surface is None:
            continue

        if surface.kind == "file":
            target_path = output_root / surface.target
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(value)

        elif surface.kind == "code_block":
            filepath, block_name = surface.target.split("::")
            target_path = output_root / filepath
            if target_path.exists():
                code = target_path.read_text()
                base = surface.base_value
                if base in code:
                    code = code.replace(base, value)
                    target_path.write_text(code)

    return output_root
