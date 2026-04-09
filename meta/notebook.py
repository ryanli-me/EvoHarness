"""Research notebook for accumulating knowledge across iterations.

Unlike better-harness's /history/ (just decisions) or Meta-Harness's raw
filesystem (unstructured), the notebook is curated knowledge that grows
over time. The proposer reads it to avoid repeating failed ideas and
to build on confirmed findings.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class IterationRecord:
    """Record of one optimization iteration."""

    iteration: int
    hypothesis: str
    surfaces_changed: list[str]
    change_summary: str
    variant_id: str
    parent_id: str

    # Results
    train_pass_rate: float | None = None
    holdout_pass_rate: float | None = None
    train_improvements: list[str] = field(default_factory=list)
    train_regressions: list[str] = field(default_factory=list)
    holdout_improvements: list[str] = field(default_factory=list)
    holdout_regressions: list[str] = field(default_factory=list)

    # Decision
    accepted: bool = False
    rejection_reason: str | None = None

    # Learnings
    learnings: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IterationRecord:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def summary(self) -> str:
        """One-paragraph summary for the proposer."""
        status = "ACCEPTED" if self.accepted else "REJECTED"
        parts = [
            f"Iteration {self.iteration} [{status}]: {self.hypothesis}",
            f"Changed: {', '.join(self.surfaces_changed)}",
            f"Train: {self.train_pass_rate:.1%}" if self.train_pass_rate is not None else "",
            f"Holdout: {self.holdout_pass_rate:.1%}" if self.holdout_pass_rate is not None else "",
        ]
        if self.train_regressions:
            parts.append(f"Regressions: {', '.join(self.train_regressions[:5])}")
        if self.train_improvements:
            parts.append(f"Improvements: {', '.join(self.train_improvements[:5])}")
        if not self.accepted and self.rejection_reason:
            parts.append(f"Rejected because: {self.rejection_reason}")
        if self.learnings:
            parts.append(f"Learnings: {self.learnings}")
        return "\n".join(p for p in parts if p)


class ResearchNotebook:
    """Persistent research notebook that grows across iterations.

    Layout:
        notebook_root/
            iterations/
                001.json
                002.json
            findings.md       # Confirmed findings
            dead_ends.md      # Things that don't work
            surface_risk.md   # Which surfaces are safe vs fragile
    """

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "iterations").mkdir(exist_ok=True)
        self._ensure_files()

    def _ensure_files(self) -> None:
        for name in ("findings.md", "dead_ends.md", "surface_risk.md"):
            path = self.root / name
            if not path.exists():
                path.write_text(f"# {name.replace('.md', '').replace('_', ' ').title()}\n\n")

    def record_iteration(self, record: IterationRecord) -> None:
        """Save an iteration record."""
        path = self.root / "iterations" / f"{record.iteration:03d}.json"
        path.write_text(json.dumps(record.to_dict(), indent=2))

    def get_iteration(self, iteration: int) -> IterationRecord | None:
        path = self.root / "iterations" / f"{iteration:03d}.json"
        if not path.exists():
            return None
        return IterationRecord.from_dict(json.loads(path.read_text()))

    def get_all_iterations(self) -> list[IterationRecord]:
        records = []
        iter_dir = self.root / "iterations"
        for path in sorted(iter_dir.glob("*.json")):
            records.append(IterationRecord.from_dict(json.loads(path.read_text())))
        return records

    def add_finding(self, finding: str) -> None:
        """Append a confirmed finding."""
        path = self.root / "findings.md"
        content = path.read_text()
        content += f"\n- {finding}"
        path.write_text(content)

    def add_dead_end(self, dead_end: str) -> None:
        """Record something that doesn't work."""
        path = self.root / "dead_ends.md"
        content = path.read_text()
        content += f"\n- {dead_end}"
        path.write_text(content)

    def update_surface_risk(self, surface_name: str, fragility: float, notes: str) -> None:
        """Update surface risk assessment."""
        path = self.root / "surface_risk.md"
        content = path.read_text()
        # Remove existing entry for this surface
        lines = content.split("\n")
        lines = [l for l in lines if not l.startswith(f"- **{surface_name}**")]
        content = "\n".join(lines)
        risk_label = "HIGH" if fragility > 0.5 else "MEDIUM" if fragility > 0.2 else "LOW"
        content += f"\n- **{surface_name}** [{risk_label} risk, fragility={fragility:.2f}]: {notes}"
        path.write_text(content)

    def build_proposer_context(self) -> str:
        """Build the full notebook context for the proposer agent.

        Returns a structured markdown document with all accumulated knowledge.
        """
        parts = ["# Research Notebook\n"]

        # Iteration history
        iterations = self.get_all_iterations()
        if iterations:
            parts.append("## Iteration History\n")
            for record in iterations:
                parts.append(record.summary())
                parts.append("")

        # Findings
        findings = (self.root / "findings.md").read_text()
        if findings.strip() != "# Findings":
            parts.append(findings)

        # Dead ends
        dead_ends = (self.root / "dead_ends.md").read_text()
        if dead_ends.strip() != "# Dead Ends":
            parts.append(dead_ends)

        # Surface risk
        surface_risk = (self.root / "surface_risk.md").read_text()
        if surface_risk.strip() != "# Surface Risk":
            parts.append(surface_risk)

        return "\n\n".join(parts)
