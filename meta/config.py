"""Configuration loading from TOML files."""

from __future__ import annotations

import tomllib
from pathlib import Path

from .core import ExperimentConfig
from .runner import TaskCase


def load_config(config_path: Path) -> ExperimentConfig:
    """Load experiment configuration from a TOML file."""
    with open(config_path, "rb") as f:
        raw = tomllib.load(f)

    exp = raw["experiment"]
    harness_root = Path(exp.get("harness_root", config_path.parent))

    tasks = []
    for case in raw.get("cases", []):
        tasks.append(TaskCase(
            task_id=case["task_id"],
            split=case.get("split", "all"),
            difficulty=case.get("difficulty"),
            stratum=case.get("stratum"),
        ))

    output_dir = Path(exp.get("output_dir", f"runs/{exp['name']}"))

    return ExperimentConfig(
        name=exp["name"],
        harness_root=harness_root.resolve(),
        output_dir=output_dir.resolve(),
        tasks=tasks,
        max_iterations=exp.get("max_iterations", 10),
        n_trials_search=exp.get("n_trials_search", 1),
        n_trials_final=exp.get("n_trials_final", 5),
        n_canaries=exp.get("n_canaries", 5),
        regression_tolerance=exp.get("regression_tolerance", 1),
        model=exp.get("model", "anthropic/claude-opus-4-6"),
        proposer_model=exp.get("proposer_model", exp.get("model", "anthropic/claude-opus-4-6")),
        environment=exp.get("environment", "runloop"),
        max_episodes=exp.get("max_episodes", 20),
        concurrency=exp.get("concurrency", 20),
    )
