"""Validate the Sign-Robust configuration registry using only the stdlib."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: top-level JSON must be an object")
    return value


def validate() -> list[str]:
    registry_path = ROOT / "manifest.json"
    registry = load_json(registry_path)
    errors: list[str] = []
    ids: set[str] = set()
    for rel in registry.get("configs", []):
        path = ROOT / rel
        if not path.exists():
            errors.append(f"missing config: {rel}")
            continue
        try:
            config = load_json(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(str(exc))
            continue
        experiment_id = config.get("experiment_id")
        if not experiment_id:
            errors.append(f"{rel}: missing experiment_id")
        elif experiment_id in ids:
            errors.append(f"duplicate experiment_id: {experiment_id}")
        else:
            ids.add(experiment_id)
        for field in ("paper_reference", "status", "purpose", "implementation"):
            if field not in config:
                errors.append(f"{rel}: missing {field}")
        implementation = config.get("implementation", {})
        path_keys = {"generator", "identifier", "source_snapshot", "case_runner", "matrix_runner", "partition_source", "legacy_sources"}
        for key in path_keys:
            values = implementation.get(key, []) if isinstance(implementation, dict) else []
            if isinstance(values, str):
                values = [values]
            for value in values:
                candidate = (ROOT / value).resolve()
                if not candidate.exists():
                    errors.append(f"{rel}: implementation path does not exist: {value}")
    shared = ROOT / registry.get("shared_config", "")
    if not shared.exists():
        errors.append(f"missing shared_config: {registry.get('shared_config')}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    errors = validate()
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    registry = load_json(ROOT / "manifest.json")
    print(f"validated {len(registry['configs'])} Sign-Robust configurations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
