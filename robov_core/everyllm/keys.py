from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

_DEFAULT_KEYS = ["agnes", "glm", "opencode", "anyapi"]


def load_keys(keys_path: Optional[str | Path] = None) -> dict[str, str]:
    if keys_path is not None:
        p = Path(keys_path)
        if p.exists():
            return _read_keys(p)
        return {}

    candidates = [
        Path.cwd() / "KEYS.json",
        Path(__file__).resolve().parent.parent.parent / "KEYS.json",
        Path(__file__).resolve().parent.parent / "KEYS.json",
    ]
    for p in candidates:
        if p.exists():
            return _read_keys(p)

    _create_empty_keys(candidates[0])
    return {}


def _create_empty_keys(path: Path) -> None:
    try:
        data = {k: "" for k in _DEFAULT_KEYS}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except OSError:
        pass


def _read_keys(path: Path) -> dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if v}
