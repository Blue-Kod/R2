from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


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
    return {}


def _read_keys(path: Path) -> dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if v}
