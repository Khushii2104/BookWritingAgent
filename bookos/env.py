"""Tiny .env loader with no dependencies, so a missing python-dotenv can never hide your API key."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_env() -> Path | None:
    """Load KEY=VALUE lines from .env (project folder first, then the current folder).
    Real environment variables win. Returns the file that was used, or None."""
    for path in (PROJECT_ROOT / ".env", Path.cwd() / ".env"):
        if not path.is_file():
            continue
        for raw in path.read_text(encoding="utf-8-sig").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.removeprefix("export ").strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                value = value[1:-1]
            elif " #" in value:
                value = value.split(" #", 1)[0].strip()
            if key and value and not os.environ.get(key):
                os.environ[key] = value
        return path
    return None
