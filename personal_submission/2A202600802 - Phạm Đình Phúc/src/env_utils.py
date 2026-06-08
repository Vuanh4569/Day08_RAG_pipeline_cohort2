"""Environment helpers that work even when python-dotenv is not installed."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_DIR = Path(__file__).parent.parent


def load_local_env(env_path: Path | None = None) -> None:
    """Load simple KEY=VALUE lines from .env without overwriting existing env vars."""
    path = env_path or PROJECT_DIR / ".env"
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def get_env(name: str, default: str = "") -> str:
    load_local_env()
    return os.getenv(name, default)
