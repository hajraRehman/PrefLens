"""Cross-method convergence study of LLM preference elicitation.

Importing this package loads `.env` from the repository root if present, so API
credentials never need to be exported by hand. Values are read into the process
environment only and are never logged or written to any output file.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_dotenv(path: Path | None = None) -> None:
    """Minimal KEY=VALUE loader. Existing environment variables always win."""
    p = path or (ROOT / ".env")
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


load_dotenv()
