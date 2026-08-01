"""Compatibility runner for direct execution.

Prefer `uvicorn deploy.main:app --reload`, but keep this file so older local
commands like `python deploy/app.py` continue to work.
"""
from __future__ import annotations

import sys
from pathlib import Path

import uvicorn

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deploy.main import app  # noqa: F401


if __name__ == "__main__":
    uvicorn.run(
        "deploy.main:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True,
        reload_includes=["**/*.py", "**/*.html"],
    )
