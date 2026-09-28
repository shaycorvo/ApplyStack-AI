"""Resolve runtime storage paths, with an isolated root for automated tests."""

from __future__ import annotations

import os
from pathlib import Path


def data_path(*parts: str) -> Path:
    """Return a path under the configured runtime data root."""
    return Path(os.getenv("APPLYAGENT_DATA_DIR", "data")).joinpath(*parts)