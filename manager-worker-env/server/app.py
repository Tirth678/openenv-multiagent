#!/usr/bin/env python3
"""
OpenEnv-compatible server entry: delegates to the Orchestra FastAPI app.

The previous `BE.*` layout is removed; use ``orchestra_backend`` (``python -m orchestra_backend.main``)
or ``uvicorn orchestra_backend.main:app`` from the ``manager-worker-env`` directory on PYTHONPATH.

Usage:
    python -m server.app
    uvicorn server.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
import sys

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from orchestra_backend.main import app, main  # noqa: E402

__all__ = ["app", "main"]
