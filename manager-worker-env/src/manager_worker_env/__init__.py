"""
Compatibility shim: canonical implementation lives in the top-level ``env`` package.
"""

from __future__ import annotations

import os
import sys

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

from env import (  # noqa: E402
    ManagerAction,
    ManagerWorkerEnv,
    ManagerWorkerObservation,
    ManagerWorkerState,
    WorkerStateModel,
)

__version__ = "1.0.0"
__all__ = [
    "ManagerWorkerEnv",
    "ManagerWorkerObservation",
    "ManagerAction",
    "ManagerWorkerState",
    "WorkerStateModel",
]
