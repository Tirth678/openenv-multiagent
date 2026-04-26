"""Smoke tests: imports and app factory (no Mongo required)."""

from __future__ import annotations

import os
import sys


def _ensure_path() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)


def test_env_import() -> None:
    _ensure_path()
    from env import ManagerWorkerEnv

    env = ManagerWorkerEnv(
        {
            "max_workers": 2,
            "max_steps": 20,
            "token_budget": 1000,
            "task_difficulty": 2,
            "failure_injection_rate": 0.4,
        }
    )
    obs = env.reset()
    assert obs is not None


def test_orchestra_app_loads() -> None:
    _ensure_path()
    from orchestra_backend.main import app

    assert app.title


def test_health_endpoint() -> None:
    _ensure_path()
    from fastapi.testclient import TestClient
    from orchestra_backend.main import app

    with TestClient(app) as client:
        r = client.get("/health")
    assert r.status_code == 200
    assert r.json().get("status") == "healthy"
