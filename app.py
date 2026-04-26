"""
Vercel + monorepo entrypoint.

Vercel builds from the repository root. The real FastAPI app lives under
`manager-worker-env/orchestra_backend/`. This file re-exports `app` so the
Python runtime can discover it.
"""

import importlib.util
import sys
from pathlib import Path

ORCH = Path(__file__).resolve().parent / "manager-worker-env" / "orchestra_backend"
sys.path.insert(0, str(ORCH))

MAIN = ORCH / "main.py"
spec = importlib.util.spec_from_file_location("orchestra_backend_main", MAIN)
mod = importlib.util.module_from_spec(spec)
if spec.loader is None:
    raise RuntimeError(f"Failed to load {MAIN}")
sys.modules["orchestra_backend_main"] = mod
spec.loader.exec_module(mod)
app = mod.app
