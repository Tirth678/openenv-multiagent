"""
ManagerAgent: thin wrapper around a trained PPO policy that emits ManagerActions
for the ManagerWorkerEnv.

Two interchangeable backends:
- ``PPOManagerAgent``: loads a stable-baselines3 PPO checkpoint produced by
  ``training/train_manager.py``.
- ``HeuristicManagerAgent``: cheap rule-based fallback (assign → check →
  approve) useful for smoke-testing the wired-up environment without a trained
  model.

The agent operates on the dict observation produced by the gym wrapper used in
training (see ``training/train_manager.py``), but also accepts the raw
``ManagerWorkerObservation`` for convenience.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import numpy as np

from env import ManagerAction, ManagerWorkerObservation


def _to_dict_obs(obs: Union[ManagerWorkerObservation, Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """Coerce either an observation object or a dict into the dict format SB3 expects."""
    if isinstance(obs, ManagerWorkerObservation):
        return {
            "task_embedding": np.array(obs.task_embedding, dtype=np.float32),
            "worker_states": np.array(obs.worker_states, dtype=np.float32),
            "num_workers": np.array([obs.num_workers], dtype=np.float32),
            "subtask_status": np.array(obs.subtask_status, dtype=np.int8),
            "num_subtasks": np.array([obs.num_subtasks], dtype=np.float32),
            "budget_remaining": np.array([obs.budget_remaining], dtype=np.float32),
            "steps_remaining": np.array([obs.steps_remaining], dtype=np.float32),
        }
    return obs


class HeuristicManagerAgent:
    """
    Stateless rule-of-thumb manager: cycle through assign → check → approve.

    Useful as a sanity baseline against the trained PPO policy and for
    end-to-end smoke tests when no checkpoint is available.
    """

    ACTION_CYCLE = [0, 1, 5]  # assign_subtask, check_worker_output, approve_output

    def __init__(self) -> None:
        self._step = 0
        self.last_thought = "Initialized."

    def reset(self) -> None:
        self._step = 0
        self.last_thought = "Resetting..."

    def predict(
        self,
        obs: Union[ManagerWorkerObservation, Dict[str, Any]],
        deterministic: bool = True,
    ) -> ManagerAction:
        action_id = self.ACTION_CYCLE[self._step % len(self.ACTION_CYCLE)]
        self.last_thought = f"Cycle step {self._step % len(self.ACTION_CYCLE)}: taking action {action_id}."
        self._step += 1
        return ManagerAction(action_id=action_id)


class ParallelManagerAgent:
    """
    Observation-driven manager that actually *uses* the worker pool in parallel.

    Strategy:
      1. While there are unassigned subtasks AND idle workers, ``assign_subtask``.
         (This fans the work out across as many workers as the env allows.)
      2. Once everyone is busy (or no work is left to assign), iterate through
         active workers: ``check_worker_output`` then ``approve_output`` /
         ``correct_worker`` based on the revealed quality.
      3. If a checked output's revealed quality is below ``correct_threshold``
         we ``correct_worker`` once before approving on the next pass.

    This produces a clearly multi-agent execution trace where multiple workers
    are simultaneously ``is_active=True`` with different output buffers.
    """

    # Action ids — keep in sync with ManagerWorkerEnv.ACTION_NAMES
    ASSIGN = 0
    CHECK = 1
    CORRECT = 2
    APPROVE = 5

    def __init__(self, correct_threshold: float = 0.5, max_corrections_per_worker: int = 1) -> None:
        self.correct_threshold = correct_threshold
        self.max_corrections_per_worker = max_corrections_per_worker
        self._corrections: Dict[int, int] = {}
        self.last_thought = "Initialized."

    def reset(self) -> None:
        self._corrections = {}
        self.last_thought = "Resetting..."

    @staticmethod
    def _unpack_obs(obs: Union[ManagerWorkerObservation, Dict[str, Any]]):
        """Return (worker_rows, n_workers_real, subtask_status, real_subtask_count)."""
        # Default to "size of subtask_status" / 4 worker rows when the new fields
        # aren't surfaced (older obs payloads, or hand-built observations).

        def _coerce_norm(raw) -> float:
            try:
                return float(raw[0])
            except (TypeError, IndexError):
                return float(raw)

        if isinstance(obs, ManagerWorkerObservation):
            sub = list(obs.subtask_status)
            n_sub_real = (
                max(1, int(round(float(obs.num_subtasks) * len(sub))))
                if obs.num_subtasks
                else len(sub)
            )
            rows = obs.worker_states
            n_w_real = (
                max(1, int(round(float(obs.num_workers) * len(rows))))
                if getattr(obs, "num_workers", 0)
                else len(rows)
            )
            return rows, n_w_real, sub, n_sub_real

        sub = list(obs["subtask_status"])
        if "num_subtasks" in obs:
            norm = _coerce_norm(obs["num_subtasks"])
            n_sub_real = max(1, int(round(norm * len(sub)))) if norm > 0 else len(sub)
        else:
            n_sub_real = len(sub)
        rows = obs["worker_states"]
        if "num_workers" in obs:
            norm = _coerce_norm(obs["num_workers"])
            n_w_real = max(1, int(round(norm * len(rows)))) if norm > 0 else len(rows)
        else:
            n_w_real = len(rows)
        return rows, n_w_real, sub, n_sub_real

    def predict(
        self,
        obs: Union[ManagerWorkerObservation, Dict[str, Any]],
        deterministic: bool = True,
    ) -> ManagerAction:
        worker_rows, n_workers, subtask_status, n_real = self._unpack_obs(obs)
        # Only consider the real workers / subtasks; the rest is padding.
        real_status = subtask_status[:n_real]
        real_rows = worker_rows[:n_workers]

        # Worker_state row layout from _generate_observation:
        #   [is_active, progress, hallucination_risk_score, output_quality_if_checked, tokens_consumed_ratio]
        idle_workers = [i for i, row in enumerate(real_rows) if float(row[0]) < 0.5]
        active_workers = [i for i, row in enumerate(real_rows) if float(row[0]) >= 0.5]

        # subtask_status tells us what is *complete*; we approximate "in flight"
        # as the number of currently active workers, so we only assign more if
        # there are subtasks nobody has picked up yet (otherwise we'd burn
        # tokens on no-op assigns once every subtask has an owner).
        incomplete = sum(1 for s in real_status if int(s) == 0)
        not_yet_started = max(0, incomplete - len(active_workers))

        # Phase 1: fan out — assign a subtask to an idle worker, but only while
        # there is work that nobody has started yet.
        if idle_workers and not_yet_started > 0:
            self.last_thought = f"Assigning work to idle worker. {not_yet_started} subtasks remaining to start."
            return ManagerAction(action_id=self.ASSIGN)

        # Phase 2: look at the first active worker and decide check / correct / approve.
        for wid in active_workers:
            row = real_rows[wid]
            checked_quality = float(row[3])
            # If we've never seen this worker's quality, check it first.
            if checked_quality <= 0.0:
                self.last_thought = f"Checking output of active worker {wid} to reveal quality."
                return ManagerAction(action_id=self.CHECK, target_worker_id=wid)
            # Quality revealed and bad — try a correction (bounded).
            if (
                checked_quality < self.correct_threshold
                and self._corrections.get(wid, 0) < self.max_corrections_per_worker
            ):
                self._corrections[wid] = self._corrections.get(wid, 0) + 1
                self.last_thought = f"Worker {wid} quality ({checked_quality:.2f}) low. Sending correction."
                return ManagerAction(action_id=self.CORRECT, target_worker_id=wid)
            # Otherwise approve and free the worker.
            self.last_thought = f"Approving worker {wid} output (quality {checked_quality:.2f})."
            return ManagerAction(action_id=self.APPROVE, target_worker_id=wid)

        # No idle workers, no active workers, no unassigned subtasks → done. Approve as a no-op-ish.
        self.last_thought = "No more work to assign. Waiting for workers or idle."
        return ManagerAction(action_id=self.APPROVE)


class PPOManagerAgent:
    """
    Loads a PPO checkpoint produced by ``training/train_manager.py`` and emits
    ManagerActions for the env.
    """

    def __init__(self, model_path: str):
        # Defer the heavy import so the file is cheap to import for callers
        # that only want the heuristic baseline.
        from stable_baselines3 import PPO  # type: ignore

        self.model = PPO.load(model_path)
        self.model_path = model_path
        self.last_thought = "Model loaded."

    def reset(self) -> None:
        # PPO policies are stateless w.r.t. the env between episodes here; the
        # underlying RNG is handled by SB3.
        self.last_thought = "Resetting policy..."
        pass

    def predict(
        self,
        obs: Union[ManagerWorkerObservation, Dict[str, Any]],
        deterministic: bool = True,
    ) -> ManagerAction:
        action, _ = self.model.predict(_to_dict_obs(obs), deterministic=deterministic)
        self.last_thought = f"PPO Policy selected action {action} based on current observation."
        return ManagerAction(action_id=int(action))


def load_manager_agent(
    model_path: Optional[str] = None,
    parallel: bool = False,
) -> "HeuristicManagerAgent | ParallelManagerAgent | PPOManagerAgent":
    """Convenience constructor.

    Priority: ``model_path`` > ``parallel`` > heuristic.
    """
    if model_path:
        return PPOManagerAgent(model_path)
    if parallel:
        return ParallelManagerAgent()
    return HeuristicManagerAgent()


__all__ = [
    "HeuristicManagerAgent",
    "ParallelManagerAgent",
    "PPOManagerAgent",
    "load_manager_agent",
]
