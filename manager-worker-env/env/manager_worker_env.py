"""
ManagerWorkerEnv: Core OpenEnv environment for multi-agent coordination.

This environment simulates a workplace where one Manager Agent coordinates
multiple Worker Agents to complete complex multi-step tasks. Workers can fail
in realistic ways (hallucinations, off-task, incomplete, stuck), forcing the
Manager to learn detection and correction strategies under a limited token budget.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, TYPE_CHECKING
import logging

import numpy as np
from pydantic import BaseModel, Field

from openenv.core import Environment, State, Observation, Action

from env.task_library import TaskLibrary, Task, Subtask
from env.hallucination_engine import HallucinationEngine, WorkerOutput
from env.reward_calculator import RewardCalculator

if TYPE_CHECKING:
    from agents.worker_pool import WorkerPool

logger = logging.getLogger(__name__)


# ============================================================================
# Pydantic Models for OpenEnv
# ============================================================================

class WorkerStateModel(BaseModel):
    """Represents the state of a single worker agent."""
    
    worker_id: int
    is_active: bool = False
    current_subtask_id: Optional[int] = None
    progress: float = 0.0  # 0.0 to 1.0
    hallucination_risk_score: float = 0.0  # 0.0 to 1.0 (surface plausibility — what the manager sees pre-check)
    output_quality_if_checked: float = 0.0  # 0.0 to 1.0 (revealed only after a check action)
    tokens_consumed: int = 0
    failure_mode: Optional[str] = None  # 'hallucination', 'off_task', 'incomplete', 'stuck', None
    output_buffer: str = ""
    patience_counter: int = 0  # For detecting stuck loops
    skill_level: float = 0.5  # 0.3 to 1.0
    # Internal ground truth (not directly exposed in observation):
    actual_quality: float = 0.0
    is_checked: bool = False
    has_output: bool = False


class ManagerWorkerObservation(Observation):
    """Observation from the environment."""
    
    task_embedding: List[float] = Field(..., description="64-dimensional task embedding")
    worker_states: List[List[float]] = Field(..., description="4x5 worker states array (padded)")
    num_workers: float = Field(default=1.0, description="Number of real workers in this env, normalized to [0, 1] over the 4-row max")
    subtask_status: List[int] = Field(..., description="Binary array of subtask completion (padded to MAX_OBSERVABLE_SUBTASKS)")
    num_subtasks: float = Field(default=0.0, description="Number of real subtasks for this task, normalized to [0, 1]")
    budget_remaining: float = Field(..., description="Remaining budget as fraction [0, 1]")
    steps_remaining: float = Field(..., description="Remaining steps as fraction [0, 1]")
    episode_log: List[Dict[str, Any]] = Field(default_factory=list, description="Episode log")
    hallucination_catch_rate: float = Field(default=0.0, description="Hallucination detection rate")


class ManagerAction(Action):
    """Action from the Manager Agent."""
    
    action_id: int = Field(..., ge=0, le=6, description="Action ID (0-6)")
    target_worker_id: Optional[int] = Field(default=None, description="Target worker ID if applicable")


class ManagerWorkerState(State):
    """Internal state of the environment."""
    
    workers: List[WorkerStateModel] = Field(default_factory=list)
    task: Optional[Dict[str, Any]] = None
    budget_remaining: int = 1000
    step_counter: int = 0
    episode_log: List[Dict[str, Any]] = Field(default_factory=list)
    subtask_status: List[bool] = Field(default_factory=list)
    subtask_assignments: List[Optional[int]] = Field(default_factory=list)
    hallucination_catch_rate: float = 0.0
    hallucinations_detected: int = 0
    hallucinations_approved: int = 0
    false_positives: int = 0
    max_workers: int = 4
    max_steps: int = 50
    token_budget: int = 1000
    task_difficulty: int = 3
    failure_injection_rate: float = 0.6


# ============================================================================
# Main Environment Class
# ============================================================================

class ManagerWorkerEnv(Environment):
    """
    Multi-agent RL environment with Manager coordinating Workers using OpenEnv.
    
    Inherits from openenv.core.Environment and implements the OpenEnv interface.
    """

    # How many subtasks the observation exposes. Tasks in the library have up
    # to 5 subtasks today; we keep some headroom so adding longer tasks doesn't
    # silently hide work from the manager.
    MAX_OBSERVABLE_SUBTASKS = 8

    # Action names for reference
    ACTION_NAMES = [
        'assign_subtask',      # 0: 10 tokens
        'check_worker_output', # 1: 50 tokens
        'correct_worker',      # 2: 30 tokens
        'reassign_task',       # 3: 40 tokens
        'fire_and_replace',    # 4: 100 tokens
        'approve_output',      # 5: 5 tokens
        'request_clarification', # 6: 20 tokens
    ]
    
    ACTION_COSTS = {
        0: 10,   # assign_subtask
        1: 50,   # check_worker_output
        2: 30,   # correct_worker
        3: 40,   # reassign_task
        4: 100,  # fire_and_replace
        5: 5,    # approve_output
        6: 20,   # request_clarification
    }
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        worker_pool: Optional["WorkerPool"] = None,
    ):
        """
        Initialize environment with configuration.
        
        Args:
            config: Dictionary with keys:
                - max_workers: int (default 4, range 1-4)
                - max_steps: int (default 50, range 10-100)
                - token_budget: int (default 1000, range 500-5000)
                - task_difficulty: int (default 3, range 1-5)
                - failure_injection_rate: float (default 0.6, range 0.0-1.0)
            worker_pool: Optional WorkerPool of real LLM workers. When provided,
                workers actually generate text via Hugging Face models on
                ``assign_subtask`` and ``correct_worker`` actions; quality and
                failure type are still scored by ``HallucinationEngine`` for the
                Manager's reward signal. When ``None`` the env runs in
                simulator-only mode (suitable for fast PPO training).
        """
        super().__init__()
        
        # Configuration
        config = config or {}
        self.max_workers = config.get('max_workers', 4)
        self.max_steps = config.get('max_steps', 50)
        self.token_budget = config.get('token_budget', 1000)
        self.task_difficulty = config.get('task_difficulty', 3)
        self.failure_injection_rate = config.get('failure_injection_rate', 0.6)
        
        # Validate configuration
        assert 1 <= self.max_workers <= 4, "max_workers must be in [1, 4]"
        assert 10 <= self.max_steps <= 100, "max_steps must be in [10, 100]"
        assert 500 <= self.token_budget <= 5000, "token_budget must be in [500, 5000]"
        assert 1 <= self.task_difficulty <= 5, "task_difficulty must be in [1, 5]"
        assert 0.0 <= self.failure_injection_rate <= 1.0, "failure_injection_rate must be in [0.0, 1.0]"
        
        # Initialize components
        self.task_library = TaskLibrary()
        self.hallucination_engine = HallucinationEngine()
        self.reward_calculator = RewardCalculator()

        # Optional real LLM worker pool. If supplied, the env will actually
        # invoke the LLMs when assigning subtasks. Quality scoring still goes
        # through the hallucination engine so reward signal stays consistent.
        self.worker_pool = worker_pool
        if self.worker_pool is not None and len(self.worker_pool.workers) < self.max_workers:
            logger.warning(
                "WorkerPool has %d workers but env max_workers=%d; clamping max_workers to pool size.",
                len(self.worker_pool.workers),
                self.max_workers,
            )
            self.max_workers = len(self.worker_pool.workers)

        # The current task's Subtask objects (kept on the env, not in the
        # serialisable Pydantic state, so we can use them at action time).
        self._current_subtasks: List[Subtask] = []

        # Initialize state
        self._state = ManagerWorkerState(
            max_workers=self.max_workers,
            max_steps=self.max_steps,
            token_budget=self.token_budget,
            task_difficulty=self.task_difficulty,
            failure_injection_rate=self.failure_injection_rate,
        )
    
    @property
    def state(self) -> ManagerWorkerState:
        """Get the current environment state."""
        return self._state
    
    @state.setter
    def state(self, value: ManagerWorkerState) -> None:
        """Set the environment state."""
        self._state = value
    
    def reset(self, task_id: Optional[str] = None) -> ManagerWorkerObservation:
        """
        Reset environment to initial state.
        
        Args:
            task_id: If set, use this task from the library; otherwise sample by difficulty.
        
        Returns:
            observation: ManagerWorkerObservation with initial state
        """
        if task_id:
            t = self.task_library.get_task_by_id(task_id)
            task = t if t is not None else self.task_library.sample_task(
                difficulty=self.task_difficulty
            )
        else:
            # Select random task from task library
            task = self.task_library.sample_task(difficulty=self.task_difficulty)
        self._current_subtasks = list(task.subtasks)
        self.state.task = {
            'task_id': task.task_id,
            'task_type': task.task_type,
            'description': task.description,
            'difficulty': task.difficulty,
            'estimated_tokens': task.estimated_tokens,
            'num_subtasks': len(task.subtasks),
        }
        
        # Initialize workers. When a real WorkerPool is attached we mirror its
        # skill levels (and reset its per-worker bookkeeping) so manager
        # observations stay in sync with the LLMs that will actually run.
        self.state.workers = []
        for i in range(self.max_workers):
            if self.worker_pool is not None and i < len(self.worker_pool.workers):
                skill_level = float(self.worker_pool.workers[i].skill_level)
                self.worker_pool.reset_worker(i)
            else:
                skill_level = float(np.random.uniform(0.3, 1.0))
            worker = WorkerStateModel(
                worker_id=i,
                is_active=False,
                skill_level=skill_level,
            )
            self.state.workers.append(worker)
        
        # Reset budget and counters
        self.state.budget_remaining = self.token_budget
        self.state.step_counter = 0
        self.state.episode_log = []
        
        # Initialize subtask tracking
        num_subtasks = len(task.subtasks)
        self.state.subtask_status = [False] * num_subtasks
        self.state.subtask_assignments = [None] * num_subtasks
        
        # Reset hallucination tracking
        self.state.hallucination_catch_rate = 0.0
        self.state.hallucinations_detected = 0
        self.state.hallucinations_approved = 0
        self.state.false_positives = 0
        
        # Generate and return initial observation
        return self._generate_observation()
    
    def step(self, action: ManagerAction) -> Tuple[ManagerWorkerObservation, float, bool, Dict[str, Any]]:
        """
        Execute one step of environment.
        
        Args:
            action: ManagerAction with action_id and optional target_worker_id
        
        Returns:
            observation: Updated observation
            reward: float reward for this step
            done: bool indicating episode termination
            info: Dict with metadata
        """
        action_id = action.action_id
        
        # Validate action
        if not (0 <= action_id <= 6):
            raise ValueError(f"Invalid action: {action_id}. Must be in range [0, 6]")
        
        # Check if action is affordable
        action_cost = self.ACTION_COSTS[action_id]
        if self.state.budget_remaining < action_cost:
            # Action rejected due to insufficient budget
            reward = -1.0
            done = True
            info = {
                'action_valid': False,
                'reason': 'insufficient_budget',
                'budget_remaining': self.state.budget_remaining,
                'action_cost': action_cost,
            }
            return self._generate_observation(), reward, done, info
        
        # Execute action
        self._execute_action(action_id, action.target_worker_id)
        
        # Deduct tokens from budget
        self.state.budget_remaining -= action_cost
        
        # Increment step counter
        self.state.step_counter += 1
        
        # Check episode termination conditions
        done = self._check_termination()
        
        # Calculate reward
        reward = self.reward_calculator.calculate_reward(
            final_quality=self._compute_final_quality() if done else 0.0,
            steps_used=self.state.step_counter,
            max_steps=self.max_steps,
            tokens_used=self.token_budget - self.state.budget_remaining,
            token_budget=self.token_budget,
            hallucination_interventions=self.state.hallucinations_detected,
            hallucination_approvals=self.state.hallucinations_approved,
            false_positives=self.state.false_positives,
        )
        
        # Generate observation
        observation = self._generate_observation()
        
        # Create info dictionary
        info = {
            'action_valid': True,
            'action': action_id,
            'action_name': self.ACTION_NAMES[action_id],
            'action_cost': action_cost,
            'budget_remaining': self.state.budget_remaining,
            'step': self.state.step_counter,
            'hallucination_catch_rate': self.state.hallucination_catch_rate,
        }
        
        # Log step
        self.state.episode_log.append({
            'step': self.state.step_counter,
            'action': action_id,
            'reward': reward,
            'info': info,
        })
        
        return observation, reward, done, info
    
    def render(self, mode: str = 'human') -> Optional[str]:
        """
        Render environment state for visualization.
        
        Args:
            mode: 'human' for the simple multi-line view,
                  'dashboard' for a one-screen multi-worker table.
        
        Returns:
            Rendered output or None
        """
        if mode == 'dashboard':
            return self._render_dashboard()

        if mode == 'human':
            output = []
            output.append("=" * 80)
            output.append(f"Step {self.state.step_counter}/{self.max_steps}")
            output.append(f"Budget: {self.state.budget_remaining}/{self.token_budget} tokens")
            if self.state.task:
                output.append(f"Task: {self.state.task['task_type']}")
                output.append(f"Subtasks: {sum(self.state.subtask_status)}/{self.state.task['num_subtasks']} complete")
            output.append("")
            
            for worker in self.state.workers:
                status = "ACTIVE" if worker.is_active else "IDLE"
                output.append(
                    f"Worker {worker.worker_id}: {status} | "
                    f"Progress: {worker.progress:.2f} | "
                    f"Quality: {worker.output_quality_if_checked:.2f} | "
                    f"Skill: {worker.skill_level:.2f}"
                )
            
            output.append("=" * 80)
            return "\n".join(output)
        
        return None

    def _render_dashboard(self) -> str:
        """One-screen view of the whole multi-agent system."""
        rows: List[str] = []
        rows.append("┌" + "─" * 96 + "┐")
        task_label = self.state.task['task_type'] if self.state.task else '?'
        completed = sum(self.state.subtask_status)
        total = self.state.task['num_subtasks'] if self.state.task else 0
        rows.append(
            f"│ step {self.state.step_counter:>3}/{self.max_steps}   "
            f"budget {self.state.budget_remaining:>4}/{self.token_budget}   "
            f"task {task_label:<22} "
            f"subtasks {completed}/{total} complete"
            f"{' ' * 6}│"
        )
        # Subtask strip
        strip = []
        for i, done in enumerate(self.state.subtask_status):
            owner = self.state.subtask_assignments[i] if i < len(self.state.subtask_assignments) else None
            if done:
                strip.append(f"S{i}:✓")
            elif owner is not None:
                strip.append(f"S{i}:w{owner}")
            else:
                strip.append(f"S{i}:·")
        rows.append("│ subtasks: " + "  ".join(strip).ljust(85) + "│")
        rows.append("├" + "─" * 96 + "┤")
        rows.append("│ id  state    skill  subtask  failure       q(checked)  actual  output preview" + " " * 9 + "│")
        rows.append("├" + "─" * 96 + "┤")
        for w in self.state.workers:
            state = "ACTIVE " if w.is_active else "idle   "
            sub = f"S{w.current_subtask_id}" if w.current_subtask_id is not None else "  -"
            failure = (w.failure_mode or "—")[:12]
            q_checked = f"{w.output_quality_if_checked:.2f}" if w.is_checked else "  ?"
            actual = f"{w.actual_quality:.2f}" if w.has_output else "  -"
            preview = (w.output_buffer or "").replace("\n", " ")[:40]
            rows.append(
                f"│ #{w.worker_id}  {state}  {w.skill_level:>4.2f}    {sub:<5}  "
                f"{failure:<12}     {q_checked:>5}     {actual:>5}  {preview:<40}│"
            )
        rows.append("├" + "─" * 96 + "┤")
        rows.append(
            f"│ halluc detected={self.state.hallucinations_detected}  "
            f"approved={self.state.hallucinations_approved}  "
            f"false_pos={self.state.false_positives}  "
            f"catch_rate={self.state.hallucination_catch_rate:.2f}"
            f"{' ' * 22}│"
        )
        rows.append("└" + "─" * 96 + "┘")
        return "\n".join(rows)
    
    def _execute_action(self, action: int, target_worker_id: Optional[int] = None) -> None:
        """Execute manager action and update environment state."""
        if action == 0:  # assign_subtask
            self._action_assign_subtask()
        elif action == 1:  # check_worker_output
            self._action_check_worker_output()
        elif action == 2:  # correct_worker
            self._action_correct_worker()
        elif action == 3:  # reassign_task
            self._action_reassign_task()
        elif action == 4:  # fire_and_replace
            self._action_fire_and_replace()
        elif action == 5:  # approve_output
            self._action_approve_output()
        elif action == 6:  # request_clarification
            self._action_request_clarification()
    
    # ------------------------------------------------------------------
    # Worker execution helpers
    # ------------------------------------------------------------------

    def _get_subtask(self, subtask_id: int) -> Subtask:
        """Return the live Subtask object for a given index, or a synthetic one."""
        if 0 <= subtask_id < len(self._current_subtasks):
            return self._current_subtasks[subtask_id]
        # Fallback: synthesise a placeholder so the env never crashes if the
        # task library was bypassed.
        return Subtask(
            subtask_id=subtask_id,
            description=f"Subtask {subtask_id}",
            expected_output_format="text",
            quality_threshold=0.7,
        )

    def _run_worker(self, worker: WorkerStateModel, subtask: Subtask, skill_boost: float = 0.0) -> None:
        """
        Have a worker actually produce output for ``subtask``.

        The HallucinationEngine determines ground-truth quality and failure
        type. If a real WorkerPool is attached the LLM also generates the
        textual content; otherwise we keep the engine's stub content. Either
        way, the manager only learns the truth by spending tokens on a check.
        """
        effective_skill = float(np.clip(worker.skill_level + skill_boost, 0.0, 1.0))
        output: WorkerOutput = self.hallucination_engine.generate_output(
            subtask_description=subtask.description,
            worker_skill=effective_skill,
            task_difficulty=self.task_difficulty,
            failure_injection_rate=self.failure_injection_rate,
        )

        content = output.content
        if self.worker_pool is not None and worker.worker_id < len(self.worker_pool.workers):
            try:
                llm_text = self.worker_pool.assign_task(
                    worker.worker_id,
                    {
                        "description": subtask.description,
                        "context": subtask.expected_output_format,
                    },
                )
                if llm_text:
                    content = llm_text
            except Exception as exc:  # noqa: BLE001 — we want the env to keep running
                logger.warning("WorkerPool LLM call failed for worker %d: %s", worker.worker_id, exc)

        worker.output_buffer = content
        worker.actual_quality = float(output.actual_quality)
        worker.hallucination_risk_score = float(output.surface_plausibility)
        worker.failure_mode = output.failure_type
        worker.progress = 1.0
        worker.is_checked = False
        worker.has_output = True
        worker.output_quality_if_checked = 0.0  # hidden until manager checks

    # ------------------------------------------------------------------
    # Manager actions
    # ------------------------------------------------------------------

    def _action_assign_subtask(self) -> None:
        """Assign next unassigned subtask to an idle worker and run them on it."""
        for i, assigned in enumerate(self.state.subtask_assignments):
            if assigned is None and not self.state.subtask_status[i]:
                for worker in self.state.workers:
                    if not worker.is_active:
                        worker.is_active = True
                        worker.current_subtask_id = i
                        worker.progress = 0.0
                        worker.output_buffer = ""
                        worker.patience_counter = 0
                        worker.is_checked = False
                        worker.has_output = False
                        worker.failure_mode = None
                        worker.output_quality_if_checked = 0.0
                        self.state.subtask_assignments[i] = worker.worker_id
                        self._run_worker(worker, self._get_subtask(i))
                        break
                break

    def _action_check_worker_output(self) -> None:
        """Reveal the active worker's true output quality and update detection counters."""
        for worker in self.state.workers:
            if worker.is_active and worker.current_subtask_id is not None and worker.has_output:
                if not worker.is_checked:
                    if worker.failure_mode == 'hallucination':
                        # Catching a high-plausibility-but-wrong output is the
                        # behaviour the reward most strongly incentivises.
                        self.state.hallucinations_detected += 1
                    total_attempts = (
                        self.state.hallucinations_detected
                        + self.state.hallucinations_approved
                    )
                    if total_attempts > 0:
                        self.state.hallucination_catch_rate = (
                            self.state.hallucinations_detected / total_attempts
                        )
                worker.is_checked = True
                worker.output_quality_if_checked = worker.actual_quality
                break

    def _action_correct_worker(self) -> None:
        """Send corrective instructions to a worker and have them retry."""
        for worker in self.state.workers:
            if worker.is_active and worker.current_subtask_id is not None and worker.has_output:
                if worker.failure_mode is None:
                    # Correcting a clean worker is wasted budget — count it as a false positive.
                    self.state.false_positives += 1
                subtask = self._get_subtask(worker.current_subtask_id)
                # A correction nudges effective skill upward for this retry only.
                self._run_worker(worker, subtask, skill_boost=0.2)
                break

    def _action_reassign_task(self) -> None:
        """Move the current subtask to a different idle worker, who runs it fresh."""
        for worker in self.state.workers:
            if worker.is_active and worker.current_subtask_id is not None:
                if worker.failure_mode is None and worker.has_output:
                    self.state.false_positives += 1
                subtask_id = worker.current_subtask_id
                worker.is_active = False
                worker.current_subtask_id = None
                worker.progress = 0.0
                worker.output_buffer = ""
                worker.has_output = False
                worker.is_checked = False
                worker.failure_mode = None
                for other_worker in self.state.workers:
                    if not other_worker.is_active:
                        other_worker.is_active = True
                        other_worker.current_subtask_id = subtask_id
                        other_worker.progress = 0.0
                        other_worker.output_buffer = ""
                        other_worker.is_checked = False
                        other_worker.has_output = False
                        other_worker.failure_mode = None
                        self.state.subtask_assignments[subtask_id] = other_worker.worker_id
                        self._run_worker(other_worker, self._get_subtask(subtask_id))
                        break
                break

    def _action_fire_and_replace(self) -> None:
        """Replace a worker with a new one and re-run the in-flight subtask if any."""
        for i, worker in enumerate(self.state.workers):
            if worker.is_active:
                if worker.failure_mode is None and worker.has_output:
                    self.state.false_positives += 1
                subtask_id = worker.current_subtask_id
                new_skill = float(np.random.uniform(0.3, 1.0))
                replacement = WorkerStateModel(
                    worker_id=i,
                    is_active=subtask_id is not None,
                    current_subtask_id=subtask_id,
                    skill_level=new_skill,
                )
                self.state.workers[i] = replacement
                if subtask_id is not None:
                    self._run_worker(replacement, self._get_subtask(subtask_id))
                break

    def _action_approve_output(self) -> None:
        """Approve active worker's output and mark subtask complete."""
        for worker in self.state.workers:
            if worker.is_active and worker.current_subtask_id is not None:
                if worker.has_output and not worker.is_checked and worker.failure_mode == 'hallucination':
                    # Approving a hallucination without checking is the worst-case
                    # mistake — the reward calculator penalises this heavily.
                    self.state.hallucinations_approved += 1
                    total_attempts = (
                        self.state.hallucinations_detected
                        + self.state.hallucinations_approved
                    )
                    if total_attempts > 0:
                        self.state.hallucination_catch_rate = (
                            self.state.hallucinations_detected / total_attempts
                        )
                subtask_id = worker.current_subtask_id
                self.state.subtask_status[subtask_id] = True
                worker.is_active = False
                worker.current_subtask_id = None
                break

    def _action_request_clarification(self) -> None:
        """Request clarification — currently a soft no-op that just spends budget."""
        pass
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate."""
        if self.state.step_counter >= self.max_steps:
            return True
        
        if self.state.budget_remaining <= 0:
            return True
        
        if all(self.state.subtask_status):
            return True
        
        for worker in self.state.workers:
            if worker.patience_counter > 10:
                return True
        
        return False
    
    def _compute_final_quality(self) -> float:
        """
        Compute final quality across completed subtasks.

        Uses the worker's *actual* quality (from the hallucination engine),
        not ``output_quality_if_checked`` — otherwise approving without
        checking would always score 0.0 and the manager couldn't learn the
        approve-vs-check trade-off properly.
        """
        if not self.state.task or self.state.task['num_subtasks'] == 0:
            return 0.0

        total_quality = 0.0
        completed_count = 0

        for i, is_complete in enumerate(self.state.subtask_status):
            if is_complete:
                worker_id = self.state.subtask_assignments[i]
                if worker_id is not None and worker_id < len(self.state.workers):
                    worker = self.state.workers[worker_id]
                    total_quality += worker.actual_quality
                    completed_count += 1

        if completed_count == 0:
            return 0.0

        return total_quality / completed_count
    
    def _generate_observation(self) -> ManagerWorkerObservation:
        """Generate observation from current environment state."""
        # Generate task embedding
        if self.state.task:
            task_embedding = self._get_task_embedding(self.state.task['task_type'])
        else:
            task_embedding = [0.0] * 64
        
        # Generate worker states (4x5 array)
        worker_states = []
        for worker in self.state.workers:
            tokens_consumed_ratio = (self.token_budget - self.state.budget_remaining) / self.token_budget if self.token_budget > 0 else 0.0
            worker_state = [
                float(worker.is_active),
                worker.progress,
                worker.hallucination_risk_score,
                worker.output_quality_if_checked,
                tokens_consumed_ratio,
            ]
            worker_states.append(worker_state)
        
        # Pad to 4 workers if needed
        while len(worker_states) < 4:
            worker_states.append([0.0] * 5)
        
        # Generate subtask status (padded / truncated to MAX_OBSERVABLE_SUBTASKS)
        cap = self.MAX_OBSERVABLE_SUBTASKS
        subtask_status = [int(s) for s in self.state.subtask_status[:cap]]
        while len(subtask_status) < cap:
            subtask_status.append(0)
        
        # Generate budget and steps remaining
        budget_remaining = self.state.budget_remaining / self.token_budget if self.token_budget > 0 else 0.0
        steps_remaining = (self.max_steps - self.state.step_counter) / self.max_steps if self.max_steps > 0 else 0.0
        
        num_subtasks_real = len(self.state.subtask_status)
        num_subtasks_norm = (
            num_subtasks_real / self.MAX_OBSERVABLE_SUBTASKS
            if self.MAX_OBSERVABLE_SUBTASKS > 0
            else 0.0
        )

        n_workers_real = len(self.state.workers)
        # Normalized over the obs cap of 4 worker rows.
        num_workers_norm = max(0.0, min(1.0, n_workers_real / 4.0))

        return ManagerWorkerObservation(
            task_embedding=task_embedding,
            worker_states=worker_states,
            num_workers=num_workers_norm,
            subtask_status=subtask_status,
            num_subtasks=num_subtasks_norm,
            budget_remaining=budget_remaining,
            steps_remaining=steps_remaining,
            episode_log=self.state.episode_log,
            hallucination_catch_rate=self.state.hallucination_catch_rate,
        )
    
    def _get_task_embedding(self, task_type: str) -> List[float]:
        """Get fixed embedding for task type."""
        embeddings = {
            'web_development': np.random.RandomState(hash(task_type) % 2**32).uniform(-1, 1, 64),
            'research': np.random.RandomState(hash(task_type) % 2**32).uniform(-1, 1, 64),
            'software_engineering': np.random.RandomState(hash(task_type) % 2**32).uniform(-1, 1, 64),
            'product_management': np.random.RandomState(hash(task_type) % 2**32).uniform(-1, 1, 64),
            'academic_writing': np.random.RandomState(hash(task_type) % 2**32).uniform(-1, 1, 64),
        }
        embedding = embeddings.get(task_type, np.zeros(64))
        return embedding.astype(float).tolist()
