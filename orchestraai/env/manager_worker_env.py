import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import yaml
from typing import Dict, Tuple, List, Optional
import logging

from .task_library import TaskLibrary
from .quality_eval import QualityEvaluator

class ManagerWorkerEnv(gym.Env):
    """
    Gymnasium-compatible environment for multi-agent orchestration.
    The Manager must coordinate workers to complete subtasks within a budget.
    """
    
    ACTION_NAMES = ["ASSIGN", "CHECK", "REASSIGN", "APPROVE", "REJECT", "WAIT"]
    
    def __init__(self, config_path: str = "config.yaml"):
        super(ManagerWorkerEnv, self).__init__()
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.num_workers = self.config['environment']['num_workers']
        self.token_budget_total = self.config['environment']['token_budget']
        self.max_steps = self.config['environment']['max_steps']
        self.max_subtasks = 10 # Buffer size for obs space
        
        self.task_library = TaskLibrary()
        self.quality_evaluator = QualityEvaluator()
        
        # Spaces
        self.observation_space = spaces.Dict({
            "worker_states": spaces.Box(low=0, high=1, shape=(self.num_workers, 5), dtype=np.float32),
            "task_state": spaces.Box(low=0, high=1, shape=(self.max_subtasks, 3), dtype=np.float32),
            "budget_info": spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32),
            "last_action_results": spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32),
        })
        
        self.action_space = spaces.MultiDiscrete([6, self.num_workers, self.max_subtasks])
        
        # Internal state
        self.worker_configs = self.config['workers']['types']
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        # Sample new task
        self.current_task = self.task_library.get_random_task()
        self.subtasks = self.current_task['subtasks']
        self.num_subtasks = len(self.subtasks)
        
        # Initialize workers
        self.workers = []
        for i in range(self.num_workers):
            conf = self.worker_configs[i % len(self.worker_configs)]
            self.workers.append({
                "id": i,
                "name": conf['name'],
                "skill_level": conf['skill_level'],
                "cost_per_token": conf['cost_per_token'],
                "is_busy": False,
                "consecutive_failures": 0,
                "current_subtask_id": None,
                "output_buffer": "",
                "quality_revealed": -1.0, # -1 means unknown
                "failure_mode": None
            })
            
        self.subtask_status = {s['subtask_id']: "PENDING" for s in self.subtasks}
        self.subtask_quality = {s['subtask_id']: 0.0 for s in self.subtasks}
        self.subtask_assignments = {s['subtask_id']: None for s in self.subtasks}
        
        self.budget_remaining = self.token_budget_total
        self.current_step = 0
        self.last_action_results = np.zeros(4, dtype=np.float32)
        
        return self._generate_observation(), {}

    def step(self, action):
        action_type, worker_id, subtask_idx = action
        worker_id = worker_id % self.num_workers
        subtask_idx = subtask_idx % self.num_subtasks
        
        subtask = self.subtasks[subtask_idx]
        sid = subtask['subtask_id']
        worker = self.workers[worker_id]
        
        reward = 0.0
        terminated = False
        truncated = False
        info = {"action_taken": self.ACTION_NAMES[action_type], "worker_id": worker_id, "subtask_id": sid}
        
        # Costs and Logic
        cost = 0
        
        if action_type == 0: # ASSIGN
            if worker['is_busy'] or self.subtask_status[sid] != "PENDING":
                reward += self.config['rewards']['failure_penalty']
            else:
                worker['is_busy'] = True
                worker['current_subtask_id'] = sid
                self.subtask_status[sid] = "IN_PROGRESS"
                self.subtask_assignments[sid] = worker_id
                cost = worker['cost_per_token'] * subtask['complexity'] * 50
                
                # Simulate generation
                res = self.quality_evaluator.evaluate("Simulated output", subtask, worker['skill_level'])
                worker['output_buffer'] = f"Result of {sid} by {worker['name']}"
                worker['quality_revealed'] = -1.0 # Hidden initially
                worker['failure_mode'] = res['failure_mode']
                self.subtask_quality[sid] = res['quality_score']
                
        elif action_type == 1: # CHECK
            if not worker['is_busy'] or worker['current_subtask_id'] is None:
                reward += self.config['rewards']['failure_penalty']
            else:
                cost = worker['cost_per_token'] * 20
                worker['quality_revealed'] = self.subtask_quality[worker['current_subtask_id']]
                # Penalty for unnecessary check
                if worker['skill_level'] > 0.9 and subtask['complexity'] < 0.4:
                    reward += self.config['rewards']['unnecessary_check_penalty']
                    
        elif action_type == 2: # REASSIGN
            cost = worker['cost_per_token'] * 30 + 10
            # Logic similar to ASSIGN but clears previous
            if worker['current_subtask_id']:
                old_sid = worker['current_subtask_id']
                self.subtask_status[old_sid] = "PENDING"
            worker['is_busy'] = True
            worker['current_subtask_id'] = sid
            self.subtask_status[sid] = "IN_PROGRESS"
            
        elif action_type == 3: # APPROVE
            if not worker['is_busy'] or self.subtask_status[sid] != "IN_PROGRESS":
                reward += self.config['rewards']['failure_penalty']
            else:
                cost = 2
                self.subtask_status[sid] = "COMPLETED"
                worker['is_busy'] = False
                worker['current_subtask_id'] = None
                reward += self.subtask_quality[sid] * self.config['rewards']['quality_weight']
                
        elif action_type == 4: # REJECT
            cost = 5
            if worker['is_busy']:
                self.subtask_status[sid] = "PENDING"
                worker['is_busy'] = False
                worker['current_subtask_id'] = None
                worker['consecutive_failures'] += 1
                reward += self.config['rewards']['failure_penalty']
                
        elif action_type == 5: # WAIT
            cost = 1
            
        # Deduct budget
        self.budget_remaining -= cost
        self.current_step += 1
        
        # Check termination
        if self.budget_remaining <= 0 or self.current_step >= self.max_steps:
            terminated = True
            
        # Completion bonus
        if all(self.subtask_status[s['subtask_id']] == "COMPLETED" for s in self.subtasks):
            reward += self.config['rewards']['completion_bonus']
            terminated = True
            
        self.last_action_results = np.array([
            self.subtask_quality[sid] if action_type in [1, 3] else 0.0,
            cost / 100.0,
            1.0 if cost > 0 or action_type == 5 else 0.0,
            self._encode_failure_mode(worker['failure_mode'])
        ], dtype=np.float32)
        
        obs = self._generate_observation()
        info["manager_thought"] = self._generate_manager_thought(action, reward)
        info["budget_remaining"] = self.budget_remaining
        
        return obs, reward, terminated, truncated, info

    def _generate_observation(self):
        worker_states = np.zeros((self.num_workers, 5), dtype=np.float32)
        for i, w in enumerate(self.workers):
            worker_states[i] = [
                w['skill_level'],
                1.0 if w['is_busy'] else 0.0,
                w['consecutive_failures'] / 5.0,
                0.0 if not w['is_busy'] else 0.5, # Placeholder for complexity
                w['cost_per_token'] / 10.0
            ]
            
        task_state = np.zeros((self.max_subtasks, 3), dtype=np.float32)
        for i, s in enumerate(self.subtasks):
            sid = s['subtask_id']
            task_state[i] = [
                1.0 if self.subtask_status[sid] == "COMPLETED" else 0.0,
                1.0 if self.subtask_status[sid] == "IN_PROGRESS" else 0.0,
                self.workers[self.subtask_assignments[sid]]['quality_revealed'] if self.subtask_assignments[sid] is not None else 0.0
            ]
            
        budget_info = np.array([
            self.budget_remaining / self.token_budget_total,
            (self.max_steps - self.current_step) / self.max_steps
        ], dtype=np.float32)
        
        return {
            "worker_states": worker_states,
            "task_state": task_state,
            "budget_info": budget_info,
            "last_action_results": self.last_action_results
        }

    def _encode_failure_mode(self, mode):
        mapping = {None: 0, "hallucination": 1, "off_task": 2, "incomplete": 3}
        return mapping.get(mode, 0) / 3.0

    def _generate_manager_thought(self, action, reward):
        atype, wid, sidx = action
        w = self.workers[wid % self.num_workers]
        s = self.subtasks[sidx % self.num_subtasks]
        return f"Manager decided to {self.ACTION_NAMES[atype]} worker {wid} for subtask {s['name']}. Reward: {reward:.2f}."

    def render(self):
        print(f"\n--- Step {self.current_step} | Budget: {self.budget_remaining} ---")
        for w in self.workers:
            status = "BUSY" if w['is_busy'] else "IDLE"
            print(f"W{w['id']} ({w['name']}): {status} | Failures: {w['consecutive_failures']}")
        for sid, status in self.subtask_status.items():
            print(f"Subtask {sid}: {status}")
