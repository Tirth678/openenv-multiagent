import numpy as np
import os
from stable_baselines3 import PPO
from typing import Tuple, Dict, Any, Optional

class PPOManager:
    """Manager Agent powered by a trained PPO policy."""
    
    def __init__(self, env, model_path: str = None):
        self.env = env
        self.last_thought = "Initializing PPO Manager..."
        
        if model_path and os.path.exists(model_path + ".zip"):
            self.model = PPO.load(model_path, env=env)
        else:
            # Create a fresh model if no path or doesn't exist
            self.model = PPO("MultiInputPolicy", env, verbose=1)

    def predict(self, obs: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, Any]]:
        action, _states = self.model.predict(obs, deterministic=True)
        self.last_thought = self.get_thought(obs, action)
        return action, {"thought": self.last_thought}

    def get_thought(self, obs: Dict[str, np.ndarray], action: np.ndarray) -> str:
        atype, wid, sidx = action
        action_name = ["ASSIGN", "CHECK", "REASSIGN", "APPROVE", "REJECT", "WAIT"][atype]
        return f"PPO Manager chose to {action_name} for worker {wid} on subtask index {sidx} based on observed states."

class ParallelManager:
    """Deterministic Manager that fans out work as quickly as possible."""
    
    def __init__(self, env):
        self.env = env
        self.last_thought = "Initializing Parallel Manager..."

    def predict(self, obs: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, Any]]:
        # Find first idle worker
        idle_workers = np.where(obs['worker_states'][:, 1] == 0)[0]
        # Find first pending subtask
        pending_subtasks = np.where(obs['task_state'][:, 1] == 0)[0]
        
        if len(idle_workers) > 0 and len(pending_subtasks) > 0:
            action = np.array([0, idle_workers[0], pending_subtasks[0]]) # ASSIGN
        else:
            # If all busy, wait or approve if something is done
            in_progress = np.where(obs['task_state'][:, 1] == 1)[0]
            if len(in_progress) > 0:
                # Check which worker has it
                for i, w in enumerate(obs['worker_states']):
                    if w[1] == 1:
                        action = np.array([3, i, in_progress[0]]) # APPROVE blindly
                        break
                else:
                    action = np.array([5, 0, 0]) # WAIT
            else:
                action = np.array([5, 0, 0]) # WAIT
                
        self.last_thought = f"Parallel strategy: Fanning out work. Current Action: {action}"
        return action, {"thought": self.last_thought}

class HeuristicManager:
    """Safety-first Manager that checks every output before approval."""
    
    def __init__(self, env):
        self.env = env
        self.last_thought = "Initializing Heuristic Manager..."

    def predict(self, obs: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, Any]]:
        # Find first idle worker
        idle_workers = np.where(obs['worker_states'][:, 1] == 0)[0]
        pending_subtasks = np.where(obs['task_state'][:, 0] == 0)[0] # Not complete
        
        # 1. Assign if idle
        if len(idle_workers) > 0 and len(pending_subtasks) > 0:
            action = np.array([0, idle_workers[0], pending_subtasks[0]])
        else:
            # 2. Check if in progress but not revealed
            in_progress = np.where(obs['task_state'][:, 1] == 1)[0]
            if len(in_progress) > 0:
                sidx = in_progress[0]
                quality = obs['task_state'][sidx, 2]
                
                # Find worker assigned to this sidx
                # (Simple heuristic: find first busy worker)
                busy_workers = np.where(obs['worker_states'][:, 1] == 1)[0]
                if len(busy_workers) > 0:
                    wid = busy_workers[0]
                    if quality == 0: # Not checked yet
                        action = np.array([1, wid, sidx]) # CHECK
                    elif quality >= 0.7:
                        action = np.array([3, wid, sidx]) # APPROVE
                    else:
                        action = np.array([4, wid, sidx]) # REJECT
                else:
                    action = np.array([5, 0, 0])
            else:
                action = np.array([5, 0, 0])
                
        self.last_thought = f"Heuristic strategy: Verify before approve. Action: {action}"
        return action, {"thought": self.last_thought}
