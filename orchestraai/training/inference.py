import numpy as np
import time
from typing import Dict, List, Generator, Any

from env.manager_worker_env import ManagerWorkerEnv
from agents.manager_agent import PPOManager, ParallelManager, HeuristicManager

class InferencePipeline:
    """Pipeline for running and streaming agent episodes."""
    
    def __init__(self, manager_model_path: str = None, strategy: str = "PPO", use_real_llms: bool = False):
        self.env = ManagerWorkerEnv("config.yaml")
        self.strategy = strategy
        
        if strategy == "PPO":
            self.manager = PPOManager(self.env, model_path=manager_model_path)
        elif strategy == "Parallel":
            self.manager = ParallelManager(self.env)
        elif strategy == "Heuristic":
            self.manager = HeuristicManager(self.env)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def run_episode(self, task_id: str = None) -> Dict[str, Any]:
        """Runs a full episode and returns the log."""
        obs, info = self.env.reset()
        if task_id:
            # Manually set task if id provided
            self.env.current_task = self.env.task_library.get_task_by_id(task_id)
            obs, info = self.env.reset() # Re-reset to apply task
            
        done = False
        log = []
        total_reward = 0
        
        while not done:
            action, agent_info = self.manager.predict(obs)
            obs, reward, terminated, truncated, env_info = self.env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            log.append({
                "step": self.env.current_step,
                "action": self.env.ACTION_NAMES[action[0]],
                "reward": reward,
                "thought": agent_info.get("thought", ""),
                "worker_states": self.env.workers.copy(),
                "budget_remaining": self.env.budget_remaining
            })
            
        return {
            "total_reward": total_reward,
            "steps": len(log),
            "history": log
        }

    def stream_episode(self, task_id: str = None) -> Generator[Dict[str, Any], None, None]:
        """Streams episode updates step-by-step."""
        obs, info = self.env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action, agent_info = self.manager.predict(obs)
            obs, reward, terminated, truncated, env_info = self.env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            yield {
                "step": self.env.current_step,
                "action": self.env.ACTION_NAMES[action[0]],
                "reward": reward,
                "total_reward": total_reward,
                "thought": agent_info.get("thought", ""),
                "worker_states": self.env.workers,
                "budget_remaining": self.env.budget_remaining,
                "done": done
            }

    def compare_strategies(self, task_id: str, n_runs: int = 5) -> Dict[str, Dict]:
        """Compares PPO, Parallel, and Heuristic strategies."""
        results = {}
        strategies = ["PPO", "Parallel", "Heuristic"]
        
        for strat in strategies:
            strat_rewards = []
            strat_steps = []
            
            # Create a temporary pipeline for this strategy
            pipe = InferencePipeline(strategy=strat)
            
            for _ in range(n_runs):
                res = pipe.run_episode(task_id)
                strat_rewards.append(res['total_reward'])
                strat_steps.append(res['steps'])
                
            results[strat] = {
                "avg_reward": np.mean(strat_rewards),
                "avg_steps": np.mean(strat_steps),
                "success_rate": np.mean([1.0 if r > 50 else 0.0 for r in strat_rewards])
            }
            
        return results
