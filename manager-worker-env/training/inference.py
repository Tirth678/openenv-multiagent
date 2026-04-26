#!/usr/bin/env python3
"""
Inference script to run a trained PPO model on the ManagerWorkerEnv.

This script loads a trained model and runs it on test episodes,
displaying performance metrics and visualizations.

Usage:
    python training/inference.py --model models/ppo_manager --episodes 5
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from typing import Optional, List, Dict, Any
import numpy as np

from stable_baselines3 import PPO
from env import ManagerWorkerEnv, ManagerAction


class InferenceRunner:
    """Run inference with a trained model."""
    
    def __init__(self, model_path: str):
        """Initialize inference runner."""
        self.model = PPO.load(model_path)
        print(f"✓ Model loaded from {model_path}")
    
    def run_episode(
        self,
        env_config: Optional[Dict[str, Any]] = None,
        deterministic: bool = True,
        render: bool = False,
    ) -> Dict[str, Any]:
        """
        Run a single episode with the trained model.
        
        Args:
            env_config: Environment configuration
            deterministic: Use deterministic policy
            render: Print environment state
        
        Returns:
            Episode statistics
        """
        env = ManagerWorkerEnv(env_config)
        obs = env.reset()
        
        # Convert observation to dict format
        obs_dict = self._convert_observation(obs)
        
        episode_reward = 0.0
        episode_length = 0
        actions_taken = []
        rewards = []
        
        while True:
            # Get action from model
            action, _ = self.model.predict(obs_dict, deterministic=deterministic)
            
            # Execute action
            manager_action = ManagerAction(action_id=int(action))
            obs, reward, done, info = env.step(manager_action)
            
            # Convert observation
            obs_dict = self._convert_observation(obs)
            
            # Track metrics
            episode_reward += reward
            episode_length += 1
            actions_taken.append(int(action))
            rewards.append(reward)
            
            if render:
                action_name = env.ACTION_NAMES[int(action)]
                print(f"  Step {episode_length}: {action_name:25s} | Reward: {reward:6.2f}")
            
            if done:
                break
        
        return {
            'total_reward': episode_reward,
            'episode_length': episode_length,
            'average_reward': episode_reward / episode_length if episode_length > 0 else 0,
            'actions': actions_taken,
            'rewards': rewards,
        }
    
    def _convert_observation(self, obs):
        """Convert ManagerWorkerObservation to dict format."""
        return {
            'task_embedding': np.array(obs.task_embedding, dtype=np.float32),
            'worker_states': np.array(obs.worker_states, dtype=np.float32),
            'num_workers': np.array([obs.num_workers], dtype=np.float32),
            'subtask_status': np.array(obs.subtask_status, dtype=np.int8),
            'num_subtasks': np.array([obs.num_subtasks], dtype=np.float32),
            'budget_remaining': np.array([obs.budget_remaining], dtype=np.float32),
            'steps_remaining': np.array([obs.steps_remaining], dtype=np.float32),
        }
    
    def run_episodes(
        self,
        num_episodes: int = 5,
        env_config: Optional[Dict[str, Any]] = None,
        deterministic: bool = True,
        render: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Run multiple episodes.
        
        Args:
            num_episodes: Number of episodes to run
            env_config: Environment configuration
            deterministic: Use deterministic policy
            render: Print environment state
        
        Returns:
            List of episode statistics
        """
        episodes = []
        
        for episode_num in range(num_episodes):
            print(f"\nEpisode {episode_num + 1}/{num_episodes}")
            print("-" * 60)
            
            stats = self.run_episode(
                env_config=env_config,
                deterministic=deterministic,
                render=render,
            )
            
            episodes.append(stats)
            
            print(f"  Total Reward: {stats['total_reward']:.2f}")
            print(f"  Episode Length: {stats['episode_length']}")
            print(f"  Average Reward/Step: {stats['average_reward']:.2f}")
        
        return episodes
    
    def print_summary(self, episodes: List[Dict[str, Any]]) -> None:
        """Print summary statistics."""
        if not episodes:
            return
        
        total_rewards = [e['total_reward'] for e in episodes]
        episode_lengths = [e['episode_length'] for e in episodes]
        avg_rewards = [e['average_reward'] for e in episodes]
        
        print("\n" + "=" * 60)
        print("INFERENCE SUMMARY")
        print("=" * 60)
        print(f"\nEpisodes Run: {len(episodes)}")
        print(f"\nReward Statistics:")
        print(f"  Mean: {np.mean(total_rewards):.2f}")
        print(f"  Std: {np.std(total_rewards):.2f}")
        print(f"  Min: {np.min(total_rewards):.2f}")
        print(f"  Max: {np.max(total_rewards):.2f}")
        print(f"\nEpisode Length Statistics:")
        print(f"  Mean: {np.mean(episode_lengths):.1f}")
        print(f"  Std: {np.std(episode_lengths):.1f}")
        print(f"  Min: {np.min(episode_lengths)}")
        print(f"  Max: {np.max(episode_lengths)}")
        print(f"\nAverage Reward per Step:")
        print(f"  Mean: {np.mean(avg_rewards):.2f}")
        print(f"  Std: {np.std(avg_rewards):.2f}")
        print("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run inference with a trained PPO model'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='models/ppo_manager',
        help='Path to trained model (default: models/ppo_manager)',
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=5,
        help='Number of episodes to run (default: 5)',
    )
    parser.add_argument(
        '--render',
        action='store_true',
        help='Print environment state during episodes',
    )
    parser.add_argument(
        '--stochastic',
        action='store_true',
        help='Use stochastic policy (default: deterministic)',
    )
    
    args = parser.parse_args()
    
    # Create inference runner
    runner = InferenceRunner(args.model)
    
    # Run episodes
    episodes = runner.run_episodes(
        num_episodes=args.episodes,
        deterministic=not args.stochastic,
        render=args.render,
    )
    
    # Print summary
    runner.print_summary(episodes)


if __name__ == '__main__':
    main()
