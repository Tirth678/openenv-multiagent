"""
Episode business logic service — backed by ManagerWorkerEnv for steps.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from motor.motor_asyncio import AsyncIOMotorDatabase

from db.repositories.episode_repo import EpisodeRepository
from env import ManagerAction, ManagerWorkerEnv


def _serialize_observation(obs: Any) -> Dict[str, Any]:
    if hasattr(obs, "model_dump"):
        return obs.model_dump()
    if isinstance(obs, dict):
        return obs
    return {"raw": str(obs)}


class EpisodeService:
    """Service for episode management."""

    def __init__(self, db: Optional[AsyncIOMotorDatabase] = None):
        self.db = db
        self.repo = EpisodeRepository(db) if db is not None else None
        self.active_episodes: Dict[str, Any] = {}
        # Live RL env instances (not serializable to Mongo)
        self._envs: Dict[str, ManagerWorkerEnv] = {}

    async def start_episode(
        self,
        task_id: str,
        difficulty: int,
        num_workers: int,
        budget: float,
    ) -> Dict[str, Any]:
        """Start a new episode using ManagerWorkerEnv."""
        episode_id = f"ep_{uuid.uuid4().hex[:12]}"
        token_budget = int(max(500, min(5000, float(budget))))
        max_workers = int(max(1, min(4, num_workers)))
        task_difficulty = int(max(1, min(5, difficulty)))

        env_config = {
            "max_workers": max_workers,
            "max_steps": 50,
            "token_budget": token_budget,
            "task_difficulty": task_difficulty,
            "failure_injection_rate": 0.6,
        }
        env = ManagerWorkerEnv(env_config)
        obs = env.reset(task_id=task_id)
        self._envs[episode_id] = env

        obs_dict = _serialize_observation(obs)
        episode_data = {
            "episode_id": episode_id,
            "task_id": task_id,
            "difficulty": difficulty,
            "num_workers": num_workers,
            "budget": budget,
            "current_observation": obs_dict,
            "total_reward": 0.0,
            "step_count": 0,
            "is_active": True,
            "created_at": datetime.utcnow(),
            "ended_at": None,
            "episode_log": [],
            "final_quality": None,
        }

        self.active_episodes[episode_id] = episode_data
        if self.repo is not None:
            await self.repo.create(episode_data)

        return episode_data

    async def get_episode(self, episode_id: str) -> Optional[Dict[str, Any]]:
        """Get episode by ID."""
        if episode_id in self.active_episodes:
            return self.active_episodes[episode_id]
        if self.repo is None:
            return None
        return await self.repo.get_by_id(episode_id)

    async def step(
        self,
        episode_id: str,
        action_id: int,
        target_worker_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute one Manager action in the live environment."""
        env = self._envs.get(episode_id)
        if env is None:
            raise ValueError(
                f"No live environment for episode {episode_id}; start a new episode in this process."
            )

        episode = await self.get_episode(episode_id)
        if not episode:
            raise ValueError(f"Episode not found: {episode_id}")
        if not episode["is_active"]:
            raise ValueError(f"Episode not active: {episode_id}")

        action = ManagerAction(action_id=action_id, target_worker_id=target_worker_id)
        obs, reward, done, info = env.step(action)

        obs_dict = _serialize_observation(obs)
        episode["step_count"] += 1
        episode["total_reward"] = float(episode["total_reward"]) + float(reward)
        episode["current_observation"] = obs_dict
        episode["episode_log"].append(
            {
                "step": episode["step_count"],
                "action": action_id,
                "reward": reward,
                "observation": obs_dict,
            }
        )

        if self.repo is not None:
            await self.repo.update(
                episode_id,
                {
                    "step_count": episode["step_count"],
                    "total_reward": episode["total_reward"],
                    "current_observation": obs_dict,
                    "episode_log": episode["episode_log"],
                },
            )

        return {
            "observation": obs_dict,
            "reward": float(reward),
            "done": bool(done),
            "info": info if isinstance(info, dict) else {"info": info},
        }

    async def end_episode(self, episode_id: str) -> Dict[str, Any]:
        """End an episode."""
        episode = await self.get_episode(episode_id)
        if not episode:
            raise ValueError(f"Episode not found: {episode_id}")

        episode["is_active"] = False
        episode["ended_at"] = datetime.utcnow()
        episode["final_quality"] = episode["total_reward"] / max(episode["step_count"], 1)

        if self.repo is not None:
            await self.repo.update(
                episode_id,
                {
                    "is_active": False,
                    "ended_at": episode["ended_at"],
                    "final_quality": episode["final_quality"],
                },
            )

        if episode_id in self.active_episodes:
            del self.active_episodes[episode_id]
        if episode_id in self._envs:
            del self._envs[episode_id]

        return {
            "final_reward": episode["total_reward"],
            "total_steps": episode["step_count"],
        }

    async def list_episodes(self, limit: int = 10, skip: int = 0) -> Dict[str, Any]:
        """List episodes."""
        if self.repo is not None:
            episodes = await self.repo.list_all(limit=limit, skip=skip)
            total = await self.repo.count_all()
        else:
            all_ids = list(self.active_episodes.keys())
            page = all_ids[skip : skip + limit]
            episodes = [self.active_episodes[i] for i in page if i in self.active_episodes]
            total = len(self.active_episodes)

        return {
            "episodes": episodes,
            "total_count": total,
        }

    async def get_episode_history(self, episode_id: str) -> Dict[str, Any]:
        """Get episode history."""
        episode = await self.get_episode(episode_id)
        if not episode:
            raise ValueError(f"Episode not found: {episode_id}")
        return episode.get("episode_log", [])
