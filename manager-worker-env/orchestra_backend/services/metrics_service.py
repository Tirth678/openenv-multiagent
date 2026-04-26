"""
Metrics tracking and management service.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorDatabase
from db.repositories.metrics_repo import MetricsRepository
import numpy as np


class MetricsService:
    """Service for metrics management."""
    
    def __init__(self, db: Optional[AsyncIOMotorDatabase] = None):
        """Initialize service."""
        self.db = db
        self.repo = MetricsRepository(db) if db is not None else None
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.total_timesteps = 0
        self.episode_count = 0
        self.learning_rate = 3e-4
        self.hallucination_detection_rate = 0.0
    
    async def record_episode(
        self,
        episode_id: str,
        total_reward: float,
        episode_length: int,
        final_quality: float,
    ) -> None:
        """Record metrics for a completed episode."""
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(episode_length)
        self.total_timesteps += episode_length
        self.episode_count += 1
        
        metrics_doc = {
            "timestamp": datetime.utcnow(),
            "episode_id": episode_id,
            "total_timesteps": self.total_timesteps,
            "mean_reward": float(np.mean(self.episode_rewards[-100:])) if self.episode_rewards else 0.0,
            "episode_count": self.episode_count,
            "learning_rate": self.learning_rate,
            "hallucination_detection_rate": self.hallucination_detection_rate,
            "average_episode_length": float(np.mean(self.episode_lengths[-100:])) if self.episode_lengths else 0.0,
        }
        
        if self.repo is not None:
            await self.repo.create(metrics_doc)
    
    async def get_current_metrics(self) -> Dict[str, Any]:
        """Get current training metrics."""
        mean_reward = float(np.mean(self.episode_rewards[-100:])) if self.episode_rewards else 0.0
        avg_length = float(np.mean(self.episode_lengths[-100:])) if self.episode_lengths else 0.0
        
        return {
            "episode_count": self.episode_count,
            "total_steps": self.total_timesteps,
            "average_reward": mean_reward,
            "average_episode_length": int(avg_length),
        }
    
    async def get_metrics_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get historical metrics."""
        if self.repo is None:
            return []
        metrics = await self.repo.get_latest(limit=limit)
        
        for metric in metrics:
            metric["_id"] = str(metric.get("_id", ""))
            if isinstance(metric.get("timestamp"), datetime):
                metric["timestamp"] = metric["timestamp"].isoformat()
        
        return list(reversed(metrics))
    
    def update_learning_rate(self, learning_rate: float) -> None:
        """Update learning rate."""
        self.learning_rate = learning_rate
    
    def update_hallucination_detection_rate(self, rate: float) -> None:
        """Update hallucination detection rate."""
        self.hallucination_detection_rate = rate
