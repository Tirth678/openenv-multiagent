"""
Metrics repository for database operations.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorDatabase


class MetricsRepository:
    """Repository for metrics data access."""
    
    def __init__(self, db: AsyncIOMotorDatabase):
        """Initialize repository."""
        self.db = db
        self.collection = db["metrics"]
    
    async def create(self, metrics_data: Dict[str, Any]) -> str:
        """Create a new metrics record."""
        result = await self.collection.insert_one(metrics_data)
        return str(result.inserted_id)
    
    async def get_latest(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get latest metrics records."""
        cursor = self.collection.find().sort("timestamp", -1).limit(limit)
        return await cursor.to_list(length=limit)
    
    async def get_by_episode(self, episode_id: str) -> List[Dict[str, Any]]:
        """Get metrics for a specific episode."""
        cursor = self.collection.find({"episode_id": episode_id}).sort("timestamp", -1)
        return await cursor.to_list(length=None)
    
    async def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        pipeline = [
            {
                "$group": {
                    "_id": None,
                    "total_episodes": {"$sum": 1},
                    "avg_reward": {"$avg": "$mean_reward"},
                    "avg_episode_length": {"$avg": "$average_episode_length"},
                    "total_timesteps": {"$max": "$total_timesteps"},
                }
            }
        ]
        result = await self.collection.aggregate(pipeline).to_list(length=1)
        return result[0] if result else {}
