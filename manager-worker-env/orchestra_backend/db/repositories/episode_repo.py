"""
Episode repository for database operations.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorDatabase
from bson import ObjectId


class EpisodeRepository:
    """Repository for episode data access."""
    
    def __init__(self, db: AsyncIOMotorDatabase):
        """Initialize repository."""
        self.db = db
        self.collection = db["episodes"]
    
    async def create(self, episode_data: Dict[str, Any]) -> str:
        """Create a new episode."""
        result = await self.collection.insert_one(episode_data)
        return str(result.inserted_id)
    
    async def get_by_id(self, episode_id: str) -> Optional[Dict[str, Any]]:
        """Get episode by ID."""
        return await self.collection.find_one({"episode_id": episode_id})
    
    async def update(self, episode_id: str, update_data: Dict[str, Any]) -> bool:
        """Update episode."""
        result = await self.collection.update_one(
            {"episode_id": episode_id},
            {"$set": update_data}
        )
        return result.modified_count > 0
    
    async def list_active(self, limit: int = 10, skip: int = 0) -> List[Dict[str, Any]]:
        """List active episodes."""
        cursor = self.collection.find({"is_active": True}).skip(skip).limit(limit)
        episodes = await cursor.to_list(length=limit)
        return self._convert_objectids(episodes)
    
    async def list_all(self, limit: int = 10, skip: int = 0) -> List[Dict[str, Any]]:
        """List all episodes."""
        cursor = self.collection.find().skip(skip).limit(limit)
        episodes = await cursor.to_list(length=limit)
        return self._convert_objectids(episodes)
    
    async def count_active(self) -> int:
        """Count active episodes."""
        return await self.collection.count_documents({"is_active": True})
    
    async def count_all(self) -> int:
        """Count all episodes."""
        return await self.collection.count_documents({})
    
    async def delete(self, episode_id: str) -> bool:
        """Delete episode."""
        result = await self.collection.delete_one({"episode_id": episode_id})
        return result.deleted_count > 0
    
    def _convert_objectids(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert ObjectId to string in documents."""
        for doc in docs:
            if "_id" in doc:
                doc["_id"] = str(doc["_id"])
        return docs
