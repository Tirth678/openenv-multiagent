"""
MongoDB connection and utilities.
"""

import logging
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from config import settings

logger = logging.getLogger(__name__)


class MongoDB:
    """MongoDB connection manager."""
    
    client: Optional[AsyncIOMotorClient] = None
    db: Optional[AsyncIOMotorDatabase] = None
    
    @classmethod
    async def connect(cls) -> None:
        """Connect to MongoDB, or no-op if MONGODB_URL is unset (in-memory only)."""
        if not settings.MONGODB_URL:
            cls.client = None
            cls.db = None
            logger.warning(
                "MONGODB_URL not set; API runs without database persistence (episodes/metrics in-process only)"
            )
            return
        
        cls.client = AsyncIOMotorClient(settings.MONGODB_URL)
        cls.db = cls.client[settings.MONGODB_DB_NAME]
        
        # Create indexes
        await cls._create_indexes()
        
        print(f"✓ Connected to MongoDB: {settings.MONGODB_DB_NAME}")
    
    @classmethod
    async def disconnect(cls) -> None:
        """Disconnect from MongoDB."""
        if cls.client:
            cls.client.close()
            print("✓ Disconnected from MongoDB")
        cls.client = None
        cls.db = None
    
    @classmethod
    async def _create_indexes(cls) -> None:
        """Create database indexes."""
        if cls.db is None:
            return
        
        # Episodes collection indexes
        episodes = cls.db["episodes"]
        await episodes.create_index("episode_id", unique=True)
        await episodes.create_index("created_at")
        await episodes.create_index("is_active")
        
        # Metrics collection indexes
        metrics = cls.db["metrics"]
        await metrics.create_index("timestamp")
        await metrics.create_index("episode_id")
        
        # Training sessions collection indexes
        sessions = cls.db["training_sessions"]
        await sessions.create_index("session_id", unique=True)
        await sessions.create_index("created_at")
        
        print("✓ Database indexes created")
    
    @classmethod
    def get_db(cls) -> Optional[AsyncIOMotorDatabase]:
        """Get database instance (None if running without MongoDB)."""
        return cls.db


def get_db() -> Optional[AsyncIOMotorDatabase]:
    """FastAPI dependency: database or None if persistence disabled."""
    return MongoDB.get_db()
