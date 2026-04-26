"""
Configuration settings for OrchestraAI Backend.
"""

from typing import List, Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # Server
    APP_NAME: str = "OrchestraAI Backend"
    APP_VERSION: str = "1.0.0"
    # Vercel: set DEBUG=false in Environment Variables
    DEBUG: bool = True
    SERVER_HOST: str = "0.0.0.0"
    SERVER_PORT: int = 8000
    
    # CORS — optional extra origins via Vercel: CORS_EXTRA_ORIGINS as JSON, e.g. ["https://myapp.vercel.app"]
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]
    CORS_EXTRA_ORIGINS: List[str] = []
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: List[str] = ["*"]
    CORS_ALLOW_HEADERS: List[str] = ["*"]
    
    # MongoDB (set MONGODB_URL in Vercel / .env — no default in production)
    MONGODB_URL: Optional[str] = None
    MONGODB_DB_NAME: str = "openenv-project"
    
    # Environment
    MAX_EPISODES: int = 100
    EPISODE_TIMEOUT: int = 3600  # 1 hour
    
    # WebSocket
    MAX_CONNECTIONS: int = 100
    MESSAGE_QUEUE_SIZE: int = 1000
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    # JWT (for future auth)
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 60  # seconds
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
