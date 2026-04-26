"""
Model API routes.
"""

from fastapi import APIRouter
from datetime import datetime
from models.common import ModelInfoResponse, CheckpointResponse

router = APIRouter(prefix="/model", tags=["Model"])


@router.get("/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get model information."""
    return ModelInfoResponse(
        model_name="ppo_manager",
        model_version="1.0.0",
        framework="stable-baselines3",
        algorithm="PPO",
        timestamp=datetime.utcnow(),
    )


@router.post("/checkpoint", response_model=CheckpointResponse)
async def save_checkpoint():
    """Save model checkpoint."""
    import time
    return CheckpointResponse(
        checkpoint_id=f"checkpoint_{int(time.time())}",
        status="saved",
        timestamp=datetime.utcnow(),
    )
