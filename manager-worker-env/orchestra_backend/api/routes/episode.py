"""
Episode API routes.
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from datetime import datetime
from services.episode_service import EpisodeService
from models.episode import (
    EpisodeStartRequest, EpisodeStartResponse, ActionRequest, StepResponse,
    EpisodeStateResponse, EpisodeEndResponse, EpisodeListResponse,
    EpisodeHistoryResponse,
)

router = APIRouter(prefix="/episode", tags=["Episodes"])


def get_episode_service(request: Request) -> EpisodeService:
    """Singleton episode service (lives on app.state)."""
    return request.app.state.episode_service


@router.post("/start", response_model=EpisodeStartResponse)
async def start_episode(
    request: EpisodeStartRequest,
    service: EpisodeService = Depends(get_episode_service),
):
    """Start a new episode."""
    try:
        episode = await service.start_episode(
            task_id=request.task_id,
            difficulty=request.difficulty,
            num_workers=request.num_workers,
            budget=request.budget,
        )
        return EpisodeStartResponse(
            episode_id=episode["episode_id"],
            task_id=episode["task_id"],
            status="started",
            timestamp=datetime.utcnow(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{episode_id}", response_model=EpisodeStateResponse)
async def get_episode_state(
    episode_id: str,
    service: EpisodeService = Depends(get_episode_service),
):
    """Get current episode state."""
    try:
        episode = await service.get_episode(episode_id)
        if not episode:
            raise HTTPException(status_code=404, detail="Episode not found")
        return EpisodeStateResponse(
            episode_id=episode_id,
            state=episode.get("current_observation", {}),
            timestamp=datetime.utcnow(),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{episode_id}/step", response_model=StepResponse)
async def step_episode(
    episode_id: str,
    request: ActionRequest,
    service: EpisodeService = Depends(get_episode_service),
):
    """Execute action in episode."""
    try:
        result = await service.step(
            episode_id=episode_id,
            action_id=request.action_id,
            target_worker_id=request.target_worker_id,
        )
        return StepResponse(
            episode_id=episode_id,
            observation=result.get("observation", {}),
            reward=result.get("reward", 0.0),
            done=result.get("done", False),
            info=result.get("info", {}),
            timestamp=datetime.utcnow(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{episode_id}/end", response_model=EpisodeEndResponse)
async def end_episode(
    episode_id: str,
    service: EpisodeService = Depends(get_episode_service),
):
    """End an episode."""
    try:
        result = await service.end_episode(episode_id)
        return EpisodeEndResponse(
            episode_id=episode_id,
            status="ended",
            final_reward=result.get("final_reward", 0.0),
            total_steps=result.get("total_steps", 0),
            timestamp=datetime.utcnow(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("", response_model=EpisodeListResponse)
async def list_episodes(
    skip: int = 0,
    limit: int = 10,
    service: EpisodeService = Depends(get_episode_service),
):
    """List all episodes."""
    try:
        result = await service.list_episodes(limit=limit, skip=skip)
        return EpisodeListResponse(
            episodes=result.get("episodes", []),
            total_count=result.get("total_count", 0),
            timestamp=datetime.utcnow(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{episode_id}/history", response_model=EpisodeHistoryResponse)
async def get_episode_history(
    episode_id: str,
    service: EpisodeService = Depends(get_episode_service),
):
    """Get episode history."""
    try:
        history = await service.get_episode_history(episode_id)
        return EpisodeHistoryResponse(
            episode_id=episode_id,
            history=history,
            timestamp=datetime.utcnow(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
