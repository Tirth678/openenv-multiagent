"""
Budget and token tracking service.
"""

from typing import Dict, Any
from datetime import datetime


class BudgetService:
    """Service for budget management."""
    
    def __init__(self):
        """Initialize service."""
        self.episode_budgets: Dict[str, float] = {}
        self.episode_tokens_used: Dict[str, int] = {}
    
    def allocate_budget(self, episode_id: str, budget: float) -> None:
        """Allocate budget for an episode."""
        self.episode_budgets[episode_id] = budget
        self.episode_tokens_used[episode_id] = 0
    
    def use_tokens(self, episode_id: str, tokens: int) -> bool:
        """Use tokens from budget."""
        if episode_id not in self.episode_budgets:
            return False
        
        current_used = self.episode_tokens_used.get(episode_id, 0)
        budget = self.episode_budgets[episode_id]
        
        if current_used + tokens > budget:
            return False
        
        self.episode_tokens_used[episode_id] = current_used + tokens
        return True
    
    def get_budget_status(self, episode_id: str) -> Dict[str, Any]:
        """Get budget status for an episode."""
        budget = self.episode_budgets.get(episode_id, 0.0)
        used = self.episode_tokens_used.get(episode_id, 0)
        remaining = budget - used
        
        return {
            "budget": budget,
            "used": used,
            "remaining": remaining,
            "ratio": used / budget if budget > 0 else 0.0,
        }
    
    def cleanup_episode(self, episode_id: str) -> None:
        """Clean up budget tracking for an episode."""
        self.episode_budgets.pop(episode_id, None)
        self.episode_tokens_used.pop(episode_id, None)
