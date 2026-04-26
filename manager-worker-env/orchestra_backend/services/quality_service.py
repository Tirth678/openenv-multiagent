"""
Quality and hallucination detection service.
"""

from typing import Dict, Any, List
from datetime import datetime


class QualityService:
    """Service for quality assessment and hallucination detection."""
    
    def __init__(self):
        """Initialize service."""
        self.hallucination_patterns = [
            "i don't know",
            "i'm not sure",
            "i cannot",
            "i can't",
            "unknown",
            "error",
        ]
        self.quality_scores: Dict[str, List[float]] = {}
    
    def detect_hallucination(self, text: str) -> bool:
        """Detect if text contains hallucination indicators."""
        text_lower = text.lower()
        for pattern in self.hallucination_patterns:
            if pattern in text_lower:
                return True
        return False
    
    def assess_quality(self, episode_id: str, output: str, expected: str) -> float:
        """Assess quality of output."""
        if not output:
            return 0.0
        
        # Check for hallucination
        if self.detect_hallucination(output):
            quality = 0.3
        else:
            # Simple similarity check
            output_words = set(output.lower().split())
            expected_words = set(expected.lower().split())
            
            if not expected_words:
                quality = 0.5
            else:
                overlap = len(output_words & expected_words)
                quality = overlap / len(expected_words)
        
        # Track quality score
        if episode_id not in self.quality_scores:
            self.quality_scores[episode_id] = []
        self.quality_scores[episode_id].append(quality)
        
        return quality
    
    def get_episode_quality(self, episode_id: str) -> float:
        """Get average quality for an episode."""
        scores = self.quality_scores.get(episode_id, [])
        if not scores:
            return 0.0
        return sum(scores) / len(scores)
    
    def cleanup_episode(self, episode_id: str) -> None:
        """Clean up quality tracking for an episode."""
        self.quality_scores.pop(episode_id, None)
