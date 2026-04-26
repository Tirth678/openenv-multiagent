"""
Rate limiting middleware.
"""

from fastapi import Request, HTTPException, status
from datetime import datetime, timedelta
from typing import Dict, List
import time


class RateLimiter:
    """Rate limiter for API requests."""
    
    def __init__(self, requests: int = 100, window: int = 60):
        """Initialize rate limiter."""
        self.requests = requests
        self.window = window
        self.requests_by_ip: Dict[str, List[float]] = {}
    
    async def check_rate_limit(self, request: Request) -> bool:
        """Check if request is within rate limit."""
        client_ip = request.client.host if request.client else "unknown"
        current_time = time.time()
        
        if client_ip not in self.requests_by_ip:
            self.requests_by_ip[client_ip] = []
        
        # Remove old requests outside the window
        self.requests_by_ip[client_ip] = [
            req_time for req_time in self.requests_by_ip[client_ip]
            if current_time - req_time < self.window
        ]
        
        # Check if limit exceeded
        if len(self.requests_by_ip[client_ip]) >= self.requests:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
            )
        
        # Add current request
        self.requests_by_ip[client_ip].append(current_time)
        return True


rate_limiter = RateLimiter()
