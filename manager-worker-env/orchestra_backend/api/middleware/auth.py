"""
JWT authentication middleware.
"""

from fastapi import HTTPException, status


async def verify_token(token: str) -> dict:
    """Verify JWT token."""
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token",
        )
    
    # Placeholder for actual JWT verification
    # In production, use PyJWT to verify the token
    return {"user_id": "test_user"}
