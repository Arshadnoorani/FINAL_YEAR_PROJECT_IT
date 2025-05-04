# health.py - Health check API route

from fastapi import APIRouter

router = APIRouter()

@router.get("/health")
def health_check():
    """Returns API health status."""
    return {"status": "API is running!"}
