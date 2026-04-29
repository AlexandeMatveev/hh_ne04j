from fastapi import APIRouter, HTTPException, Depends

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from api.models import UserCreate, UserResponse
from api.dependencies import get_user_service

router = APIRouter(prefix="/api/v1/users", tags=["users"])


@router.post("/", response_model=UserResponse)
async def create_user(
        user_data: UserCreate,
        user_service=Depends(get_user_service)
):
    """Создание нового пользователя"""
    from src.database.models import User
    import time

    user_id = f"user_{int(time.time())}"
    new_user = User(
        id=user_id,
        username=user_data.username,
        resume_text=user_data.resume_text,
        skills=user_data.skills
    )

    success = user_service.create_or_update_user(new_user)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to create user")

    return UserResponse(
        id=user_id,
        username=user_data.username,
        resume_text=user_data.resume_text,
        skills=user_data.skills,
        preferences={}
    )


@router.get("/{user_id}", response_model=UserResponse)
async def get_user(
        user_id: str,
        user_service=Depends(get_user_service)
):
    """Получение пользователя по ID"""
    user = user_service.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return UserResponse(
        id=user.id,
        username=user.username,
        resume_text=user.resume_text or "",
        skills=user.skills,
        preferences=user.preferences
    )