from fastapi import APIRouter, HTTPException, Depends

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from api.models import FeedbackCreate
from api.dependencies import get_feedback_service

router = APIRouter(prefix="/api/v1/feedback", tags=["feedback"])


@router.post("/")
async def add_feedback(
        feedback: FeedbackCreate,
        feedback_service=Depends(get_feedback_service)
):
    """Добавить обратную связь по вакансии"""
    from src.database.models import UserFeedback, FeedbackType

    # Конвертируем строку в enum
    feedback_type_map = {
        "LIKED": FeedbackType.LIKE,
        "DISLIKED": FeedbackType.DISLIKE,
        "VIEWED": FeedbackType.VIEW,
        "APPLIED": FeedbackType.APPLY
    }

    fb_type = feedback_type_map.get(feedback.feedback_type)
    if not fb_type:
        raise HTTPException(status_code=400, detail="Invalid feedback type")

    user_feedback = UserFeedback(
        user_id=feedback.user_id,
        vacancy_id=feedback.vacancy_id,
        feedback_type=fb_type
    )

    success = feedback_service.record_feedback(user_feedback)
    return {"success": success, "message": "Feedback recorded"}
