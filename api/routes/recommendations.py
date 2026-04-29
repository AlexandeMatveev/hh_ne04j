from fastapi import APIRouter, HTTPException, Depends
from typing import List

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from api.models import RecommendationRequest, RecommendationResponse, RecommendationScoreResponse, VacancyResponse
from api.dependencies import get_vacancy_service, get_user_service

router = APIRouter(prefix="/api/v1/recommendations", tags=["recommendations"])


@router.post("/", response_model=RecommendationResponse)
async def get_recommendations(
        request: RecommendationRequest,
        vacancy_service=Depends(get_vacancy_service),
        user_service=Depends(get_user_service)
):
    """
    Получение персональных рекомендаций
    """
    # Проверяем существование пользователя
    user = user_service.get_user_by_id(request.user_id)
    if not user:
        raise HTTPException(status_code=404, detail=f"User {request.user_id} not found")

    # Получаем рекомендации из вашего сервиса
    recommendations = vacancy_service.get_recommendations(
        request.user_id,
        request.top_n,
        content_weight=request.content_weight,
        graph_weight=request.graph_weight,
        semantic_weight=request.semantic_weight
    )

    # Конвертируем в ответ
    result = []
    for rec in recommendations:
        v = rec.vacancy
        result.append(RecommendationScoreResponse(
            vacancy=VacancyResponse(
                id=v.id,
                hh_id=v.external_id,
                title=v.title,
                description=v.description,
                company_name=v.company_name,
                location_name=v.location_name,
                salary_from=v.salary_from,
                salary_to=v.salary_to,
                salary_currency=v.currency,
                skills=v.skills,
                experience=v.experience,
                employment=v.employment,
                url=getattr(v, 'url', None),
                published_at=v.published_at
            ),
            content_score=rec.content_score,
            graph_score=rec.graph_score,
            semantic_score=rec.semantic_score,
            total_score=rec.total_score
        ))

    return RecommendationResponse(
        user_id=request.user_id,
        recommendations=result,
        total_found=len(result)
    )