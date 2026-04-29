from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional
import asyncio

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from api.models import VacancyCreate, VacancyResponse, VacancySearchRequest
from api.dependencies import get_hh_parser, get_vacancy_service

router = APIRouter(prefix="/api/v1/vacancies", tags=["vacancies"])


@router.get("/search", response_model=List[VacancyCreate])
async def search_vacancies(
        query: str = Query("Python", description="Поисковый запрос"),
        area: int = Query(113, description="Регион (113-Россия)"),
        per_page: int = Query(20, ge=1, le=100),
        salary_from: Optional[float] = None,
        parser=Depends(get_hh_parser)
):
    """
    Поиск вакансий через HH.ru API
    """
    try:
        # Синхронный вызов (адаптируем под существующий код)
        items = parser.search_vacancies(
            text=query,
            area=area,
            per_page=per_page,
            page=0
        )

        # Загружаем детали асинхронно
        if items:
            vacancy_ids = [item['id'] for item in items[:per_page]]
            vacancies = await parser.fetch_and_parse_vacancies_async(vacancy_ids)

            # Конвертируем в Pydantic модели
            result = []
            for v in vacancies:
                result.append(VacancyCreate(
                    hh_id=v.external_id,
                    title=v.title,
                    description=v.description,
                    company_name=v.company_name,
                    location_name=v.location_name,
                    salary_from=v.salary_from,
                    salary_to=v.salary_to,
                    skills=v.skills,
                    experience=v.experience,
                    employment=v.employment
                ))
            return result

        return []

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка поиска: {str(e)}")


@router.get("/{vacancy_id}", response_model=VacancyResponse)
async def get_vacancy(
        vacancy_id: str,
        vacancy_service=Depends(get_vacancy_service)
):
    """Получить вакансию по ID"""
    # Адаптируем под ваш VacancyService
    result = vacancy_service.neo4j.execute_query(
        "MATCH (v:Vacancy {hh_id: $hh_id}) RETURN v",
        {'hh_id': vacancy_id}
    )

    if not result:
        raise HTTPException(status_code=404, detail="Vacancy not found")

    v = result[0]['v']
    return VacancyResponse(
        id=v.get('id', ''),
        hh_id=v.get('hh_id', ''),
        title=v.get('title', ''),
        description=v.get('description', ''),
        company_name=v.get('company_name'),
        location_name=v.get('location_name'),
        salary_from=v.get('salary_from'),
        salary_to=v.get('salary_to'),
        salary_currency=v.get('salary_currency', 'RUB'),
        skills=v.get('skills', []),
        experience=v.get('experience'),
        employment=v.get('employment'),
        url=v.get('url'),
        published_at=v.get('published_at')
    )


@router.post("/save", response_model=dict)
async def save_vacancy(
        vacancy: VacancyCreate,
        vacancy_service=Depends(get_vacancy_service)
):
    """Сохранить вакансию в Neo4j"""
    from src.database.models import Vacancy

    # Конвертируем в модель Vacancy
    vac_obj = Vacancy(
        id=f"hh_{vacancy.hh_id}",
        external_id=vacancy.hh_id,
        title=vacancy.title,
        description=vacancy.description,
        company_name=vacancy.company_name,
        location_name=vacancy.location_name,
        salary_from=vacancy.salary_from,
        salary_to=vacancy.salary_to,
        currency=vacancy.salary_currency,
        skills=vacancy.skills,
        experience=vacancy.experience,
        employment=vacancy.employment
    )

    success = vacancy_service.save_vacancy(vac_obj)
    return {"success": success, "vacancy_id": vacancy.hh_id}