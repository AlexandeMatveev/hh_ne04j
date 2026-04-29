from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

# ========== Vacancy Models ==========
class VacancyCreate(BaseModel):
    """Создание вакансии"""
    hh_id: str
    title: str
    description: Optional[str] = None
    company_name: Optional[str] = None
    location_name: Optional[str] = None
    salary_from: Optional[float] = None
    salary_to: Optional[float] = None
    salary_currency: str = "RUB"
    skills: List[str] = []
    experience: Optional[str] = None
    employment: Optional[str] = None
    url: Optional[str] = None

class VacancyResponse(BaseModel):
    """Ответ с вакансией"""
    id: str
    hh_id: str
    title: str
    description: Optional[str]
    company_name: Optional[str]
    location_name: Optional[str]
    salary_from: Optional[float]
    salary_to: Optional[float]
    salary_currency: str
    skills: List[str]
    experience: Optional[str]
    employment: Optional[str]
    url: Optional[str]
    published_at: Optional[datetime]

# ========== User Models ==========
class UserCreate(BaseModel):
    """Создание пользователя"""
    username: str
    resume_text: str
    skills: List[str] = []

class UserResponse(BaseModel):
    """Ответ с пользователем"""
    id: str
    username: str
    resume_text: str
    skills: List[str]
    preferences: dict

# ========== Recommendation Models ==========
class RecommendationRequest(BaseModel):
    """Запрос на рекомендации"""
    user_id: str
    top_n: int = Field(default=10, ge=1, le=50)
    min_salary: Optional[float] = None
    content_weight: Optional[float] = Field(default=0.33, ge=0, le=1)
    graph_weight: Optional[float] = Field(default=0.34, ge=0, le=1)
    semantic_weight: Optional[float] = Field(default=0.33, ge=0, le=1)

class RecommendationScoreResponse(BaseModel):
    """Оценка рекомендации"""
    vacancy: VacancyResponse
    content_score: float
    graph_score: float
    semantic_score: float
    total_score: float

class RecommendationResponse(BaseModel):
    """Ответ с рекомендациями"""
    user_id: str
    recommendations: List[RecommendationScoreResponse]
    total_found: int

# ========== Feedback Models ==========
class FeedbackCreate(BaseModel):
    """Создание обратной связи"""
    user_id: str
    vacancy_id: str
    feedback_type: str  # LIKED, DISLIKED, VIEWED, APPLIED

# ========== Search Models ==========
class VacancySearchRequest(BaseModel):
    """Поиск вакансий"""
    query: str = "Python"
    area: int = 113
    per_page: int = 20
    page: int = 0
    salary_from: Optional[float] = None
    only_with_salary: bool = False

class VacancySearchResponse(BaseModel):
    """Ответ поиска"""
    query: str
    total_found: int
    vacancies: List[VacancyCreate]