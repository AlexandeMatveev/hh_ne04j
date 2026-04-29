# api/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import sys
from pathlib import Path

# Добавляем корень проекта в путь
sys.path.append(str(Path(__file__).parent.parent))

from api.routes import vacancies, users, recommendations, feedback, analytics
from api.dependencies import (
    get_neo4j_client,
    get_embedding_service,
    get_vacancy_service,
    get_user_service,
    get_recommendation_service,
    get_hh_parser
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Создаем приложение
app = FastAPI(
    title="Vacancy Recommendation System",
    description="Система рекомендаций вакансий на основе AI",
    version="1.0.0"
)

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Инициализация при старте
@app.on_event("startup")
async def startup_event():
    """Инициализация при запуске"""
    logger.info("Starting up application...")

    # Инициализируем сервисы
    try:
        neo4j = get_neo4j_client()
        logger.info("✅ Neo4j client initialized")

        embeddings = get_embedding_service()
        logger.info("✅ Embedding service initialized")

        vacancy_service = get_vacancy_service()
        logger.info("✅ Vacancy service initialized")

        user_service = get_user_service()
        logger.info("✅ User service initialized")

        recommendation_service = get_recommendation_service()
        logger.info("✅ Recommendation service initialized")

        hh_parser = get_hh_parser()
        logger.info("✅ HH Parser initialized")

    except Exception as e:
        logger.error(f"❌ Failed to initialize services: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Очистка при завершении"""
    logger.info("Shutting down application...")

    # Закрываем соединения
    try:
        neo4j = get_neo4j_client()
        neo4j.close()
        logger.info("✅ Neo4j connection closed")
    except Exception as e:
        logger.error(f"❌ Error closing Neo4j: {e}")


# Подключаем роутеры
app.include_router(vacancies.router, prefix="/api/vacancies", tags=["vacancies"])
app.include_router(users.router, prefix="/api/users", tags=["users"])
app.include_router(recommendations.router, prefix="/api/recommendations", tags=["recommendations"])
app.include_router(feedback.router, prefix="/api/feedback", tags=["feedback"])
app.include_router(analytics.router, prefix="/api/analytics", tags=["analytics"])


@app.get("/")
async def root():
    """Корневой эндпоинт"""
    return {
        "message": "Vacancy Recommendation System API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health_check():
    """Проверка здоровья системы"""
    health_status = {
        "status": "healthy",
        "services": {}
    }

    # Проверяем Neo4j
    try:
        neo4j = get_neo4j_client()
        result = neo4j.execute_query("RETURN 1 as test")
        health_status["services"]["neo4j"] = "up" if result else "down"
    except Exception as e:
        health_status["services"]["neo4j"] = f"error: {str(e)}"
        health_status["status"] = "unhealthy"

    # Проверяем эмбеддинги
    try:
        embeddings = get_embedding_service()
        health_status["services"]["embeddings"] = "up"
    except Exception as e:
        health_status["services"]["embeddings"] = f"error: {str(e)}"
        health_status["status"] = "unhealthy"

    return health_status