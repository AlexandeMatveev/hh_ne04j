from fastapi import APIRouter, Depends
from typing import Dict, Any
from api.dependencies import get_neo4j_client
from src.database.neo4j_client import Neo4jClient

router = APIRouter()


@router.get("/stats")
async def get_stats(neo4j: Neo4jClient = Depends(get_neo4j_client)):
    """Получить статистику системы"""
    try:
        # Количество вакансий
        vacancies_count = neo4j.execute_query("MATCH (v:Vacancy) RETURN COUNT(v) as count")

        # Количество пользователей
        users_count = neo4j.execute_query("MATCH (u:User) RETURN COUNT(u) as count")

        # Количество просмотров
        views_count = neo4j.execute_query("MATCH ()-[r:VIEWED]->() RETURN COUNT(r) as count")

        return {
            "vacancies": vacancies_count[0]['count'] if vacancies_count else 0,
            "users": users_count[0]['count'] if users_count else 0,
            "views": views_count[0]['count'] if views_count else 0
        }
    except Exception as e:
        return {"error": str(e)}