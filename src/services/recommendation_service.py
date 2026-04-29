# src/services/recommendation_service.py
from typing import List, Dict, Any, Optional
import logging
from src.database.neo4j_client import Neo4jClient
from src.ai.embeddings import EmbeddingService
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class RecommendationService:
    def __init__(self, neo4j_client: Neo4jClient, embedding_service: EmbeddingService):
        self.neo4j = neo4j_client
        self.embeddings = embedding_service

    def get_recommendations_for_user(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Получить рекомендации для пользователя"""
        try:
            # Получаем профиль пользователя
            user = self.neo4j.execute_query(
                "MATCH (u:User {id: $user_id}) RETURN u",
                {'user_id': user_id}
            )

            if not user:
                logger.warning(f"User {user_id} not found")
                return []

            # Получаем просмотренные вакансии пользователя
            viewed_vacancies = self.neo4j.execute_query("""
                MATCH (u:User {id: $user_id})-[:VIEWED]->(v:Vacancy)
                RETURN v.id as id, v.title as title, v.skills as skills
                LIMIT 10
            """, {'user_id': user_id})

            # Получаем избранные вакансии
            favorite_vacancies = self.neo4j.execute_query("""
                MATCH (u:User {id: $user_id})-[:FAVORITED]->(v:Vacancy)
                RETURN v.id as id, v.title as title, v.skills as skills
                LIMIT 10
            """, {'user_id': user_id})

            # Формируем предпочтения пользователя
            user_skills = set()
            for vac in (viewed_vacancies or []) + (favorite_vacancies or []):
                if vac.get('skills'):
                    user_skills.update(vac['skills'])

            # Рекомендуем вакансии на основе навыков
            recommendations = self.get_recommendations_by_skills(
                list(user_skills),
                exclude_user_id=user_id,
                limit=limit
            )

            return recommendations

        except Exception as e:
            logger.error(f"Error getting recommendations for user {user_id}: {e}")
            return []

    def get_recommendations_by_skills(self, skills: List[str], exclude_user_id: str = None, limit: int = 10) -> List[
        Dict[str, Any]]:
        """Получить рекомендации на основе навыков"""
        try:
            if not skills:
                return []

            # Ищем вакансии с похожими навыками
            query = """
            MATCH (v:Vacancy)
            WHERE ANY(skill IN $skills WHERE skill IN v.skills)
            """

            params = {'skills': skills}

            if exclude_user_id:
                query += " AND NOT EXISTS { MATCH (u:User {id: $exclude_user_id})-[:VIEWED|FAVORITED]->(v) }"
                params['exclude_user_id'] = exclude_user_id

            query += """
            RETURN v.id as id, 
                   v.title as title, 
                   v.company_name as company_name,
                   v.location_name as location_name,
                   v.salary_from as salary_from,
                   v.salary_to as salary_to,
                   v.salary_currency as salary_currency,
                   v.skills as skills,
                   v.url as url
            ORDER BY v.published_at DESC
            LIMIT $limit
            """

            params['limit'] = limit

            results = self.neo4j.execute_query(query, params)

            # Сортируем по релевантности (количество совпадающих навыков)
            if results:
                for result in results:
                    matching_skills = set(skills) & set(result.get('skills', []))
                    result['matching_skills_count'] = len(matching_skills)
                    result['matching_skills'] = list(matching_skills)

                results.sort(key=lambda x: x.get('matching_skills_count', 0), reverse=True)

            return results or []

        except Exception as e:
            logger.error(f"Error getting recommendations by skills: {e}")
            return []

    def get_similar_vacancies(self, vacancy_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Найти похожие вакансии"""
        try:
            # Получаем текущую вакансию
            vacancy = self.neo4j.execute_query(
                "MATCH (v:Vacancy {id: $vacancy_id}) RETURN v",
                {'vacancy_id': vacancy_id}
            )

            if not vacancy:
                logger.warning(f"Vacancy {vacancy_id} not found")
                return []

            vacancy_skills = vacancy[0].get('v', {}).get('skills', [])

            if not vacancy_skills:
                return []

            # Ищем вакансии с похожими навыками
            results = self.neo4j.execute_query("""
                MATCH (v:Vacancy)
                WHERE v.id <> $vacancy_id
                AND ANY(skill IN $skills WHERE skill IN v.skills)
                RETURN v.id as id, 
                       v.title as title, 
                       v.company_name as company_name,
                       v.location_name as location_name,
                       v.salary_from as salary_from,
                       v.salary_to as salary_to,
                       v.skills as skills,
                       v.url as url
                ORDER BY v.published_at DESC
                LIMIT $limit
            """, {
                'vacancy_id': vacancy_id,
                'skills': vacancy_skills,
                'limit': limit
            })

            # Сортируем по релевантности
            if results:
                for result in results:
                    matching_skills = set(vacancy_skills) & set(result.get('skills', []))
                    result['similarity_score'] = len(matching_skills) / len(vacancy_skills) if vacancy_skills else 0

                results.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)

            return results or []

        except Exception as e:
            logger.error(f"Error getting similar vacancies: {e}")
            return []

    def get_hybrid_recommendations(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Гибридные рекомендации (коллаборативная фильтрация + контентная)"""
        try:
            # 1. Получаем рекомендации на основе навыков
            user = self.neo4j.execute_query(
                "MATCH (u:User {id: $user_id}) RETURN u",
                {'user_id': user_id}
            )

            if not user:
                return []

            # Получаем навыки пользователя из просмотренных вакансий
            user_skills_query = """
            MATCH (u:User {id: $user_id})-[:VIEWED|FAVORITED]->(v:Vacancy)
            UNWIND v.skills AS skill
            RETURN DISTINCT skill as skill, COUNT(*) as frequency
            ORDER BY frequency DESC
            LIMIT 20
            """

            skills_result = self.neo4j.execute_query(user_skills_query, {'user_id': user_id})

            if not skills_result:
                # Если нет истории, рекомендуем популярные вакансии
                return self.get_popular_vacancies(limit)

            top_skills = [item['skill'] for item in skills_result[:10]]

            # Получаем рекомендации по навыкам
            content_recommendations = self.get_recommendations_by_skills(
                top_skills,
                exclude_user_id=user_id,
                limit=limit
            )

            # 2. Получаем коллаборативные рекомендации (пользователи с похожими интересами)
            collaborative_query = """
            MATCH (u:User {id: $user_id})-[:VIEWED|FAVORITED]->(v:Vacancy)<-[:VIEWED|FAVORITED]-(other:User)
            WHERE other.id <> $user_id
            MATCH (other)-[:VIEWED|FAVORITED]->(rec:Vacancy)
            WHERE NOT EXISTS((u)-[:VIEWED|FAVORITED]->(rec))
            RETURN rec.id as id, 
                   rec.title as title,
                   rec.company_name as company_name,
                   COUNT(*) as similarity_score
            ORDER BY similarity_score DESC
            LIMIT $limit
            """

            collaborative_results = self.neo4j.execute_query(collaborative_query, {
                'user_id': user_id,
                'limit': limit
            })

            # Объединяем результаты
            seen_ids = set()
            recommendations = []

            # Добавляем контентные рекомендации
            for rec in content_recommendations:
                if rec.get('id') not in seen_ids:
                    seen_ids.add(rec.get('id'))
                    rec['recommendation_type'] = 'content_based'
                    recommendations.append(rec)

            # Добавляем коллаборативные рекомендации
            for rec in collaborative_results or []:
                if rec.get('id') not in seen_ids:
                    seen_ids.add(rec.get('id'))
                    rec['recommendation_type'] = 'collaborative'
                    rec['matching_skills_count'] = 0
                    recommendations.append(rec)

            return recommendations[:limit]

        except Exception as e:
            logger.error(f"Error getting hybrid recommendations: {e}")
            return []

    def get_popular_vacancies(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Получить популярные вакансии"""
        try:
            query = """
            MATCH (v:Vacancy)
            OPTIONAL MATCH (v)<-[r:VIEWED]-(u:User)
            RETURN v.id as id,
                   v.title as title,
                   v.company_name as company_name,
                   v.location_name as location_name,
                   v.salary_from as salary_from,
                   v.salary_to as salary_to,
                   v.skills as skills,
                   COUNT(DISTINCT u) as views_count
            ORDER BY views_count DESC, v.published_at DESC
            LIMIT $limit
            """

            results = self.neo4j.execute_query(query, {'limit': limit})
            return results or []

        except Exception as e:
            logger.error(f"Error getting popular vacancies: {e}")
            return []