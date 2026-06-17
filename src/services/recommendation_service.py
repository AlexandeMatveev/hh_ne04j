# src/services/recommendation_service.py
from typing import List, Dict, Any, Optional
import logging
from src.database.neo4j_client import Neo4jClient
from src.ai.embeddings import EmbeddingService

logger = logging.getLogger(__name__)


class RecommendationService:
    def __init__(self, neo4j_client: Neo4jClient, embedding_service: EmbeddingService):
        self.neo4j = neo4j_client
        self.embeddings = embedding_service

    def _execute_recommendation_query(self, query: str, params: Dict) -> List[Dict]:
        """Выполнить запрос рекомендаций с обработкой ошибок"""
        try:
            return self.neo4j.execute_query(query, params) or []
        except Exception as e:
            logger.error(f"Error executing recommendation query: {e}")
            return []

    def _filter_and_score_vacancies(self, results: List[Dict], skills: List[str] = None) -> List[Dict]:
        """Отфильтровать и добавить баллы совпадения"""
        if not results:
            return []
        
        for result in results:
            vacancy_skills = set(result.get('skills', []))
            if skills:
                matching = set(skills) & vacancy_skills
                result['matching_skills_count'] = len(matching)
                result['matching_skills'] = list(matching)
            
            # Проверка заголовка
            title = result.get('title', '')
            if not title or title.strip() == '' or title == 'Без названия':
                continue
                
        # Удаляем вакансии с плохим заголовком
        results = [r for r in results 
                   if r.get('title') and r.get('title').strip() and r.get('title') != 'Без названия']
        
        # Сортировка
        if results and skills:
            results.sort(key=lambda x: x.get('matching_skills_count', 0), reverse=True)
        
        return results

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

            # Получаем навыки из просмотренных вакансий
            user_skills_query = """
            MATCH (u:User {id: $user_id})-[:VIEWED|:RATED]->(v:Vacancy)
            UNWIND v.skills AS skill
            RETURN DISTINCT skill as skill, COUNT(*) as frequency
            ORDER BY frequency DESC
            LIMIT 20
            """
            
            skills_result = self.neo4j.execute_query(user_skills_query, {'user_id': user_id})
            user_skills = [item['skill'] for item in skills_result[:10]] if skills_result else []
            
            return self.get_recommendations_by_skills(user_skills, exclude_user_id=user_id, limit=limit)
        except Exception as e:
            logger.error(f"Error getting recommendations for user {user_id}: {e}")
            return []

    def get_recommendations_by_skills(self, skills: List[str], exclude_user_id: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Получить рекомендации на основе навыков"""
        if not skills:
            return []

        query = """
        MATCH (v:Vacancy)
        WHERE ANY(skill IN $skills WHERE skill IN v.skills)
        AND v.title IS NOT NULL
        AND v.title <> ''
        AND v.title <> 'Без названия'
        """
        
        params = {'skills': skills}
        
        if exclude_user_id:
            query += " AND NOT EXISTS { MATCH (u:User {id: $exclude_user_id})-[:VIEWED|:FAVORITED|:LIKED]->(v) }"
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
        results = self._execute_recommendation_query(query, params)
        return self._filter_and_score_vacancies(results, skills)

    def get_similar_vacancies(self, vacancy_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Найти похожие вакансии"""
        try:
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

            results = self._execute_recommendation_query("""
                MATCH (v:Vacancy)
                WHERE v.id <> $vacancy_id
                AND ANY(skill IN $skills WHERE skill IN v.skills)
                AND v.title IS NOT NULL
                AND v.title <> ''
                AND v.title <> 'Без названия'
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

            for result in results:
                vacancy_skills_set = set(vacancy_skills)
                matching_skills = vacancy_skills_set & set(result.get('skills', []))
                result['similarity_score'] = len(matching_skills) / len(vacancy_skills_set) if vacancy_skills_set else 0

            return results or []
        except Exception as e:
            logger.error(f"Error getting similar vacancies: {e}")
            return []

    def get_hybrid_recommendations(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Гибридные рекомендации (коллаборативная фильтрация + контентная)"""
        try:
            # Получаем навыки пользователя
            skills_result = self.neo4j.execute_query("""
                MATCH (u:User {id: $user_id})-[:VIEWED|:LIKED|:RATED]->(v:Vacancy)
                UNWIND v.skills AS skill
                RETURN DISTINCT skill as skill, COUNT(*) as frequency
                ORDER BY frequency DESC
                LIMIT 20
            """, {'user_id': user_id})

            if not skills_result:
                return self.get_popular_vacancies(limit)

            top_skills = [item['skill'] for item in skills_result[:10]]

            # Контентные рекомендации
            content_recs = self.get_recommendations_by_skills(
                top_skills,
                exclude_user_id=user_id,
                limit=limit
            )
            
            for rec in content_recs:
                rec['recommendation_type'] = 'content_based'

            # Коллаборативные рекомендации
            collaborative_results = self._execute_recommendation_query("""
                MATCH (u:User {id: $user_id})-[:VIEWED|:LIKED|:RATED]->(v:Vacancy)<-[:VIEWED|:LIKED|:RATED]-(other:User)
                WHERE other.id <> $user_id
                MATCH (other)-[:VIEWED|:LIKED|:RATED]->(rec:Vacancy)
                WHERE NOT EXISTS((u)-[:VIEWED|:LIKED|:DISLIKED|:RATED]->(rec))
                RETURN rec.id as id, 
                       rec.title as title,
                       rec.company_name as company_name,
                       COUNT(*) as similarity_score
                ORDER BY similarity_score DESC
                LIMIT $limit
            """, {
                'user_id': user_id,
                'limit': limit
            })
            
            for rec in collaborative_results:
                rec['recommendation_type'] = 'collaborative'
                rec['matching_skills_count'] = 0

            # Объединяем и дедуплицируем
            seen_ids = set()
            recommendations = []
            
            for rec in content_recs + collaborative_results:
                if rec.get('id') not in seen_ids:
                    seen_ids.add(rec.get('id'))
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
            results = self._execute_recommendation_query(query, {'limit': limit})
            return results or []
        except Exception as e:
            logger.error(f"Error getting popular vacancies: {e}")
            return []


# === backward compatibility ===

# Убираем неиспользуемые импорты (numpy, sklearn) - не нужные для работы
# Если не нужна косинусная близость - можно полностью убрать
# В текущей версии косинусная близость не используется
