import logging
import numpy as np
from src.database.neo4j_client import Neo4jClient
from src.database.models import Vacancy, RecommendationScore
from src.ai.embeddings import EmbeddingService
from config import settings

logger = logging.getLogger(__name__)


class VacancyService:
    def __init__(self, neo4j_client, embedding_service):
        self.db = neo4j_client
        self.embedding_service = embedding_service

    def save_vacancy(self, vacancy):
        """Сохранение вакансии с проверкой данных"""
        if not vacancy or not vacancy.id:
            logger.error("Attempted to save empty vacancy")
            return False

        # Проверяем обязательные поля
        if not vacancy.title or not vacancy.description:
            logger.warning(f"Vacancy {vacancy.id} missing title or description")

        try:
            # Генерация эмбеддинга
            if vacancy.description and not vacancy.embedding:
                vacancy.embedding = self.embedding_service.get_embedding(
                    f"{vacancy.title}. {vacancy.description}"
                )

            # Подготавливаем данные
            vacancy_dict = vacancy.to_dict()

            # Проверяем типы данных для Neo4j
            for key, value in vacancy_dict.items():
                if value is None:
                    vacancy_dict[key] = ""
                elif isinstance(value, list):
                    vacancy_dict[key] = [str(item) for item in value] if value else []

            query = """
            MERGE (v:Vacancy {id: $id})
            SET v.title = $title,
                v.description = $description,
                v.salary_from = $salary_from,
                v.salary_to = $salary_to,
                v.currency = $currency,
                v.experience = $experience,
                v.employment = $employment,
                v.published_at = $published_at,
                v.embedding = $embedding,
                v.external_id = $external_id,
                v.company_name = $company_name,
                v.location_name = $location_name,
                v.skills = $skills
            """

            self.db.execute_query(query, vacancy_dict)

            # Сохраняем навыки
            if vacancy.skills:
                for skill_name in vacancy.skills:
                    if not skill_name or not isinstance(skill_name, str):
                        continue

                    skill_id = skill_name.lower().replace(' ', '_').replace('.', '')
                    skill_query = """
                    MERGE (s:Skill {id: $skill_id})
                    SET s.name = $skill_name
                    MERGE (v:Vacancy {id: $vacancy_id})-[:REQUIRES]->(s)
                    """
                    self.db.execute_query(skill_query, {
                        'skill_id': skill_id,
                        'skill_name': skill_name,
                        'vacancy_id': vacancy.id
                    })

            # Сохраняем компанию
            if vacancy.company_name:
                company_id = vacancy.company_name.lower().replace(' ', '_').replace('.', '')
                company_query = """
                MERGE (c:Company {id: $company_id})
                SET c.name = $company_name
                MERGE (v:Vacancy {id: $vacancy_id})-[:FROM_COMPANY]->(c)
                """
                self.db.execute_query(company_query, {
                    'company_id': company_id,
                    'company_name': vacancy.company_name,
                    'vacancy_id': vacancy.id
                })

            # Сохраняем локацию
            if vacancy.location_name:
                location_id = vacancy.location_name.lower().replace(' ', '_').replace('.', '')
                location_query = """
                MERGE (l:Location {id: $location_id})
                SET l.name = $location_name
                MERGE (v:Vacancy {id: $vacancy_id})-[:IN_LOCATION]->(l)
                """
                self.db.execute_query(location_query, {
                    'location_id': location_id,
                    'location_name': vacancy.location_name,
                    'vacancy_id': vacancy.id
                })

            logger.info(f"Successfully saved vacancy: {vacancy.id}")
            return True

        except Exception as e:
            logger.error(f"Error saving vacancy {vacancy.id}: {e}")
            # Пробуем упрощенный вариант сохранения
            try:
                simple_query = """
                MERGE (v:Vacancy {id: $id})
                SET v.title = $title,
                    v.description = $description
                """
                self.db.execute_query(simple_query, {
                    'id': vacancy.id,
                    'title': vacancy.title or "",
                    'description': vacancy.description or ""
                })
                logger.info(f"Saved vacancy {vacancy.id} with minimal data")
                return True
            except Exception as e2:
                logger.error(f"Failed even minimal save for {vacancy.id}: {e2}")
                return False
    def get_recommendations(self, user_id, limit=10):
        """Получение рекомендаций для пользователя"""
        # 1. Контентная фильтрация
        content_scores = self._get_content_recommendations(user_id)

        # 2. Графовая фильтрация
        graph_scores = self._get_graph_recommendations(user_id)

        # 3. Семантическая фильтрация
        semantic_scores = self._get_semantic_recommendations(user_id)

        # Комбинируем
        recommendations = self._combine_recommendations(
            content_scores, graph_scores, semantic_scores, limit
        )

        return recommendations

    def _get_content_recommendations(self, user_id):
        """Контентная фильтрация на основе навыков"""
        query = """
        MATCH (u:User {id: $user_id})
        WITH u.skills AS user_skills

        MATCH (v:Vacancy)
        WITH v, user_skills, v.skills AS vacancy_skills

        // Считаем совпадения
        WITH v, 
             size([skill IN vacancy_skills WHERE skill IN user_skills]) AS matches,
             size(vacancy_skills) AS total_skills

        WHERE total_skills > 0
        RETURN v.id AS vacancy_id, 
               1.0 * matches / total_skills AS score
        ORDER BY score DESC
        LIMIT 100
        """

        results = self.db.execute_query(query, {'user_id': user_id})
        return {r['vacancy_id']: r['score'] for r in results}

    def _get_graph_recommendations(self, user_id):
        """Графовая коллаборативная фильтрация"""
        query = """
        // Находим похожих пользователей
        MATCH (u1:User {id: $user_id})-[:HAS_SKILL]->(s:Skill)<-[:HAS_SKILL]-(u2:User)
        WHERE u1 <> u2

        WITH u2, COUNT(DISTINCT s) AS common_skills

        // Находим вакансии, которые нравятся похожим пользователям
        MATCH (u2)-[:LIKED]->(v:Vacancy)
        WHERE NOT EXISTS((:User {id: $user_id})-[:LIKED|DISLIKED]->(v))

        WITH v, COUNT(DISTINCT u2) AS similar_users
        RETURN v.id AS vacancy_id, 
               similar_users AS score
        ORDER BY score DESC
        LIMIT 100
        """

        results = self.db.execute_query(query, {'user_id': user_id})
        return {r['vacancy_id']: r['score'] for r in results}

    def _get_semantic_recommendations(self, user_id):
        """Семантические рекомендации"""
        # Эмбеддинг пользователя
        user_query = """
        MATCH (u:User {id: $user_id})
        RETURN u.embedding AS embedding
        """

        user_data = self.db.execute_query(user_query, {'user_id': user_id})
        if not user_data or not user_data[0].get('embedding'):
            return {}

        user_embedding = user_data[0]['embedding']

        # Эмбеддинги вакансий
        vacancies_query = """
        MATCH (v:Vacancy)
        WHERE v.embedding IS NOT NULL
        RETURN v.id AS vacancy_id, v.embedding AS vacancy_embedding
        LIMIT 100
        """

        vacancies = self.db.execute_query(vacancies_query)
        scores = {}

        for vac in vacancies:
            if vac['vacancy_embedding']:
                similarity = self.embedding_service.get_similarity(
                    user_embedding, vac['vacancy_embedding']
                )
                scores[vac['vacancy_id']] = max(0, similarity)

        return scores

    def _combine_recommendations(self, content_scores, graph_scores, semantic_scores, limit):
        """Комбинирование рекомендаций"""
        all_ids = set(content_scores.keys()) | set(graph_scores.keys()) | set(semantic_scores.keys())
        recommendations = []

        for vacancy_id in all_ids:
            # Берем оценки
            content_score = content_scores.get(vacancy_id, 0)
            graph_score = graph_scores.get(vacancy_id, 0)
            semantic_score = semantic_scores.get(vacancy_id, 0)

            # Нормализация
            max_content = max(content_scores.values()) if content_scores else 1
            max_graph = max(graph_scores.values()) if graph_scores else 1
            max_semantic = max(semantic_scores.values()) if semantic_scores else 1

            if max_content > 0:
                content_score /= max_content
            if max_graph > 0:
                graph_score /= max_graph
            if max_semantic > 0:
                semantic_score /= max_semantic

            # Общий score (используем строчные атрибуты)
            total_score = (
                    settings.content_weight * content_score +
                    settings.graph_weight * graph_score +
                    settings.semantic_weight * semantic_score
            )

            # Получаем вакансию
            vacancy = self._get_vacancy_by_id(vacancy_id)
            if vacancy:
                recommendations.append(RecommendationScore(
                    vacancy=vacancy,
                    content_score=content_score,
                    graph_score=graph_score,
                    semantic_score=semantic_score,
                    total_score=total_score
                ))

        # Сортировка
        recommendations.sort(key=lambda x: x.total_score, reverse=True)
        return recommendations[:limit]

    def _get_vacancy_by_id(self, vacancy_id):
        """Получение вакансии по ID"""
        query = """
        MATCH (v:Vacancy {id: $vacancy_id})
        RETURN v
        """

        results = self.db.execute_query(query, {'vacancy_id': vacancy_id})
        if not results:
            return None

        vacancy_data = results[0]['v']
        return Vacancy.from_dict(vacancy_data)