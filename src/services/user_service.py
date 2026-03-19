import logging
import json
from typing import List, Dict

from src.database.neo4j_client import Neo4jClient
from src.database.models import User, FeedbackType
from src.ai.embeddings import EmbeddingService
from config import settings

logger = logging.getLogger(__name__)


class UserService:
    def __init__(self, neo4j_client, embedding_service):
        self.db = neo4j_client
        self.embedding_service = embedding_service
        self.neo4j = neo4j_client

    def get_similar_users_vacancies(self, user_id: str, limit: int = 5) -> List[Dict]:
        """
        Находит вакансии, которые понравились пользователям с похожими навыками
        """
        query = """
        MATCH (u:User {id: $user_id})-[:HAS_SKILL]->(skill:Skill)
        WITH u, collect(skill.name) AS user_skills

        // Ищем других пользователей с общими навыками
        MATCH (other:User)-[:HAS_SKILL]->(common_skill:Skill)
        WHERE other.id <> u.id 
          AND common_skill.name IN user_skills

        WITH other, COUNT(common_skill) AS common_count
        ORDER BY common_count DESC
        LIMIT 10

        // Получаем вакансии, которым они поставили лайк
        MATCH (other)-[r:LIKED]->(v:Vacancy)

        // Явно собираем навыки вакансии
        OPTIONAL MATCH (v)-[:REQUIRES]->(vs:Skill)
        WITH 
            v, 
            other,
            COUNT(r) AS like_count,
            COLLECT(DISTINCT vs.name) AS vacancy_skills

        // Фильтруем пустые — ИСПРАВЛЕНО: используем IS NOT NULL
        WHERE size(vacancy_skills) > 0 OR v.skills IS NOT NULL

        RETURN 
            v.id AS vacancy_id,
            v.title AS title,
            v.company_name AS company,
            v.salary_from AS salary_from,
            v.salary_to AS salary_to,
            v.currency AS currency,
            COALESCE(vacancy_skills, v.skills, []) AS skills,
            like_count
        ORDER BY like_count DESC
        LIMIT $limit
        """

        result = self.neo4j.execute_query(query, {'user_id': user_id, 'limit': limit})
        return result

    def create_or_update_user(self, user):
        """Создание или обновление пользователя"""
        # Генерация эмбеддинга
        if user.resume_text and not user.embedding:
            user.embedding = self.embedding_service.get_embedding(user.resume_text)

        query = """
        MERGE (u:User {id: $id})
        SET u.username = $username,
            u.resume_text = $resume_text,
            u.embedding = $embedding,
            u.preferences = $preferences,
            u.skills = $skills
        """

        params = {
            'id': user.id,
            'username': user.username,
            'resume_text': user.resume_text or "",
            'embedding': user.embedding,
            'preferences': json.dumps(user.preferences) if user.preferences else "{}",
            'skills': user.skills
        }

        try:
            self.db.execute_query(query, params)

            # Создаем узлы навыков
            for skill_name in user.skills:
                skill_id = skill_name.lower().replace(' ', '_')
                skill_query = """
                MERGE (s:Skill {id: $skill_id})
                SET s.name = $skill_name
                MERGE (u:User {id: $user_id})-[:HAS_SKILL]->(s)
                """
                self.db.execute_query(skill_query, {
                    'skill_id': skill_id,
                    'skill_name': skill_name,
                    'user_id': user.id
                })

            return True
        except Exception as e:
            logger.error(f"Error saving user {user.id}: {e}")
            return False

    def get_user_by_id(self, user_id):
        """Получение пользователя по ID"""
        query = """
        MATCH (u:User {id: $user_id})
        RETURN u
        """

        results = self.db.execute_query(query, {'user_id': user_id})
        if not results:
            return None

        user_data = results[0]['u']
        return User.from_dict(user_data)



    def update_user_preferences(self, user_id, feedback_type, skill_weights):
        """Обновление предпочтений пользователя"""
        user = self.get_user_by_id(user_id)
        if not user:
            return

        current_prefs = user.preferences

        # Обновляем веса
        for skill_name, weight_change in skill_weights.items():
            skill_id = skill_name.lower().replace(' ', '_')
            current_weight = current_prefs.get(skill_id, 0)

            if feedback_type == FeedbackType.LIKE:
                new_weight = current_weight + settings.learning_rate * weight_change
            else:  # DISLIKE
                new_weight = current_weight - settings.learning_rate * weight_change

            # Регуляризация
            new_weight = new_weight * (1 - settings.regularization_lambda)
            current_prefs[skill_id] = max(0, min(1, new_weight))

        # Сохраняем
        update_query = """
        MATCH (u:User {id: $user_id})
        SET u.preferences = $preferences
        """

        self.db.execute_query(update_query, {
            'user_id': user_id,
            'preferences': json.dumps(current_prefs)
        })