import logging
import json
from src.database.neo4j_client import Neo4jClient
from src.database.models import User, FeedbackType
from src.ai.embeddings import EmbeddingService
from config import settings

logger = logging.getLogger(__name__)


class UserService:
    def __init__(self, neo4j_client, embedding_service):
        self.db = neo4j_client
        self.embedding_service = embedding_service

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