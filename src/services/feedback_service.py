import logging
from src.database.neo4j_client import Neo4jClient
from src.database.models import UserFeedback, FeedbackType
from src.services.user_service import UserService
from config import settings

logger = logging.getLogger(__name__)


class FeedbackService:
    def __init__(self, neo4j_client, user_service):
        self.db = neo4j_client
        self.user_service = user_service

    def record_feedback(self, feedback):
        """Запись обратной связи"""
        relationship_type = feedback.feedback_type.value

        query = f"""
        MATCH (u:User {{id: $user_id}})
        MATCH (v:Vacancy {{id: $vacancy_id}})
        MERGE (u)-[r:{relationship_type}]->(v)
        SET r.timestamp = $timestamp,
            r.interaction_time = $interaction_time
        """

        params = feedback.to_dict()

        try:
            self.db.execute_query(query, params)

            # Обновляем предпочтения для лайков/дизлайков
            if feedback.feedback_type in [FeedbackType.LIKE, FeedbackType.DISLIKE]:
                self._update_user_preferences(feedback)

            return True
        except Exception as e:
            logger.error(f"Error recording feedback: {e}")
            return False

    def _update_user_preferences(self, feedback):
        """Обновление предпочтений пользователя"""
        # Получаем навыки вакансии
        skills_query = """
        MATCH (v:Vacancy {id: $vacancy_id})
        RETURN v.skills AS skills
        """

        results = self.db.execute_query(skills_query, {'vacancy_id': feedback.vacancy_id})
        if not results:
            return

        skills = results[0].get('skills', [])
        if not skills:
            return

        # Веса навыков
        skill_weights = {}
        weight_per_skill = 1.0 / len(skills)

        for skill in skills:
            if isinstance(skill, str):
                skill_id = skill.lower().replace(' ', '_')
                skill_weights[skill_id] = weight_per_skill

        # Обновляем предпочтения
        self.user_service.update_user_preferences(
            feedback.user_id,
            feedback.feedback_type,
            skill_weights
        )

    def get_user_feedback_history(self, user_id, limit=20):
        """История обратной связи пользователя"""
        query = """
        MATCH (u:User {id: $user_id})-[r]->(v:Vacancy)
        RETURN type(r) AS feedback_type,
               v.id AS vacancy_id,
               v.title AS vacancy_title,
               r.timestamp AS timestamp
        ORDER BY r.timestamp DESC
        LIMIT $limit
        """

        return self.db.execute_query(query, {'user_id': user_id, 'limit': limit})