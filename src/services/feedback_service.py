# src/services/feedback_service.py
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
from src.database.models import FeedbackType

logger = logging.getLogger(__name__)


class FeedbackService:
    """
    Сервис обратной связи

    ПРАВИЛЬНАЯ ВЕРСИЯ: __init__ с опциональным аргументом
    """

    def __init__(self, neo4j_client=None):
        """
        Args:
            neo4j_client: Клиент Neo4j (опционально, можно передать в init())
        """
        self.neo4j = neo4j_client
        self._initialized = False

    def init(self, neo4j_client=None):
        """Инициализация сервиса"""
        if neo4j_client is not None:
            self.neo4j = neo4j_client

        if self.neo4j is None:
            raise ValueError("FeedbackService: neo4j_client обязателен")

        self._initialized = True
        logger.info("FeedbackService инициализирован")

    def _check_initialized(self):
        """Проверка инициализации"""
        if not self._initialized:
            raise RuntimeError("FeedbackService не инициализирован. Вызовите init()")

    def _execute_feedback_query(self, query: str, params: Dict) -> bool:
        """Выполнить запрос обратной связи с обработкой ошибок"""
        self._check_initialized()
        try:
            result = self.neo4j.execute_query(query, params)
            return bool(result)
        except Exception as e:
            logger.error(f"Error executing feedback query: {e}")
            return False

    # === Базовые операции ===

    def add_feedback(self, user_id: str, vacancy_id: str, rating: int, comment: str = None) -> bool:
        """Добавить обратную связь"""
        return self._execute_feedback_query("""
            MATCH (u:User {id: $user_id})
            MATCH (v:Vacancy {id: $vacancy_id})
            MERGE (u)-[r:RATED]->(v)
            SET r.rating = $rating,
                r.comment = $comment,
                r.created_at = datetime($created_at)
            RETURN r
        """, {
            'user_id': user_id,
            'vacancy_id': vacancy_id,
            'rating': rating,
            'comment': comment,
            'created_at': datetime.now().isoformat()
        })

    def record_view(self, user_id: str, vacancy_id: str) -> bool:
        """Записать просмотр"""
        return self._execute_feedback_query(
            "MATCH (u:User {id: $user_id}) MATCH (v:Vacancy {id: $vacancy_id}) MERGE (u)-[r:VIEWED]->(v) RETURN r",
            {'user_id': user_id, 'vacancy_id': vacancy_id}
        )

    def add_like(self, user_id: str, vacancy_id: str) -> bool:
        """Добавить лайк (rating=1)"""
        return self.add_feedback(user_id, vacancy_id, rating=1)

    def add_dislike(self, user_id: str, vacancy_id: str) -> bool:
        """Добавить дизлайк (rating=0)"""
        return self.add_feedback(user_id, vacancy_id, rating=0)

    def record_apply(self, user_id: str, vacancy_id: str) -> bool:
        """Записать отклик (как оценку 5)"""
        return self._execute_feedback_query(
            "MATCH (u:User {id: $user_id}) MATCH (v:Vacancy {id: $vacancy_id}) "
            "MERGE (u)-[r:RATED]->(v) "
            "ON CREATE SET r.rating = 5, r.created_at = datetime() "
            "RETURN r",
            {'user_id': user_id, 'vacancy_id': vacancy_id}
        )

    def record_feedback(self, feedback) -> bool:
        """Записать обратную связь (совместимость с UserFeedback)"""
        self._check_initialized()
        
        try:
            feedback_type = getattr(feedback, 'feedback_type', None)
            user_id = getattr(feedback, 'user_id', None)
            vacancy_id = getattr(feedback, 'vacancy_id', None)
            
            if not all([user_id, vacancy_id, feedback_type]):
                logger.warning(f"Invalid feedback object: {feedback}")
                return False
            
            # Маппинг типов на методы
            type_mapping = {
                (FeedbackType.LIKE, 'LIKED'): lambda: self.add_like(user_id, vacancy_id),
                (FeedbackType.DISLIKE, 'DISLIKED'): lambda: self.add_dislike(user_id, vacancy_id),
                (FeedbackType.VIEW, 'VIEWED'): lambda: self.record_view(user_id, vacancy_id),
                (FeedbackType.APPLY, 'APPLIED'): lambda: self.record_apply(user_id, vacancy_id),
            }
            
            for (enum_val, str_val), method in type_mapping.items():
                if feedback_type == enum_val or feedback_type == str_val:
                    return method()
            
            return False
        except Exception as e:
            logger.error(f"Error recording feedback: {e}")
            return False

    # === Аналитика ===

    def get_vacancy_rating(self, vacancy_id: str) -> Dict[str, Any]:
        """Получить средний рейтинг вакансии"""
        self._check_initialized()
        try:
            result = self.neo4j.execute_query("""
                MATCH (v:Vacancy {id: $vacancy_id})<-[r:RATED]-(:User)
                RETURN 
                    AVG(r.rating) as average_rating,
                    COUNT(r) as ratings_count,
                    MIN(r.rating) as min_rating,
                    MAX(r.rating) as max_rating
            """, {'vacancy_id': vacancy_id})
            return result[0] if result else {'average_rating': 0, 'ratings_count': 0}
        except Exception as e:
            logger.error(f"Error getting vacancy rating: {e}")
            return {'average_rating': 0, 'ratings_count': 0}

    def get_user_feedback(self, user_id: str, limit: int = 50) -> List[Dict]:
        """Получить историю оценок пользователя"""
        self._check_initialized()
        try:
            result = self.neo4j.execute_query("""
                MATCH (u:User {id: $user_id})-[r:RATED]->(v:Vacancy)
                RETURN v.id as vacancy_id,
                       v.title as title,
                       r.rating as rating,
                       r.comment as comment,
                       r.created_at as created_at
                ORDER BY r.created_at DESC
                LIMIT $limit
            """, {'user_id': user_id, 'limit': limit})
            return result if result else []
        except Exception as e:
            logger.error(f"Error getting user feedback: {e}")
            return []

    def get_feedback_stats(self, user_id: str) -> Dict:
        """Получить статистику оценок пользователя"""
        self._check_initialized()
        try:
            result = self.neo4j.execute_query("""
                MATCH (u:User {id: $user_id})-[r:RATED]->(:Vacancy)
                RETURN 
                    COUNT(r) as total_ratings,
                    AVG(r.rating) as avg_rating,
                    SUM(CASE WHEN r.rating >= 4 THEN 1 ELSE 0 END) as likes,
                    SUM(CASE WHEN r.rating <= 2 THEN 1 ELSE 0 END) as dislikes
            """, {'user_id': user_id})
            return result[0] if result else {}
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}

    def get_user_feedback_history(self, user_id: str, limit: int = 20) -> List[Dict]:
        """Получить историю действий пользователя"""
        self._check_initialized()
        try:
            result = self.neo4j.execute_query(
                "MATCH (u:User {id: $user_id})-[r:VIEWED|RATED]->(v:Vacancy) "
                "RETURN v.id as vacancy_id, v.title as vacancy_title, "
                "CASE WHEN type(r) = 'VIEWED' THEN 'VIEWED' "
                "     WHEN r.rating >= 4 THEN 'LIKED' "
                "     ELSE 'DISLIKED' "
                "END as feedback_type, "
                "r.created_at as timestamp "
                "ORDER BY timestamp DESC LIMIT $limit",
                {'user_id': user_id, 'limit': limit}
            )
            return result if result else []
        except Exception as e:
            logger.error(f"Error getting feedback history: {e}")
            return []
