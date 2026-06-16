# src/services/feedback_service.py
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class FeedbackService:
    """
    Сервис обратной связи

    ПРАВИЛЬНАЯ ВЕРСИЯ: __init__ с опциональным аргументом
    """

    def __init__(self, neo4j_client=None):  # ← Опционально! По умолчанию None
        """
        Args:
            neo4j_client: Клиент Neo4j (опционально, можно передать в init())
        """
        self.neo4j = neo4j_client
        self._initialized = False

    def init(self, neo4j_client=None):  # ← Принимает 1 аргумент (+ self)
        """
        Инициализация сервиса

        Args:
            neo4j_client: Клиент Neo4j
        """
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

    def add_feedback(
            self,
            user_id: str,
            vacancy_id: str,
            rating: int,
            comment: str = None
    ) -> bool:
        """Добавить обратную связь"""
        self._check_initialized()

        try:
            result = self.neo4j.execute_query("""
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

            if result:
                logger.info(f"Feedback added: user {user_id} rated vacancy {vacancy_id} with {rating}")
                return True
            return False

        except Exception as e:
            logger.error(f"Error adding feedback: {e}")
            return False

    def add_like(self, user_id: str, vacancy_id: str) -> bool:
        """Добавить лайк"""
        return self.add_feedback(user_id, vacancy_id, rating=1)

    def add_dislike(self, user_id: str, vacancy_id: str) -> bool:
        """Добавить дизлайк"""
        return self.add_feedback(user_id, vacancy_id, rating=0)

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

            if result:
                return result[0]
            return {'average_rating': 0, 'ratings_count': 0}

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
            """, {
                'user_id': user_id,
                'limit': limit
            })

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