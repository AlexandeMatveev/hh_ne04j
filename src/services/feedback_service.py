# src/services/feedback_service.py
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
from src.database.models import FeedbackType

logger = logging.getLogger(__name__)


class FeedbackService:
    """
    Сервис обратной связи
    """

    def __init__(self, neo4j_client=None):
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

    def add_dislike(self, user_id: str, vacancy_id: str) -> bool:
        """Добавить дизлайк (rating=1) и создать прямую связь DISLIKED"""
        self._check_initialized()
        try:
            result = self.add_feedback(user_id, vacancy_id, rating=1)
            self.neo4j.execute_query(
                "MATCH (u:User {id: $user_id}) MATCH (v:Vacancy {id: $vacancy_id}) "
                "MERGE (u)-[r:DISLIKED]->(v) SET r.created_at = datetime() RETURN r",
                {'user_id': user_id, 'vacancy_id': vacancy_id}
            )
            return result
        except Exception as e:
            logger.error(f"Error adding dislike: {e}")
            return False

    def add_like(self, user_id: str, vacancy_id: str) -> bool:
        """Добавить лайк (rating=5) и создать прямую связь LIKED"""
        self._check_initialized()
        try:
            result = self.add_feedback(user_id, vacancy_id, rating=5)
            self.neo4j.execute_query(
                "MATCH (u:User {id: $user_id}) MATCH (v:Vacancy {id: $vacancy_id}) "
                "MERGE (u)-[r:LIKED]->(v) SET r.created_at = datetime() RETURN r",
                {'user_id': user_id, 'vacancy_id': vacancy_id}
            )
            return result
        except Exception as e:
            logger.error(f"Error adding like: {e}")
            return False

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

    def get_vacancy_rating(self, vacancy_id: str) -> Dict[str, Any]:
        """Получить средний рейтинг вакансии"""
        self._check_initialized()
        try:
            result = self.neo4j.execute_query("""
                MATCH (v:Vacancy {id: $vacancy_id})<-[r:LIKED|DISLIKED|RATED]-(:User)
                RETURN 
                    AVG(CASE WHEN type(r) = 'LIKED' OR r.rating >= 4 THEN 5 
                             WHEN type(r) = 'DISLIKED' OR r.rating <= 2 THEN 1 
                             ELSE r.rating END) as average_rating,
                    SUM(CASE WHEN type(r) = 'LIKED' OR r.rating >= 4 THEN 1 ELSE 0 END) as likes,
                    SUM(CASE WHEN type(r) = 'DISLIKED' OR r.rating <= 2 THEN 1 ELSE 0 END) as dislikes,
                    COUNT(r) as ratings_count
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
                MATCH (u:User {id: $user_id})-[r:LIKED|DISLIKED|RATED]->(v:Vacancy)
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
                MATCH (u:User {id: $user_id})-[r:LIKED|DISLIKED|RATED]->(:Vacancy)
                RETURN 
                    COUNT(r) as total_ratings,
                    AVG(CASE WHEN type(r) = 'LIKED' OR r.rating >= 4 THEN 1 ELSE 0 END) as like_rate,
                    SUM(CASE WHEN type(r) = 'LIKED' OR r.rating >= 4 THEN 1 ELSE 0 END) as likes,
                    SUM(CASE WHEN type(r) = 'DISLIKED' OR r.rating <= 2 THEN 1 ELSE 0 END) as dislikes
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
                "MATCH (u:User {id: $user_id})-[r:VIEWED|LIKED|DISLIKED|RATED]->(v:Vacancy) "
                "RETURN v.id as vacancy_id, v.title as vacancy_title, "
                "CASE WHEN type(r) = 'VIEWED' THEN 'VIEWED' "
                "     WHEN type(r) = 'LIKED' THEN 'LIKED' "
                "     WHEN type(r) = 'DISLIKED' THEN 'DISLIKED' "
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

    def get_user_likes(self, user_id: str, limit: int = 20) -> List[Dict]:
        """Получить список понравившихся вакансий"""
        self._check_initialized()
        try:
            result = self.neo4j.execute_query(
                "MATCH (u:User {id: $user_id})-[r:LIKED]->(v:Vacancy) "
                "RETURN v.id as vacancy_id, v.title as title, v.company_name as company_name, "
                "v.salary_from as salary_from, v.salary_to as salary_to, v.skills as skills, "
                "r.created_at as liked_at "
                "ORDER BY r.created_at DESC LIMIT $limit",
                {'user_id': user_id, 'limit': limit}
            )
            return result if result else []
        except Exception as e:
            logger.error(f"Error getting user likes: {e}")
            return []

    def get_user_dislikes(self, user_id: str, limit: int = 20) -> List[Dict]:
        """Получить список непонравившихся вакансий"""
        self._check_initialized()
        try:
            result = self.neo4j.execute_query(
                "MATCH (u:User {id: $user_id})-[r:DISLIKED]->(v:Vacancy) "
                "RETURN v.id as vacancy_id, v.title as title, v.company_name as company_name, "
                "r.created_at as disliked_at "
                "ORDER BY r.created_at DESC LIMIT $limit",
                {'user_id': user_id, 'limit': limit}
            )
            return result if result else []
        except Exception as e:
            logger.error(f"Error getting user dislikes: {e}")
            return []

    def remove_like(self, user_id: str, vacancy_id: str) -> bool:
        """Удалить лайк"""
        self._check_initialized()
        try:
            result = self.neo4j.execute_query(
                "MATCH (u:User {id: $user_id})-[r:LIKED]->(v:Vacancy {id: $vacancy_id}) "
                "DELETE r "
                "RETURN COUNT(r) as deleted",
                {'user_id': user_id, 'vacancy_id': vacancy_id}
            )
            return result and result[0].get('deleted', 0) > 0
        except Exception as e:
            logger.error(f"Error removing like: {e}")
            return False

    def remove_dislike(self, user_id: str, vacancy_id: str) -> bool:
        """Удалить дизлайк"""
        self._check_initialized()
        try:
            result = self.neo4j.execute_query(
                "MATCH (u:User {id: $user_id})-[r:DISLIKED]->(v:Vacancy {id: $vacancy_id}) "
                "DELETE r "
                "RETURN COUNT(r) as deleted",
                {'user_id': user_id, 'vacancy_id': vacancy_id}
            )
            return result and result[0].get('deleted', 0) > 0
        except Exception as e:
            logger.error(f"Error removing dislike: {e}")
            return False
