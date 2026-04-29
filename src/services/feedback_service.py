# src/services/feedback_service.py
from typing import List, Dict, Any, Optional
import logging
from src.database.neo4j_client import Neo4jClient
from datetime import datetime

logger = logging.getLogger(__name__)


class FeedbackService:
    def __init__(self, neo4j_client: Neo4jClient):
        self.neo4j = neo4j_client

    def add_feedback(self, user_id: str, vacancy_id: str, rating: int, comment: str = None) -> bool:
        """Добавить обратную связь"""
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

    def get_vacancy_rating(self, vacancy_id: str) -> Dict[str, Any]:
        """Получить средний рейтинг вакансии"""
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