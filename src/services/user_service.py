# src/services/user_service.py
from typing import List, Dict, Any, Optional
import logging
from src.database.neo4j_client import Neo4jClient
from datetime import datetime

logger = logging.getLogger(__name__)


class UserService:
    def __init__(self, neo4j_client: Neo4jClient):
        self.neo4j = neo4j_client

    def create_user(self, user_id: str, name: str = None, email: str = None) -> bool:
        """Создать пользователя"""
        try:
            result = self.neo4j.execute_query("""
                MERGE (u:User {id: $user_id})
                SET u.username = $name,
                    u.email = $email,
                    u.created_at = datetime($created_at),
                    u.updated_at = datetime($updated_at)
                RETURN u
            """, {
                'user_id': user_id,
                'name': name or f"User_{user_id[:8]}",
                'email': email,
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            })

            if result:
                logger.info(f"User created/updated: {user_id}")
                return True
            return False

        except Exception as e:
            logger.error(f"Error creating user {user_id}: {e}")
            return False

    def create_or_update_user(self, user) -> bool:
        """
        Создать или обновить пользователя из объекта User

        Args:
            user: Объект User с атрибутами id, username, resume_text, skills
        """
        try:
            # Создаём пользователя
            result = self.neo4j.execute_query("""
                MERGE (u:User {id: $user_id})
                SET u.username = $username,
                    u.resume_text = $resume_text,
                    u.skills = $skills,
                    u.updated_at = datetime($updated_at)
                RETURN u
            """, {
                'user_id': user.id,
                'username': user.username,
                'resume_text': getattr(user, 'resume_text', ''),
                'skills': getattr(user, 'skills', []),
                'updated_at': datetime.now().isoformat()
            })

            if result:
                # Добавляем навыки как отдельные узлы
                if hasattr(user, 'skills') and user.skills:
                    for skill_name in user.skills:
                        self.neo4j.execute_query("""
                            MATCH (u:User {id: $user_id})
                            MERGE (s:Skill {name: $skill_name})
                            MERGE (u)-[:HAS_SKILL]->(s)
                        """, {
                            'user_id': user.id,
                            'skill_name': skill_name
                        })

                logger.info(f"User created/updated: {user.username}")
                return True
            return False

        except Exception as e:
            logger.error(f"Error creating/updating user {user.id}: {e}")
            return False

    def get_user_by_id(self, user_id: str):
        """
        Получить пользователя по ID и вернуть объект User

        Args:
            user_id: ID пользователя

        Returns:
            Объект User или None
        """
        try:
            result = self.neo4j.execute_query(
                """MATCH (u:User {id: $user_id})
                   OPTIONAL MATCH (u)-[:HAS_SKILL]->(s:Skill)
                   RETURN u, COLLECT(s.name) as skills""",
                {'user_id': user_id}
            )

            if result and result[0].get('u'):
                user_data = result[0]['u']
                skills = result[0].get('skills', [])

                from src.database.models import User
                return User(
                    id=user_data.get('id'),
                    username=user_data.get('username', ''),
                    resume_text=user_data.get('resume_text', ''),
                    skills=[s for s in skills if s is not None]
                )
            return None

        except Exception as e:
            logger.error(f"Error getting user {user_id}: {e}")
            return None

    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Получить пользователя как словарь"""
        try:
            result = self.neo4j.execute_query(
                "MATCH (u:User {id: $user_id}) RETURN u",
                {'user_id': user_id}
            )

            if result and result[0].get('u'):
                user_data = result[0]['u']
                return {
                    'id': user_data.get('id'),
                    'name': user_data.get('name'),
                    'username': user_data.get('username'),
                    'email': user_data.get('email'),
                    'resume_text': user_data.get('resume_text'),
                    'skills': user_data.get('skills', []),
                    'created_at': user_data.get('created_at'),
                    'updated_at': user_data.get('updated_at')
                }
            return None

        except Exception as e:
            logger.error(f"Error getting user {user_id}: {e}")
            return None

    def record_view(self, user_id: str, vacancy_id: str) -> bool:
        """Записать просмотр вакансии"""
        try:
            result = self.neo4j.execute_query("""
                MATCH (u:User {id: $user_id})
                MATCH (v:Vacancy {id: $vacancy_id})
                MERGE (u)-[r:VIEWED]->(v)
                SET r.viewed_at = datetime($viewed_at)
                RETURN r
            """, {
                'user_id': user_id,
                'vacancy_id': vacancy_id,
                'viewed_at': datetime.now().isoformat()
            })

            if result:
                logger.info(f"Recorded view: user {user_id} -> vacancy {vacancy_id}")
                return True
            return False

        except Exception as e:
            logger.error(f"Error recording view: {e}")
            return False

    def add_favorite(self, user_id: str, vacancy_id: str) -> bool:
        """Добавить в избранное"""
        try:
            result = self.neo4j.execute_query("""
                MATCH (u:User {id: $user_id})
                MATCH (v:Vacancy {id: $vacancy_id})
                MERGE (u)-[r:FAVORITED]->(v)
                SET r.added_at = datetime($added_at)
                RETURN r
            """, {
                'user_id': user_id,
                'vacancy_id': vacancy_id,
                'added_at': datetime.now().isoformat()
            })

            if result:
                logger.info(f"Added favorite: user {user_id} -> vacancy {vacancy_id}")
                return True
            return False

        except Exception as e:
            logger.error(f"Error adding favorite: {e}")
            return False

    def remove_favorite(self, user_id: str, vacancy_id: str) -> bool:
        """Удалить из избранного"""
        try:
            result = self.neo4j.execute_query("""
                MATCH (u:User {id: $user_id})-[r:FAVORITED]->(v:Vacancy {id: $vacancy_id})
                DELETE r
                RETURN COUNT(r) as deleted
            """, {
                'user_id': user_id,
                'vacancy_id': vacancy_id
            })

            if result and result[0].get('deleted', 0) > 0:
                logger.info(f"Removed favorite: user {user_id} -> vacancy {vacancy_id}")
                return True
            return False

        except Exception as e:
            logger.error(f"Error removing favorite: {e}")
            return False

    def get_user_history(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Получить историю пользователя"""
        try:
            history = self.neo4j.execute_query("""
                MATCH (u:User {id: $user_id})-[r:VIEWED]->(v:Vacancy)
                RETURN v.id as vacancy_id,
                       v.title as title,
                       v.company_name as company_name,
                       r.viewed_at as viewed_at,
                       'viewed' as interaction_type
                UNION
                MATCH (u:User {id: $user_id})-[r:FAVORITED]->(v:Vacancy)
                RETURN v.id as vacancy_id,
                       v.title as title,
                       v.company_name as company_name,
                       r.added_at as viewed_at,
                       'favorited' as interaction_type
                ORDER BY viewed_at DESC
                LIMIT $limit
            """, {'user_id': user_id, 'limit': limit})

            return history or []

        except Exception as e:
            logger.error(f"Error getting user history: {e}")
            return []