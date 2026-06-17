from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime
import logging
from src.database.neo4j_client import Neo4jClient
from src.ai.embeddings import EmbeddingService

logger = logging.getLogger(__name__)


class VacancyService:
    def __init__(
            self,
            neo4j_client: Neo4jClient,
            embedding_service: EmbeddingService
    ):
        self.neo4j = neo4j_client
        self.embeddings = embedding_service

    def _extract_vacancy_data(self, vacancy) -> Dict[str, Any]:
        """Извлечь данные вакансии независимо от типа (объект, dict, dataclass)"""
        attrs = {}
        
        if hasattr(vacancy, '__dict__'):
            attrs = vacancy.__dict__.copy()
        elif hasattr(vacancy, '__dataclass_fields__'):
            for field in vacancy.__dataclass_fields__:
                attrs[field] = getattr(vacancy, field, None)
        elif isinstance(vacancy, dict):
            attrs = vacancy.copy()
        else:
            try:
                attrs = vars(vacancy).copy()
            except:
                pass
        
        hh_id = attrs.get('hh_id') or attrs.get('id') or attrs.get('external_id')
        if not hh_id:
            logger.error("Vacancy has no id")
            return {}
        
        def safe_get(key, default=None):
            return attrs.get(key, default)
        
        def safe_slice(attr, max_len, default=''):
            val = safe_get(attr, default)
            return val[:max_len] if val else default
        
        return {
            'hh_id': str(hh_id),
            'title': safe_slice('title', 200, ''),
            'description': safe_slice('description', 5000, ''),
            'company_name': safe_get('company_name') or safe_get('company') or 'Не указана',
            'location_name': safe_get('location_name') or safe_get('location') or 'Не указана',
            'salary_from': safe_get('salary_from'),
            'salary_to': safe_get('salary_to'),
            'salary_currency': safe_get('salary_currency') or safe_get('currency', 'RUB'),
            'skills': safe_get('skills', [])[:20],
            'experience': safe_get('experience', 'Не указан'),
            'employment': safe_get('employment', 'Не указан'),
            'schedule': safe_get('schedule', 'Не указан'),
            'url': safe_get('url', f"https://hh.ru/vacancy/{hh_id}"),
            'published_at': safe_get('published_at'),
            'embedding': safe_get('embedding')
        }

    def save_vacancy(self, vacancy) -> bool:
        """Сохранить вакансию (синхронно для Streamlit)"""
        try:
            vacancy_data = self._extract_vacancy_data(vacancy)
            if not vacancy_data:
                return False

            hh_id = vacancy_data['hh_id']

            # Проверяем существование
            existing = self.neo4j.execute_query(
                "MATCH (v:Vacancy {hh_id: $hh_id}) RETURN v",
                {'hh_id': hh_id}
            )

            if existing:
                logger.info(f"Vacancy {hh_id} already exists")
                return True

            # Пробуем создать эмбеддинг
            text_for_embedding = f"{vacancy_data['title']} {vacancy_data['description']}"
            embedding = self._get_embedding_sync(text_for_embedding)
            if embedding:
                vacancy_data['embedding'] = embedding

            # Применяем значения по умолчанию
            vacancy_data = self._apply_defaults(vacancy_data, hh_id)

            # Сохраняем в Neo4j
            result = self.neo4j.execute_query(self._get_create_vacancy_query(), vacancy_data)

            if result:
                logger.info(f"Saved vacancy {hh_id}")
                return True
            else:
                logger.warning(f"Failed to save vacancy {hh_id}")
                return False

        except Exception as e:
            logger.error(f"Error saving vacancy: {e}")
            return self._save_vacancy_minimal(vacancy)

    def _save_vacancy_minimal(self, vacancy) -> bool:
        """Сохранить вакансию только с базовыми полями"""
        try:
            vacancy_data = self._extract_vacancy_data(vacancy)
            if not vacancy_data:
                return False

            hh_id = vacancy_data['hh_id']
            vacancy_data['title'] = vacancy_data['title'][:200]
            vacancy_data['description'] = vacancy_data['description'][:500]
            vacancy_data['company_name'] = vacancy_data['company_name'][:100]
            vacancy_data['location_name'] = vacancy_data['location_name'][:100]
            vacancy_data['skills'] = vacancy_data['skills'][:10]
            vacancy_data['experience'] = vacancy_data['experience'] or 'Не указан'
            vacancy_data['employment'] = vacancy_data['employment'] or 'Не указан'
            vacancy_data['schedule'] = 'Не указан'
            vacancy_data['published_at'] = datetime.now().isoformat()

            result = self.neo4j.execute_query(self._get_create_vacancy_query(False), vacancy_data)

            if result:
                logger.info(f"Saved vacancy {hh_id} with minimal data")
                return True
            return False

        except Exception as e:
            logger.error(f"Minimal save failed: {e}")
            return False

    def _apply_defaults(self, data: Dict[str, Any], hh_id: str) -> Dict[str, Any]:
        """Применить значения по умолчанию для None значений"""
        defaults = {
            'schedule': 'Не указан',
            'url': f"https://hh.ru/vacancy/{hh_id}",
            'company_name': 'Не указана',
            'location_name': 'Не указана',
            'experience': 'Не указан',
            'employment': 'Не указан'
        }
        for key, default in defaults.items():
            if data.get(key) is None:
                data[key] = default
        return data

    def _get_create_vacancy_query(self, with_embedding: bool = True) -> str:
        """Вернуть SQL-запрос для создания вакансии"""
        if with_embedding:
            return """
                CREATE (v:Vacancy {
                    id: $id,
                    hh_id: $hh_id,
                    title: $title,
                    description: $description,
                    company_name: $company_name,
                    location_name: $location_name,
                    salary_from: $salary_from,
                    salary_to: $salary_to,
                    salary_currency: $salary_currency,
                    skills: $skills,
                    experience: $experience,
                    employment: $employment,
                    schedule: $schedule,
                    url: $url,
                    published_at: datetime($published_at),
                    embedding: $embedding
                })
                RETURN v.id
            """
        return """
            CREATE (v:Vacancy {
                id: $id,
                hh_id: $hh_id,
                title: $title,
                description: $description,
                company_name: $company_name,
                location_name: $location_name,
                salary_from: $salary_from,
                salary_to: $salary_to,
                salary_currency: $salary_currency,
                skills: $skills,
                experience: $experience,
                employment: $employment,
                schedule: $schedule,
                url: $url,
                published_at: datetime($published_at)
            })
            RETURN v.id
        """

    def _vacancy_to_dict(self, vacancy, hh_id: str = None) -> Dict[str, Any]:
        """Преобразовать вакансию в словарь для Neo4j (для совместимости)"""
        data = self._extract_vacancy_data(vacancy)
        if not data:
            return None
        return self._apply_defaults(data, data.get('hh_id') or str(hh_id) or 'unknown')

    def _format_date(self, date_value) -> str:
        """Форматирование даты для Neo4j"""
        if not date_value:
            return datetime.now().isoformat()

        if isinstance(date_value, datetime):
            return date_value.isoformat()

        if isinstance(date_value, str):
            return date_value

        return datetime.now().isoformat()

    def _get_embedding_sync(self, text: str) -> Optional[List[float]]:
        """Синхронное получение эмбеддинга"""
        if not text or len(text.strip()) < 10:
            return None

        try:
            # Используем правильный метод get_embedding_sync
            return self.embeddings.get_embedding_sync(text)
        except Exception as e:
            logger.warning(f"Could not generate embedding: {e}")
            return None

    def get_recommendations(self, user_id: str, top_n: int = 10,
                            content_weight: float = 0.33,
                            graph_weight: float = 0.34,
                            semantic_weight: float = 0.33) -> List:
        """Получение гибридных рекомендаций"""
        from src.database.models import RecommendationScore

        # Получаем навыки пользователя (они хранятся как свойство skills в узле User)
        user = self.neo4j.execute_query(
            "MATCH (u:User {id: $user_id}) RETURN u.skills as skills",
            {'user_id': user_id}
        )
        
        if not user or not user[0].get('skills'):
            logger.warning(f"User {user_id} has no skills")
            return []

        user_skills = user[0]['skills']
        logger.info(f"User {user_id} has {len(user_skills)} skills: {user_skills}")

        # Запрос к Neo4j - ищем вакансии, где хотя бы один навык совпадает
        query = """
        // Ищем вакансии с совпадающими навыками
        MATCH (v:Vacancy)
        WHERE ANY(skill IN $user_skills WHERE skill IN v.skills)
        AND v.title IS NOT NULL
        AND v.title <> ''
        AND v.title <> 'Без названия'
        
        // Content score (доля совпадающих навыков)
        WITH v, 
             [skill IN $user_skills WHERE skill IN v.skills] as matched_skills,
             size($user_skills) as user_skills_count
        
        WITH v,
             matched_skills,
             size(matched_skills) * 1.0 / user_skills_count as content_score
        
        // Graph score (по компании - если пользователь работал в такой компании ранее)
        // Проверяем, есть ли у пользователя в истории вакансии от этой же компании
        OPTIONAL MATCH (u:User {id: $user_id})-[:RATED|VIEWED|LIKED]->(v2:Vacancy {company_name: v.company_name})
        WITH v, content_score,
             CASE WHEN count(v2) > 0 THEN 1.0 ELSE 0.0 END as graph_score
        
        // Compute final score (без дизлайков)
        WITH v, content_score, graph_score,
             content_score * $content_weight + graph_score * $graph_weight as total_score
        
        // Exclude vacancies user has explicitly disliked
        OPTIONAL MATCH (u:User {id: $user_id})-[d:DISLIKED]->(v)
        WITH v, content_score, graph_score, total_score,
             CASE WHEN count(d) > 0 THEN 0.0 ELSE total_score END as final_score
        
        RETURN v.id as vacancy_id,
               content_score,
               graph_score,
               final_score as total_score
        ORDER BY final_score DESC
        LIMIT $top_n
        """

        results = self.neo4j.execute_query(query, {
            'user_id': user_id,
            'top_n': top_n,
            'content_weight': content_weight,
            'graph_weight': graph_weight,
            'user_skills': user_skills
        })
        
        logger.info(f"Found {len(results if results else [])} vacancy recommendations")
        
        # Обработка None или пустого результата
        if not results:
            return []

        recommendations = []
        for r in results:
            vacancy_obj = self.get_vacancy_object_by_id(r['vacancy_id'])
            if vacancy_obj:
                title = vacancy_obj.title
                if title and title.strip() and title != 'Без названия':
                    recommendations.append(RecommendationScore(
                        vacancy=vacancy_obj,
                        content_score=r['content_score'],
                        graph_score=r['graph_score'],
                        semantic_score=0,  # можно добавить позже
                        total_score=r['total_score']
                    ))
                else:
                    logger.warning(f"Filtered vacancy with bad title: {title}")
            else:
                logger.warning(f"Vacancy {r['vacancy_id']} not found or has no title")

        logger.info(f"Returned {len(recommendations)} recommendations after filtering")
        return recommendations if recommendations else []

    def get_vacancy_by_id(self, vacancy_id: str, as_object: bool = False) -> Optional[Dict[str, Any]]:
        """Получить вакансию по ID (id узла в Neo4j)"""
        try:
            result = self.neo4j.execute_query(
                "MATCH (v:Vacancy {id: $vacancy_id}) RETURN v",
                {'vacancy_id': vacancy_id}
            )
            
            if result and len(result) > 0:
                vacancy_node = result[0]['v']
                vacancy_dict = dict(vacancy_node)
                
                # Фильтруем некорректные заголовки
                title = vacancy_dict.get('title', '')
                if not title or title.strip() == '' or title == 'Без названия':
                    logger.warning(f"get_vacancy_by_id: Filtering vacancy {vacancy_id} with bad title: {title}")
                    return None
                
                # Если нужен объект Vacancy, преобразуем
                if as_object:
                    return self._dict_to_vacancy(vacancy_dict)
                
                return vacancy_dict
            
            logger.warning(f"get_vacancy_by_id: Vacancy {vacancy_id} not found")
            return None
        except Exception as e:
            logger.error(f"Error getting vacancy by id {vacancy_id}: {e}")
            return None
    
    def get_vacancy_object_by_id(self, vacancy_id: str) -> Optional['Vacancy']:
        """Получить вакансию по ID и вернуть объект Vacancy"""
        return self.get_vacancy_by_id(vacancy_id, as_object=True)
    
    def _dict_to_vacancy(self, data: Dict[str, Any]) -> Optional['Vacancy']:
        """Преобразовать словарь в объект Vacancy"""
        try:
            from src.database.models import Vacancy
            return Vacancy(
                id=data.get('id', ''),
                external_id=str(data.get('hh_id', '')),
                title=data.get('title', ''),
                description=data.get('description', ''),
                company_name=data.get('company_name', ''),
                location_name=data.get('location_name', ''),
                salary_from=float(data.get('salary_from')) if data.get('salary_from') else None,
                salary_to=float(data.get('salary_to')) if data.get('salary_to') else None,
                currency=data.get('salary_currency', 'RUB'),
                experience=data.get('experience', ''),
                employment=data.get('employment', ''),
                skills=data.get('skills', []),
                published_at=data.get('published_at')
            )
        except Exception as e:
            logger.error(f"Error converting dict to Vacancy: {e}")
            return None
