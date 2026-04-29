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

    def save_vacancy(self, vacancy) -> bool:
        """Сохранить вакансию (синхронно для Streamlit)"""
        try:
            # Определяем hh_id (пытаемся получить из разных атрибутов)
            hh_id = None
            if hasattr(vacancy, 'hh_id'):
                hh_id = vacancy.hh_id
            elif hasattr(vacancy, 'id'):
                hh_id = vacancy.id
            elif hasattr(vacancy, 'external_id'):
                hh_id = vacancy.external_id
            elif isinstance(vacancy, dict):
                hh_id = vacancy.get('hh_id') or vacancy.get('id') or vacancy.get('external_id')

            if not hh_id:
                logger.error("Vacancy has no id")
                return False

            # Проверяем существование
            existing = self.neo4j.execute_query(
                "MATCH (v:Vacancy {hh_id: $hh_id}) RETURN v",
                {'hh_id': str(hh_id)}
            )

            if existing:
                logger.info(f"Vacancy {hh_id} already exists")
                return True

            # Создаем словарь для сохранения
            vacancy_dict = self._vacancy_to_dict(vacancy, hh_id)

            if not vacancy_dict:
                logger.error(f"Failed to convert vacancy to dict: {hh_id}")
                return False

            # Пробуем создать эмбеддинг
            text_for_embedding = f"{vacancy_dict.get('title', '')} {vacancy_dict.get('description', '')}"
            embedding = self._get_embedding_sync(text_for_embedding)
            if embedding:
                vacancy_dict['embedding'] = embedding

            # Сохраняем в Neo4j
            result = self.neo4j.execute_query("""
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
            """, vacancy_dict)

            if result:
                logger.info(f"Saved vacancy {hh_id}")
                return True
            else:
                logger.warning(f"Failed to save vacancy {hh_id}")
                return False

        except Exception as e:
            logger.error(f"Error saving vacancy: {e}")
            return self._save_vacancy_minimal(vacancy)

    def _vacancy_to_dict(self, vacancy, hh_id: str = None) -> Dict[str, Any]:
        """Преобразовать вакансию в словарь для Neo4j"""

        # Получаем все атрибуты объекта
        attrs = {}

        # Пробуем получить атрибуты разными способами
        if hasattr(vacancy, '__dict__'):
            attrs = vacancy.__dict__.copy()
        elif hasattr(vacancy, '__dataclass_fields__'):
            # Для dataclass
            for field in vacancy.__dataclass_fields__:
                attrs[field] = getattr(vacancy, field, None)
        elif isinstance(vacancy, dict):
            attrs = vacancy.copy()
        else:
            try:
                attrs = vars(vacancy).copy()
            except:
                pass

        # Если не нашли hh_id, используем переданный
        if not hh_id:
            hh_id = attrs.get('hh_id') or attrs.get('id') or attrs.get('external_id')

        if not hh_id:
            logger.error("Cannot determine hh_id for vacancy")
            return None

        # Создаем словарь с ВСЕМИ необходимыми полями
        result = {
            'id': str(uuid.uuid4()),
            'hh_id': str(hh_id),
            'title': attrs.get('title', '')[:200],
            'description': attrs.get('description', '')[:5000] if attrs.get('description') else '',
            'company_name': attrs.get('company_name') or attrs.get('company', 'Не указана'),
            'location_name': attrs.get('location_name') or attrs.get('location', 'Не указана'),
            'salary_from': attrs.get('salary_from'),
            'salary_to': attrs.get('salary_to'),
            'salary_currency': attrs.get('salary_currency') or attrs.get('currency', 'RUB'),
            'skills': attrs.get('skills', [])[:20],
            'experience': attrs.get('experience', 'Не указан'),
            'employment': attrs.get('employment', 'Не указан'),
            'schedule': attrs.get('schedule', 'Не указан'),
            'url': attrs.get('url', f"https://hh.ru/vacancy/{hh_id}"),
            'published_at': self._format_date(attrs.get('published_at')),
            'embedding': attrs.get('embedding')
        }

        # Очищаем None значения
        for key, value in result.items():
            if value is None and key not in ['embedding', 'salary_from', 'salary_to']:
                if key == 'schedule':
                    result[key] = 'Не указан'
                elif key == 'url':
                    result[key] = f"https://hh.ru/vacancy/{hh_id}"
                elif key == 'company_name':
                    result[key] = 'Не указана'
                elif key == 'location_name':
                    result[key] = 'Не указана'

        logger.debug(f"Converted vacancy {hh_id} to dict with fields: {list(result.keys())}")
        return result

    def _format_date(self, date_value) -> str:
        """Форматирование даты для Neo4j"""
        if not date_value:
            return datetime.now().isoformat()

        if isinstance(date_value, datetime):
            return date_value.isoformat()

        if isinstance(date_value, str):
            return date_value

        return datetime.now().isoformat()

    def _save_vacancy_minimal(self, vacancy) -> bool:
        """Сохранить вакансию только с базовыми полями"""
        try:
            # Определяем hh_id
            hh_id = None
            if hasattr(vacancy, 'hh_id'):
                hh_id = vacancy.hh_id
            elif hasattr(vacancy, 'id'):
                hh_id = vacancy.id
            elif hasattr(vacancy, 'external_id'):
                hh_id = vacancy.external_id
            elif isinstance(vacancy, dict):
                hh_id = vacancy.get('hh_id') or vacancy.get('id') or vacancy.get('external_id')

            if not hh_id:
                logger.error("Cannot determine hh_id for minimal save")
                return False

            # Получаем базовые атрибуты
            title = ''
            if hasattr(vacancy, 'title'):
                title = vacancy.title[:200]
            elif isinstance(vacancy, dict):
                title = vacancy.get('title', '')[:200]

            description = ''
            if hasattr(vacancy, 'description'):
                description = vacancy.description[:500] if vacancy.description else ''
            elif isinstance(vacancy, dict):
                description = vacancy.get('description', '')[:500]

            company_name = 'Не указана'
            if hasattr(vacancy, 'company_name'):
                company_name = vacancy.company_name[:100] if vacancy.company_name else 'Не указана'
            elif isinstance(vacancy, dict):
                company_name = vacancy.get('company_name', 'Не указана')[:100]

            location_name = 'Не указана'
            if hasattr(vacancy, 'location_name'):
                location_name = vacancy.location_name[:100] if vacancy.location_name else 'Не указана'
            elif isinstance(vacancy, dict):
                location_name = vacancy.get('location_name', 'Не указана')[:100]

            # Зарплата
            salary_from = None
            salary_to = None
            salary_currency = 'RUB'

            if hasattr(vacancy, 'salary_from'):
                salary_from = vacancy.salary_from
                salary_to = vacancy.salary_to if hasattr(vacancy, 'salary_to') else None
                salary_currency = vacancy.currency if hasattr(vacancy, 'currency') else 'RUB'
            elif isinstance(vacancy, dict):
                salary_from = vacancy.get('salary_from')
                salary_to = vacancy.get('salary_to')
                salary_currency = vacancy.get('currency', 'RUB')

            # Навыки
            skills = []
            if hasattr(vacancy, 'skills'):
                skills = vacancy.skills[:10] if vacancy.skills else []
            elif isinstance(vacancy, dict):
                skills = vacancy.get('skills', [])[:10]

            # Опыт и занятость
            experience = 'Не указан'
            if hasattr(vacancy, 'experience'):
                experience = vacancy.experience or 'Не указан'
            elif isinstance(vacancy, dict):
                experience = vacancy.get('experience', 'Не указан')

            employment = 'Не указан'
            if hasattr(vacancy, 'employment'):
                employment = vacancy.employment or 'Не указан'
            elif isinstance(vacancy, dict):
                employment = vacancy.get('employment', 'Не указан')

            # URL
            url = f"https://hh.ru/vacancy/{hh_id}"
            if hasattr(vacancy, 'url') and vacancy.url:
                url = vacancy.url
            elif isinstance(vacancy, dict) and vacancy.get('url'):
                url = vacancy.get('url')

            vacancy_dict = {
                'id': str(uuid.uuid4()),
                'hh_id': str(hh_id),
                'title': title,
                'description': description,
                'company_name': company_name,
                'location_name': location_name,
                'salary_from': salary_from,
                'salary_to': salary_to,
                'salary_currency': salary_currency,
                'skills': skills,
                'experience': experience,
                'employment': employment,
                'schedule': 'Не указан',
                'url': url,
                'published_at': datetime.now().isoformat()
            }

            result = self.neo4j.execute_query("""
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
            """, vacancy_dict)

            if result:
                logger.info(f"Saved vacancy {hh_id} with minimal data")
                return True
            return False

        except Exception as e:
            logger.error(f"Minimal save failed: {e}")
            return False

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

    # Добавьте в VacancyService

    def get_recommendations(self, user_id: str, top_n: int = 10,
                            content_weight: float = 0.33,
                            graph_weight: float = 0.34,
                            semantic_weight: float = 0.33) -> List:
        """Получение гибридных рекомендаций"""
        from src.database.models import RecommendationScore

        # Запрос к Neo4j
        query = """
        MATCH (u:User {id: $user_id})-[:HAS_SKILL]->(us:Skill)
        MATCH (v:Vacancy)-[:REQUIRES]->(vs:Skill)
        WHERE vs.name IN collect(us.name)

        WITH u, v, collect(DISTINCT us.name) as user_skills,
             collect(DISTINCT vs.name) as vacancy_skills

        // Content score (совпадение навыков)
        WITH u, v, user_skills, vacancy_skills,
             [skill IN user_skills WHERE skill IN vacancy_skills] as matched

        WITH u, v, 
             size(matched) * 1.0 / size(user_skills + vacancy_skills) as content_score

        // Graph score (компании)
        OPTIONAL MATCH (u)-[:WORKED_IN]->(c:Company)
        OPTIONAL MATCH (v)-[:FROM_COMPANY]->(vc:Company)

        WITH u, v, content_score,
             CASE WHEN vc.name IN collect(c.name) THEN 1.0 ELSE 0.0 END as graph_score

        // Total score
        RETURN v.id as vacancy_id,
               content_score,
               graph_score,
               (content_score * $content_weight + 
                graph_score * $graph_weight) as total_score
        ORDER BY total_score DESC
        LIMIT $top_n
        """

        results = self.neo4j.execute_query(query, {
            'user_id': user_id,
            'top_n': top_n,
            'content_weight': content_weight,
            'graph_weight': graph_weight
        })

        recommendations = []
        for r in results:
            vacancy = self.get_vacancy_by_id(r['vacancy_id'])
            if vacancy:
                recommendations.append(RecommendationScore(
                    vacancy=vacancy,
                    content_score=r['content_score'],
                    graph_score=r['graph_score'],
                    semantic_score=0,  # можно добавить позже
                    total_score=r['total_score']
                ))

        return recommendations