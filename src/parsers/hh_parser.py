import requests
from datetime import datetime
import logging
import time
import re
from typing import List, Optional, Dict, Any
from src.database.models import Vacancy

logger = logging.getLogger(__name__)


class HHParser:
    def __init__(self):
        self.base_url = "https://api.hh.ru"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7'
        })

    def search_vacancies(self, text: str = "", area: int = 1, per_page: int = 50, page: int = 0) -> List[
        Dict[str, Any]]:
        """Поиск вакансий через API HH.ru"""
        params = {
            'text': text,
            'area': area,
            'per_page': min(per_page, 100),
            'page': page,
            'search_field': 'name'
        }

        try:
            logger.info(f"Searching vacancies: {text}")
            response = self.session.get(f"{self.base_url}/vacancies", params=params, timeout=10)

            if response.status_code != 200:
                logger.error(f"HTTP error {response.status_code}: {response.text[:200]}")
                return []

            data = response.json()
            items = data.get('items', [])
            logger.info(f"Found {len(items)} vacancies")

            vacancies_data = []
            for i, item in enumerate(items):
                vacancy_id = item.get('id')
                if not vacancy_id:
                    continue

                # Получаем детали
                vacancy_detail = self._get_vacancy_details_safe(vacancy_id)
                if vacancy_detail:
                    vacancies_data.append(vacancy_detail)

                # Пауза чтобы не нагружать API
                time.sleep(0.2)

            return vacancies_data

        except requests.exceptions.RequestException as e:
            logger.error(f"Network error: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in search_vacancies: {e}")
            return []

    def _get_vacancy_details_safe(self, vacancy_id: str) -> Optional[Dict[str, Any]]:
        """Безопасное получение деталей вакансии"""
        try:
            response = self.session.get(f"{self.base_url}/vacancies/{vacancy_id}", timeout=10)

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                logger.debug(f"Vacancy {vacancy_id} not found")
            elif response.status_code == 429:
                logger.warning("Rate limit exceeded")
                time.sleep(2)
                return self._get_vacancy_details_safe(vacancy_id)
            else:
                logger.warning(f"HTTP {response.status_code} for vacancy {vacancy_id}")

        except requests.exceptions.RequestException as e:
            logger.error(f"Network error for vacancy {vacancy_id}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error for vacancy {vacancy_id}: {e}")

        return None

    def _clean_html(self, text: str) -> str:
        """Очистка HTML тегов из текста"""
        if not text:
            return ""

        # Удаляем HTML теги
        text = re.sub(r'<[^>]+>', '', text)
        # Заменяем HTML entities
        text = text.replace('&nbsp;', ' ').replace('&quot;', '"').replace('&amp;', '&')
        text = text.replace('&lt;', '<').replace('&gt;', '>').replace('&#39;', "'")

        return text.strip()

    def _safe_get(self, data: Dict, key: str, default: Any = None) -> Any:
        """Безопасное получение значения из словаря"""
        if not data:
            return default

        keys = key.split('.')
        current = data

        for k in keys:
            if isinstance(current, dict):
                current = current.get(k)
                if current is None:
                    return default
            else:
                return default

        return current

    def parse_to_model(self, hh_data: Optional[Dict[str, Any]]) -> Optional[Vacancy]:
        """Преобразование данных HH.ru в модель Vacancy"""
        if not hh_data:
            logger.debug("No data to parse")
            return None

        try:
            # Безопасное извлечение данных
            vacancy_id = self._safe_get(hh_data, 'id')
            if not vacancy_id:
                logger.warning("No ID in vacancy data")
                return None

            # Навыки
            skills = []
            key_skills = self._safe_get(hh_data, 'key_skills', [])
            if isinstance(key_skills, list):
                for skill in key_skills:
                    skill_name = self._safe_get(skill, 'name')
                    if skill_name and isinstance(skill_name, str):
                        skills.append(skill_name[:100])  # Ограничиваем длину

            # Компания
            company_name = self._safe_get(hh_data, 'employer.name', '')

            # Локация
            location_name = self._safe_get(hh_data, 'area.name', '')

            # Зарплата
            salary_data = self._safe_get(hh_data, 'salary', {})
            salary_from = self._safe_get(salary_data, 'from')
            salary_to = self._safe_get(salary_data, 'to')
            currency = self._safe_get(salary_data, 'currency')

            # Опыт и занятость
            experience = self._safe_get(hh_data, 'experience.name', '')
            employment = self._safe_get(hh_data, 'employment.name', '')

            # Дата публикации
            published_at = None
            published_str = self._safe_get(hh_data, 'published_at')
            if published_str:
                try:
                    # Преобразуем дату
                    if 'T' in published_str:
                        published_at = datetime.fromisoformat(published_str.replace('Z', '+00:00'))
                    else:
                        published_at = datetime.fromisoformat(published_str)
                except (ValueError, TypeError) as e:
                    logger.debug(f"Could not parse date {published_str}: {e}")

            # Название и описание
            title = self._safe_get(hh_data, 'name', 'Без названия')
            description = self._safe_get(hh_data, 'description', '')

            # Очистка описания
            title = self._clean_html(title)[:200]
            description = self._clean_html(description)[:5000]

            # Создаем вакансию
            return Vacancy(
                id=f"hh_{vacancy_id}",
                external_id=str(vacancy_id),
                title=title,
                description=description,
                salary_from=float(salary_from) if salary_from else None,
                salary_to=float(salary_to) if salary_to else None,
                currency=currency,
                experience=experience,
                employment=employment,
                skills=skills,
                company_name=company_name[:100],
                location_name=location_name[:100],
                published_at=published_at
            )

        except Exception as e:
            logger.error(f"Error parsing vacancy data: {e}")
            return None

    def fetch_and_parse_vacancies(self, search_query: str = "Python", limit: int = 20) -> List[Vacancy]:
        """Получение и парсинг вакансий"""
        logger.info(f"Starting fetch_and_parse_vacancies: '{search_query}', limit={limit}")

        try:
            # Получаем сырые данные
            vacancies_data = self.search_vacancies(text=search_query, per_page=limit)

            if not vacancies_data:
                logger.warning(f"No vacancies found for query: {search_query}")
                return []

            # Парсим каждую вакансию
            vacancies = []
            for i, data in enumerate(vacancies_data):
                try:
                    vacancy = self.parse_to_model(data)
                    if vacancy:
                        vacancies.append(vacancy)
                        logger.debug(f"Parsed vacancy {i + 1}/{len(vacancies_data)}: {vacancy.title}")
                    else:
                        logger.warning(f"Failed to parse vacancy {i + 1}")
                except Exception as e:
                    logger.error(f"Error parsing vacancy {i + 1}: {e}")

            logger.info(f"Successfully parsed {len(vacancies)}/{len(vacancies_data)} vacancies")
            return vacancies

        except Exception as e:
            logger.error(f"Error in fetch_and_parse_vacancies: {e}")
            return []

    def test_connection(self) -> bool:
        """Тестирование подключения к API HH.ru"""
        try:
            response = self.session.get(f"{self.base_url}/vacancies",
                                        params={'text': 'test', 'per_page': 1},
                                        timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False