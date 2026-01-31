import aiohttp
import asyncio
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

    def search_vacancies(self, text: str = "", area: int = 1, per_page: int = 20, page: int = 0) -> List[Dict[str, Any]]:
        """Синхронный поиск вакансий (без деталей) — для кэширования"""
        params = {
            'text': text,
            'area': area,
            'per_page': min(per_page, 100),
            'page': page,
            'search_field': 'name'
        }

        try:
            logger.info(f"Searching vacancies: {text}, page {page}")
            response = self.session.get(f"{self.base_url}/vacancies", params=params, timeout=10)

            if response.status_code != 200:
                logger.error(f"HTTP error {response.status_code}: {response.text[:200]}")
                return []

            data = response.json()
            items = data.get('items', [])
            logger.info(f"Found {len(items)} vacancies on page {page}")

            return items

        except Exception as e:
            logger.error(f"Error in search_vacancies: {e}")
            return []

    async def _fetch_vacancy_detail(self, session: aiohttp.ClientSession, vacancy_id: str) -> Optional[Dict]:
        """Асинхронное получение одной вакансии"""
        url = f"{self.base_url}/vacancies/{vacancy_id}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json'
        }

        for attempt in range(3):
            try:
                async with session.get(url, headers=headers, timeout=10) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    elif resp.status == 429:
                        delay = 2 ** attempt
                        logger.warning(f"Rate limit. Retrying in {delay}s...")
                        await asyncio.sleep(delay)
                        continue
                    else:
                        logger.warning(f"HTTP {resp.status} for {vacancy_id}")
                        return None
            except Exception as e:
                logger.error(f"Error fetching {vacancy_id}: {e}")
                await asyncio.sleep(1)
                continue
        return None

    async def fetch_and_parse_vacancies_async(self, vacancy_ids: List[str]) -> List[Optional[Vacancy]]:
        """Асинхронная загрузка и парсинг полных данных вакансий"""
        connector = aiohttp.TCPConnector(limit=20, limit_per_host=5)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [self._fetch_vacancy_detail(session, vid) for vid in vacancy_ids]
            results = await asyncio.gather(*tasks, return_exceptions=False)

        # Парсим каждую вакансию
        parsed = []
        for data in results:
            if data:
                model = self.parse_to_model(data)
                if model:
                    parsed.append(model)
        return parsed

    def _clean_html(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r'<[^>]+>', '', text)
        text = text.replace('&nbsp;', ' ').replace('&quot;', '"').replace('&amp;', '&')
        text = text.replace('&lt;', '<').replace('&gt;', '>').replace('&#39;', "'")
        return text.strip()

    def _safe_get(self, data: Dict, key: str, default: Any = None) -> Any:
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
        if not hh_data:
            return None

        try:
            vacancy_id = self._safe_get(hh_data, 'id')
            if not vacancy_id:
                return None

            skills = []
            key_skills = self._safe_get(hh_data, 'key_skills', [])
            if isinstance(key_skills, list):
                for skill in key_skills:
                    skill_name = self._safe_get(skill, 'name')
                    if skill_name and isinstance(skill_name, str):
                        skills.append(skill_name[:100])

            company_name = self._safe_get(hh_data, 'employer.name', '')
            location_name = self._safe_get(hh_data, 'area.name', '')

            salary_data = self._safe_get(hh_data, 'salary', {})
            salary_from = self._safe_get(salary_data, 'from')
            salary_to = self._safe_get(salary_data, 'to')
            currency = self._safe_get(salary_data, 'currency')

            experience = self._safe_get(hh_data, 'experience.name', '')
            employment = self._safe_get(hh_data, 'employment.name', '')

            published_at = None
            published_str = self._safe_get(hh_data, 'published_at')
            if published_str:
                try:
                    if 'T' in published_str:
                        published_at = datetime.fromisoformat(published_str.replace('Z', '+00:00'))
                    else:
                        published_at = datetime.fromisoformat(published_str)
                except (ValueError, TypeError) as e:
                    logger.debug(f"Could not parse date {published_str}: {e}")

            title = self._clean_html(self._safe_get(hh_data, 'name', 'Без названия'))[:200]
            description = self._clean_html(self._safe_get(hh_data, 'description', ''))[:5000]

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

    def fetch_and_parse_vacancies_sync(self, search_query: str, limit: int = 20) -> List[Vacancy]:
        """Синхронная версия (для обратной совместимости)"""
        pages = (limit + 100 - 1) // 100
        all_items = []

        for page in range(pages):
            remaining = limit - len(all_items)
            if remaining <= 0:
                break
            per_page = min(100, remaining)
            items = self.search_vacancies(text=search_query, per_page=per_page, page=page)
            all_items.extend(items)
            if len(items) < per_page:
                break

        # Извлекаем ID и загружаем детали последовательно (медленно, но надёжно)
        vacancies = []
        for item in all_items[:limit]:
            vac_id = item.get('id')
            if not vac_id:
                continue
            detail = self.session.get(f"{self.base_url}/vacancies/{vac_id}", timeout=10).json()
            model = self.parse_to_model(detail)
            if model:
                vacancies.append(model)
        return vacancies

    def test_connection(self) -> bool:
        try:
            response = self.session.get(f"{self.base_url}/vacancies", params={'text': 'Python', 'per_page': 1}, timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False