import os
from dotenv import load_dotenv
import requests
import aiohttp
import asyncio
from datetime import datetime
import logging
import time
import re
from typing import List, Optional, Dict, Any, Union
from src.database.models import Vacancy

# Загружаем переменные из .env файла
load_dotenv()

logger = logging.getLogger(__name__)


class HHParser:
    def __init__(self):
        self.base_url = "https://api.hh.ru"
        self.session = requests.Session()

        # Базовые заголовки
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7'
        }

        # Загружаем токен из .env
        access_token = os.getenv("HH_ACCESS_TOKEN")

        if access_token and access_token.strip():
            headers['Authorization'] = f'Bearer {access_token}'
            logger.info(f"✅ Access token loaded from .env (token: {access_token[:20]}...)")
            self.token_configured = True
        else:
            logger.warning("⚠️ No access token found in .env file")
            self.token_configured = False

        self.session.headers.update(headers)
        self._last_request_time = 0

    def _rate_limit(self):
        """Ограничение частоты запросов"""
        elapsed = time.time() - self._last_request_time
        delay = 0.17 if self.token_configured else 0.5
        if elapsed < delay:
            time.sleep(delay - elapsed)
        self._last_request_time = time.time()

    async def _rate_limit_async(self):
        """Асинхронное ограничение частоты запросов"""
        elapsed = time.time() - self._last_request_time
        delay = 0.17 if self.token_configured else 0.5
        if elapsed < delay:
            await asyncio.sleep(delay - elapsed)
        self._last_request_time = time.time()

    def search_vacancies(self, text: str = "", area: int = 1, per_page: int = 50, page: int = 0) -> List[
        Dict[str, Any]]:
        """Поиск вакансий через API HH.ru (синхронный)"""
        params = {
            'text': text,
            'area': area,
            'per_page': min(per_page, 100),
            'page': page,
            'search_field': 'name'
        }

        self._rate_limit()

        try:
            logger.info(f"🔍 Searching vacancies: {text}")
            response = self.session.get(f"{self.base_url}/vacancies", params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                items = data.get('items', [])
                total_found = data.get('found', 0)
                logger.info(f"✅ Found {total_found} vacancies total, processing {len(items)} items")

                vacancies_data = []
                for i, item in enumerate(items[:per_page]):
                    vacancy_id = item.get('id')
                    if not vacancy_id:
                        continue

                    logger.debug(f"📥 Fetching details for vacancy {i + 1}/{len(items)}: {vacancy_id}")
                    vacancy_detail = self._get_vacancy_details_safe(vacancy_id)
                    if vacancy_detail:
                        vacancies_data.append(vacancy_detail)

                    time.sleep(0.2)

                return vacancies_data

            elif response.status_code == 403:
                logger.error(f"❌ HTTP 403 - Access denied.")
                return []
            else:
                logger.error(f"❌ HTTP error {response.status_code}: {response.text[:200]}")
                return []

        except Exception as e:
            logger.error(f"❌ Unexpected error in search_vacancies: {e}")
            return []

    async def search_vacancies_async(self, text: str = "", area: int = 1, per_page: int = 50, page: int = 0) -> List[
        Dict[str, Any]]:
        """Асинхронный поиск вакансий через API HH.ru"""
        if isinstance(text, list):
            logger.warning(f"search_vacancies_async called with list, fetching by IDs directly")
            return await self._fetch_vacancies_by_ids_async(text[:per_page])

        params = {
            'text': text,
            'area': area,
            'per_page': min(per_page, 100),
            'page': page,
            'search_field': 'name'
        }

        await self._rate_limit_async()

        try:
            logger.info(f"🔍 Async searching vacancies: {text}")

            async with aiohttp.ClientSession() as session:
                headers = self.session.headers.copy()
                async with session.get(f"{self.base_url}/vacancies", headers=headers, params=params,
                                       timeout=10) as response:

                    if response.status == 200:
                        data = await response.json()
                        items = data.get('items', [])
                        total_found = data.get('found', 0)
                        logger.info(f"✅ Found {total_found} vacancies total")
                        return items
                    elif response.status == 403:
                        logger.error(f"❌ HTTP 403 - Access denied.")
                        return []
                    else:
                        logger.error(f"❌ HTTP error {response.status}")
                        return []

        except Exception as e:
            logger.error(f"❌ Error in async search: {e}")
            return []

    async def _fetch_vacancies_by_ids_async(self, vacancy_ids: List[str]) -> List[Dict[str, Any]]:
        """Асинхронное получение вакансий по ID"""
        logger.info(f"📥 Fetching {len(vacancy_ids)} vacancies by IDs")

        async with aiohttp.ClientSession() as session:
            tasks = []
            for vacancy_id in vacancy_ids:
                task = self._get_vacancy_details_async(session, vacancy_id)
                tasks.append(task)

            results = await asyncio.gather(*tasks)
            vacancies = [v for v in results if v is not None]

            logger.info(f"✅ Retrieved {len(vacancies)} vacancies by IDs")
            return vacancies

    def _get_vacancy_details_safe(self, vacancy_id: str) -> Optional[Dict[str, Any]]:
        """Безопасное получение деталей вакансии (синхронный)"""
        self._rate_limit()

        try:
            response = self.session.get(f"{self.base_url}/vacancies/{vacancy_id}", timeout=10)

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                logger.debug(f"Vacancy {vacancy_id} not found")
            elif response.status_code == 429:
                logger.warning("⚠️ Rate limit exceeded, waiting 2 seconds...")
                time.sleep(2)
                return self._get_vacancy_details_safe(vacancy_id)
            else:
                logger.warning(f"⚠️ HTTP {response.status_code} for vacancy {vacancy_id}")

        except Exception as e:
            logger.error(f"❌ Error for vacancy {vacancy_id}: {e}")

        return None

    async def _get_vacancy_details_async(self, session: aiohttp.ClientSession, vacancy_id: str) -> Optional[
        Dict[str, Any]]:
        """Асинхронное получение деталей вакансии"""
        await self._rate_limit_async()

        try:
            headers = self.session.headers.copy()
            async with session.get(f"{self.base_url}/vacancies/{vacancy_id}", headers=headers, timeout=10) as response:

                if response.status == 200:
                    return await response.json()
                elif response.status == 404:
                    logger.debug(f"Vacancy {vacancy_id} not found")
                elif response.status == 429:
                    logger.warning("⚠️ Rate limit exceeded, waiting...")
                    await asyncio.sleep(2)
                    return await self._get_vacancy_details_async(session, vacancy_id)
                else:
                    logger.warning(f"⚠️ HTTP {response.status} for vacancy {vacancy_id}")

        except Exception as e:
            logger.error(f"❌ Error for vacancy {vacancy_id}: {e}")

        return None

    def _clean_html(self, text: str) -> str:
        """Очистка HTML тегов из текста"""
        if not text:
            return ""
        text = re.sub(r'<[^>]+>', '', text)
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
        """
        Преобразование данных HH.ru в модель Vacancy
        Адаптировано под вашу модель (поля: id, external_id, title, description,
        company_name, location_name, salary_from, salary_to, currency,
        experience, employment, skills, published_at, embedding)
        """
        if not hh_data:
            logger.debug("No data to parse")
            return None

        try:
            # ID вакансии
            vacancy_id = self._safe_get(hh_data, 'id')
            if not vacancy_id:
                logger.warning("No ID in vacancy data")
                return None

            # Навыки (skills)
            skills = []
            key_skills = self._safe_get(hh_data, 'key_skills', [])
            if isinstance(key_skills, list):
                for skill in key_skills:
                    skill_name = self._safe_get(skill, 'name')
                    if skill_name and isinstance(skill_name, str):
                        skills.append(skill_name[:100])

            # Компания
            company_name = self._safe_get(hh_data, 'employer.name', '')
            company_name = company_name[:100] if company_name else 'Не указана'

            # Локация
            location_name = self._safe_get(hh_data, 'area.name', '')
            location_name = location_name[:100] if location_name else 'Не указана'

            # Зарплата
            salary_data = self._safe_get(hh_data, 'salary', {})
            salary_from = self._safe_get(salary_data, 'from')
            salary_to = self._safe_get(salary_data, 'to')
            currency = self._safe_get(salary_data, 'currency', 'RUB')

            # Опыт и занятость
            experience = self._safe_get(hh_data, 'experience.name', '')
            employment = self._safe_get(hh_data, 'employment.name', '')

            # Дата публикации
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
                    published_at = datetime.now()
            else:
                published_at = datetime.now()

            # Название и описание
            title = self._safe_get(hh_data, 'name', 'Без названия')
            description = self._safe_get(hh_data, 'description', '')

            # Очистка текста
            title = self._clean_html(title)[:200]
            description = self._clean_html(description)[:5000]

            # Создаем вакансию в соответствии с вашей моделью
            vacancy = Vacancy(
                id=f"hh_{vacancy_id}",  # Уникальный ID в системе
                external_id=str(vacancy_id),  # ID из HH.ru
                title=title,
                description=description,
                company_name=company_name,
                location_name=location_name,
                salary_from=float(salary_from) if salary_from else None,
                salary_to=float(salary_to) if salary_to else None,
                currency=currency,
                experience=experience,
                employment=employment,
                skills=skills,
                published_at=published_at
                # embedding будет добавлен позже сервисом эмбеддингов
            )

            logger.debug(f"✅ Successfully parsed vacancy: {vacancy.title}")
            return vacancy

        except Exception as e:
            logger.error(f"Error parsing vacancy data: {e}")
            logger.debug(f"Problematic data: {hh_data.get('id', 'unknown') if hh_data else 'None'}")
            return None

    def fetch_and_parse_vacancies(self, search_query: Union[str, List[str]] = "Python", limit: int = 20) -> List[
        Vacancy]:
        """Синхронное получение и парсинг вакансий"""
        logger.info(f"🚀 Starting sync fetch")

        try:
            # Если передан список ID
            if isinstance(search_query, list):
                logger.info(f"Fetching {len(search_query)} vacancies by IDs")
                vacancies = []
                for vacancy_id in search_query[:limit]:
                    details = self._get_vacancy_details_safe(vacancy_id)
                    if details:
                        vacancy = self.parse_to_model(details)
                        if vacancy:
                            vacancies.append(vacancy)
                    time.sleep(0.2)
                logger.info(f"✅ Retrieved {len(vacancies)} vacancies")
                return vacancies

            # Обычный поиск по строке
            vacancies_data = self.search_vacancies(text=search_query, per_page=limit)

            if not vacancies_data:
                logger.warning(f"❌ No vacancies found")
                return []

            vacancies = []
            for i, data in enumerate(vacancies_data):
                try:
                    vacancy = self.parse_to_model(data)
                    if vacancy:
                        vacancies.append(vacancy)
                except Exception as e:
                    logger.error(f"❌ Error parsing vacancy {i + 1}: {e}")

            logger.info(f"✨ Successfully parsed {len(vacancies)}/{len(vacancies_data)} vacancies")
            return vacancies

        except Exception as e:
            logger.error(f"❌ Error in fetch_and_parse_vacancies: {e}")
            return []

    async def fetch_and_parse_vacancies_async(self, search_query: Union[str, List[str]] = "Python", limit: int = 20) -> \
    List[Vacancy]:
        """
        Асинхронное получение и парсинг вакансий

        Args:
            search_query: Строка поиска или список ID вакансий
            limit: Максимальное количество вакансий
        """
        logger.info(f"🚀 Starting async fetch")

        try:
            # Если передан список ID
            if isinstance(search_query, list):
                logger.info(f"Fetching {len(search_query[:limit])} vacancies by IDs asynchronously")
                items = await self._fetch_vacancies_by_ids_async(search_query[:limit])

                vacancies = []
                for item in items:
                    vacancy = self.parse_to_model(item)
                    if vacancy:
                        vacancies.append(vacancy)

                logger.info(f"✨ Successfully parsed {len(vacancies)} vacancies")
                return vacancies

            # Обычный поиск по строке
            items = await self.search_vacancies_async(text=search_query, per_page=limit)

            if not items:
                logger.warning(f"❌ No vacancies found")
                return []

            # Асинхронная загрузка деталей
            async with aiohttp.ClientSession() as session:
                tasks = []
                for item in items[:limit]:
                    vacancy_id = item.get('id')
                    if vacancy_id:
                        task = self._get_vacancy_details_async(session, vacancy_id)
                        tasks.append(task)

                logger.info(f"📥 Loading details for {len(tasks)} vacancies asynchronously")
                details_list = await asyncio.gather(*tasks)

                # Парсим результаты
                vacancies = []
                for details in details_list:
                    if details:
                        vacancy = self.parse_to_model(details)
                        if vacancy:
                            vacancies.append(vacancy)

                logger.info(f"✨ Successfully parsed {len(vacancies)} vacancies")
                return vacancies

        except Exception as e:
            logger.error(f"❌ Error in async fetch: {e}")
            return []

    def test_connection(self) -> bool:
        """Тестирование подключения к API HH.ru"""
        try:
            response = self.session.get(f"{self.base_url}/vacancies",
                                        params={'text': 'python', 'per_page': 1},
                                        timeout=5)

            if response.status_code == 200:
                logger.info("✅ Connection successful")
                return True
            else:
                logger.warning(f"⚠️ Connection returned {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"❌ Connection test failed: {e}")
            return False


# ===================== ИСПОЛЬЗОВАНИЕ =====================
async def main():
    parser = HHParser()

    if parser.test_connection():
        print("\n" + "=" * 60)
        print("SEARCHING VACANCIES")
        print("=" * 60)

        # Поиск вакансий
        vacancies = await parser.fetch_and_parse_vacancies_async(
            search_query="Python разработчик",
            limit=5
        )

        print(f"\n📊 Found {len(vacancies)} vacancies:\n")
        for i, v in enumerate(vacancies, 1):
            print(f"{i}. {v.title}")
            print(f"   ID: {v.id}")
            print(f"   External ID: {v.external_id}")
            print(f"   Company: {v.company_name}")
            print(f"   Location: {v.location_name}")
            if v.salary_from or v.salary_to:
                print(f"   Salary: {v.salary_from or ''} - {v.salary_to or ''} {v.currency}")
            print(f"   Experience: {v.experience}")
            print(f"   Employment: {v.employment}")
            if v.skills:
                print(f"   Skills: {', '.join(v.skills[:5])}")
            print()
    else:
        print("❌ Connection failed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())