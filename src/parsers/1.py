import requests
import time
import json
import os
import re
from datetime import datetime
from typing import List, Dict, Optional, Any


class HHParser:
    """
    Парсер для сбора вакансий с hh.ru через официальное API.
    """
    BASE_URL = "https://api.hh.ru/vacancies"
    TOKEN_URL = "https://hh.ru/oauth/token"
    TOKEN_FILE = "hh_token.json"

    def __init__(self, client_id=None, client_secret=None, proxy=None):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json",
            "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
            "Connection": "keep-alive",
        }

        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token = None
        self.proxies = {"http": proxy, "https": proxy} if proxy else None
        if proxy:
            print(f"🌐 Использую прокси: {proxy}")

    def _get_saved_token(self):
        """Загружает сохраненный токен из файла"""
        if os.path.exists(self.TOKEN_FILE):
            try:
                with open(self.TOKEN_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    saved_time = datetime.fromisoformat(data.get("saved_at", "2000-01-01"))
                    if (datetime.now() - saved_time).days < 13:
                        return data["access_token"]
            except:
                pass
        return None

    def _save_token(self, token):
        """Сохраняет токен в файл"""
        with open(self.TOKEN_FILE, "w", encoding="utf-8") as f:
            json.dump({
                "access_token": token,
                "saved_at": datetime.now().isoformat()
            }, f)

    def _delete_token_file(self):
        """Удаляет файл с токеном"""
        if os.path.exists(self.TOKEN_FILE):
            os.remove(self.TOKEN_FILE)
            print("🗑️ Старый токен удален")

    def get_client_credentials_token(self, force_new=False):
        """
        Получить токен приложения (client credentials flow).
        """
        if not self.client_id or not self.client_secret:
            print("⚠️ Client ID или Client Secret не указаны")
            return None

        # Проверяем сохраненный токен (если не требуем новый)
        if not force_new:
            saved_token = self._get_saved_token()
            if saved_token:
                print("📂 Использую сохраненный токен")
                self.access_token = saved_token
                self.headers["Authorization"] = f"Bearer {self.access_token}"
                return saved_token

        print("🔐 Получение нового токена...")

        data = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret
        }

        try:
            response = requests.post(self.TOKEN_URL, data=data, proxies=self.proxies, timeout=30)
        except Exception as e:
            print(f"❌ Ошибка соединения: {e}")
            return None

        if response.status_code != 200:
            print(f"❌ Ошибка {response.status_code}: {response.text}")

            # Если ошибка - удаляем старый токен
            self._delete_token_file()

            if response.status_code == 403:
                print("\n❌ ВАШ IP ЗАБЛОКИРОВАН!")
                print("📌 Решения:")
                print("   1. Включите VPN на компьютере")
                print("   2. Используйте прокси: parser = HHParser(client_id, client_secret, proxy='http://ip:port')")
                print("   3. Подождите 24 часа (блокировка временная)")

            return None

        token_data = response.json()
        self.access_token = token_data.get("access_token")

        if not self.access_token:
            print("❌ Токен не найден в ответе")
            return None

        self.headers["Authorization"] = f"Bearer {self.access_token}"
        self._save_token(self.access_token)
        print(f"✅ Токен получен и сохранен!")

        return self.access_token

    def search_vacancies(self, text, area=113, per_page=100, max_pages=5, **kwargs):
        """
        Ищет вакансии по заданным параметрам.
        Теперь загружает ПОЛНОЕ описание каждой вакансии.
        """
        all_vacancies = []

        params = {
            "text": text,
            "area": area,
            "per_page": per_page,
            "page": 0,
            **kwargs
        }

        # Убираем невалидные параметры
        params.pop("only_with_salary", None)

        for page in range(max_pages):
            params["page"] = page
            print(f"📄 Загрузка страницы {page + 1}...")

            try:
                response = requests.get(
                    self.BASE_URL,
                    headers=self.headers,
                    params=params,
                    proxies=self.proxies,
                    timeout=30
                )

                print(f"Статус ответа: {response.status_code}")

                if response.status_code == 403:
                    print("\n❌ ДОСТУП ЗАПРЕЩЕН! Ваш IP заблокирован.")
                    print("📌 Нужно сменить IP через VPN или прокси!")
                    break

                if response.status_code == 401:
                    print("⚠️ Токен истек, пробую получить новый...")
                    self._delete_token_file()
                    if self.get_client_credentials_token(force_new=True):
                        response = requests.get(
                            self.BASE_URL,
                            headers=self.headers,
                            params=params,
                            proxies=self.proxies,
                            timeout=30
                        )
                    else:
                        break

                if response.status_code != 200:
                    print(f"Тело ответа: {response.text}")
                    response.raise_for_status()

                data = response.json()
                items = data.get("items", [])

                if not items:
                    print("Вакансии не найдены на этой странице.")
                    break

                # Загружаем детальную информацию для каждой вакансии
                for i, item in enumerate(items):
                    vacancy_id = item.get("id")
                    print(f"  🔍 Загрузка деталей вакансии {i + 1}/{len(items)} (ID: {vacancy_id})...")

                    # Получаем полное описание вакансии
                    full_vacancy = self.get_vacancy_details(vacancy_id)

                    if full_vacancy:
                        all_vacancies.append(full_vacancy)
                    else:
                        # Если не удалось загрузить детали, используем базовую информацию
                        basic_info = self._parse_vacancy_item(item)
                        all_vacancies.append(basic_info)

                    # Небольшая задержка между запросами
                    time.sleep(0.3)

                total_pages = data.get("pages", 0)
                print(f"Страница {page + 1} из {total_pages}, найдено вакансий: {len(items)}")

                if page >= total_pages - 1:
                    print("✅ Достигнут конец списка вакансий.")
                    break

                time.sleep(0.3)

            except requests.exceptions.RequestException as e:
                print(f"❌ Ошибка при запросе к API: {e}")
                break

        print(f"📊 Всего собрано вакансий: {len(all_vacancies)}")
        return all_vacancies

    def get_vacancy_details(self, vacancy_id: str) -> Optional[Dict[str, Any]]:
        """
        Получение ДЕТАЛЬНОЙ информации о вакансии (включая полное описание).
        """
        try:
            response = requests.get(
                f"{self.BASE_URL}/{vacancy_id}",
                headers=self.headers,
                proxies=self.proxies,
                timeout=30
            )

            if response.status_code != 200:
                print(f"  ⚠️ Не удалось загрузить детали вакансии {vacancy_id}: {response.status_code}")
                return None

            data = response.json()

            # Извлекаем навыки
            skills = [skill["name"] for skill in data.get("key_skills", [])]

            # Полное описание вакансии
            description = data.get("description", "")
            if description:
                # Очищаем HTML теги
                description = re.sub(r'<[^>]+>', ' ', description)
                description = re.sub(r'\s+', ' ', description).strip()

            # Требования (для краткого отображения)
            requirement = data.get("requirement", "")
            if requirement:
                requirement = re.sub(r'<[^>]+>', ' ', requirement)
                requirement = re.sub(r'\s+', ' ', requirement).strip()

            # Обязанности
            responsibility = data.get("responsibility", "")
            if responsibility:
                responsibility = re.sub(r'<[^>]+>', ' ', responsibility)
                responsibility = re.sub(r'\s+', ' ', responsibility).strip()

            # Зарплата
            salary = data.get("salary") or {}

            # Компания
            employer = data.get("employer", {})

            # Адрес
            address = data.get("address", {})

            return {
                "id": data.get("id"),
                "name": data.get("name", ""),
                "employer": employer.get("name", "Не указана"),
                "employer_url": employer.get("alternate_url", ""),
                "employer_industries": employer.get("industries", []),
                "description": description,  # ПОЛНОЕ ОПИСАНИЕ
                "requirement": requirement,
                "responsibility": responsibility,
                "skills": skills,
                "salary_from": salary.get("from"),
                "salary_to": salary.get("to"),
                "currency": salary.get("currency"),
                "salary_gross": salary.get("gross"),
                "city": address.get("city", address.get("town", address.get("name", "Не указан"))),
                "street": address.get("street", ""),
                "building": address.get("building", ""),
                "lat": address.get("lat"),
                "lng": address.get("lng"),
                "experience": data.get("experience", {}).get("name", "Не указан"),
                "employment": data.get("employment", {}).get("name", "Не указана"),
                "schedule": data.get("schedule", {}).get("name", "Не указан"),
                "published_at": data.get("published_at"),
                "url": data.get("alternate_url", f"https://hh.ru/vacancy/{vacancy_id}"),
                "department": data.get("department", {}).get("name"),
                "contacts": data.get("contacts"),
                "professional_roles": [role.get("name") for role in data.get("professional_roles", [])],
            }

        except Exception as e:
            print(f"❌ Ошибка получения вакансии {vacancy_id}: {e}")
            return None

    def _parse_vacancy_item(self, item):
        """
        Обрабатывает базовую информацию о вакансии (без полного описания).
        """
        salary = item.get("salary")
        if salary:
            salary_from = salary.get("from")
            salary_to = salary.get("to")
            currency = salary.get("currency")

            if salary_from and salary_to:
                salary_str = f"{salary_from:,} - {salary_to:,} {currency}".replace(",", " ")
            elif salary_from:
                salary_str = f"от {salary_from:,} {currency}".replace(",", " ")
            elif salary_to:
                salary_str = f"до {salary_to:,} {currency}".replace(",", " ")
            else:
                salary_str = "Не указана"
        else:
            salary_str = "Не указана"

        area = item.get("area", {})
        city = area.get("name", "Не указан")

        snippet = item.get("snippet", {})
        requirement = snippet.get("requirement", "Не указаны")
        if requirement:
            requirement = re.sub(r'<[^>]+>', '', requirement)

        # Описание из snippets (краткое)
        description = snippet.get("responsibility", "")
        if description:
            description = re.sub(r'<[^>]+>', '', description)

        return {
            "id": item.get("id"),
            "name": item.get("name"),
            "employer": item.get("employer", {}).get("name", "Не указана"),
            "salary": salary_str,
            "salary_from": salary.get("from") if salary else None,
            "salary_to": salary.get("to") if salary else None,
            "currency": salary.get("currency") if salary else None,
            "city": city,
            "published_at": item.get("published_at"),
            "url": item.get("alternate_url"),
            "requirement": requirement,
            "description": description,  # Краткое описание из сниппета
            "skills": [],
        }

    def save_to_json(self, data, filename="vacancies.json"):
        """Сохраняет результат в JSON-файл."""
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"💾 Данные сохранены в {filename}")

    def save_to_csv(self, data, filename="vacancies.csv"):
        """Сохраняет результат в CSV-файл."""
        import csv

        if not data:
            print("Нет данных для сохранения в CSV")
            return

        # Расширенный список полей для CSV
        fieldnames = [
            "id", "name", "employer", "city", "salary_from", "salary_to",
            "currency", "experience", "employment", "schedule",
            "published_at", "url", "requirement", "description"
        ]

        with open(filename, "w", encoding="utf-8-sig", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()

            for row in data:
                # Очищаем описание от переносов строк для CSV
                if "description" in row and row["description"]:
                    row["description"] = row["description"].replace('\n', ' ').replace('\r', ' ')
                if "requirement" in row and row["requirement"]:
                    row["requirement"] = row["requirement"].replace('\n', ' ').replace('\r', ' ')
                writer.writerow(row)

        print(f"💾 Данные сохранены в {filename}")


# ===================== ЗАПУСК =====================
if __name__ == "__main__":


    # Если IP заблокирован — добавьте прокси или включите VPN
    PROXY = None  # Например: "http://51.89.96.239:3128"

    # Создаем парсер
    parser = HHParser(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        proxy=PROXY
    )

    # Получаем токен
    token = parser.get_client_credentials_token()

    if not token:
        print("\n⚠️ Работаю без токена...\n")
        parser.headers.pop("Authorization", None)

    print("=" * 60)
    print("🔍 Поиск 100 вакансий с ПОЛНЫМ описанием")
    print("=" * 60)

    # ===== ИЗМЕНЕНИЯ ЗДЕСЬ =====
    vacancies = parser.search_vacancies(
        text="Экономист",  # Поисковый запрос
        area=1,  # 1 - Москва, 113 - Россия
        per_page=30,  # Сколько на странице (макс 100)
        max_pages=1  # 1 страница * 100 = 100 вакансий
    )
    # ============================

    if vacancies:
        print(f"\n✅ Найдено вакансий: {len(vacancies)}")

        print("\n📋 Примеры вакансий:")
        for i, vacancy in enumerate(vacancies[:3], 1):
            print(f"\n{'=' * 60}")
            print(f"{i}. {vacancy.get('name', 'Без названия')}")
            print(f"   Компания: {vacancy.get('employer', 'Не указана')}")
            print(f"   Город: {vacancy.get('city', 'Не указан')}")
            print(
                f"   Зарплата: от {vacancy.get('salary_from', '?')} до {vacancy.get('salary_to', '?')} {vacancy.get('currency', '')}")
            print(f"   Опыт: {vacancy.get('experience', 'Не указан')}")
            print(f"   URL: {vacancy.get('url', '')}")
            print(f"\n   📝 Требования: {vacancy.get('requirement', 'Не указаны')[:200]}...")
            print(f"\n   📄 Описание: {vacancy.get('description', 'Отсутствует')[:300]}...")
            if vacancy.get('skills'):
                print(f"\n   🔧 Навыки: {', '.join(vacancy['skills'][:5])}")

        # Сохраняем результаты
        parser.save_to_json(vacancies, "pyt1v1ac11a.json")
        parser.save_to_csv(vacancies, "py1t1va11.csv")

        print(f"\n✅ Сохранено {len(vacancies)} вакансий с полным описанием")
        print(f"📁 Файлы: python_vacancies_100.json и python_vacancies_100.csv")
    else:
        print("❌ Вакансии не найдены.")