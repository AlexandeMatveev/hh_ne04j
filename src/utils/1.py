#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Парсер вакансий с hh.ru
Использует Application Token для доступа к API
"""

import requests
import pandas as pd
import time
import json
from datetime import datetime
import sys

# ==================== КОНФИГУРАЦИЯ ====================
# !!! ВАЖНО: Замените на ваш реальный токен приложения !!!
APPLICATION_TOKEN = "APPLMTROVEECLGIM4TPGOJRH00N0S7M12UTSIIL0938B4QJ9CITHCCH1PACER27F"

# Коды регионов (полный список: https://api.hh.ru/areas)
REGIONS = {
    "Москва": 1,
    "Санкт-Петербург": 2,
    "Россия": 113,
    "Новосибирск": 4,
    "Екатеринбург": 3,
    "Казань": 88,
    "Нижний Новгород": 66,
    "Краснодар": 53
}

# Настройки парсинга
DEFAULT_KEYWORD = "Python developer"  # Поисковый запрос
DEFAULT_REGION = "Москва"  # Регион поиска
DEFAULT_PAGES = 5  # Количество страниц (по 100 вакансий на странице)
REQUEST_DELAY = 0.5  # Задержка между запросами (секунды)


# ==================== ОСНОВНОЙ КЛАСС ПАРСЕРА ====================
class HHParser:
    """Парсер вакансий HeadHunter"""

    def __init__(self, token=None):
        """Инициализация парсера"""
        self.token = token or APPLICATION_TOKEN
        self.base_url = "https://api.hh.ru"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Authorization": f"Bearer {self.token}"
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def test_connection(self):
        """Проверка подключения к API"""
        try:
            response = self.session.get(f"{self.base_url}/vacancies", params={"text": "test", "per_page": 1})
            if response.status_code == 200:
                print("✅ Подключение к API успешно!")
                return True
            elif response.status_code == 403:
                print("❌ Ошибка: Токен недействителен или истек")
                return False
            else:
                print(f"❌ Ошибка: HTTP {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Ошибка подключения: {e}")
            return False

    def search_vacancies(self, keyword, region_name="Москва", pages=DEFAULT_PAGES):
        """
        Поиск вакансий

        Args:
            keyword (str): Поисковый запрос
            region_name (str): Название региона
            pages (int): Количество страниц для парсинга

        Returns:
            list: Список вакансий
        """
        # Получаем код региона
        area_id = REGIONS.get(region_name, 113)  # 113 - Россия по умолчанию

        print(f"\n🔍 Поиск вакансий: '{keyword}' в регионе '{region_name}'")
        print(f"📄 Будет обработано страниц: {pages}")

        all_vacancies = []

        for page in range(pages):
            params = {
                "text": keyword,
                "area": area_id,
                "per_page": 100,
                "page": page
            }

            print(f"  ⏳ Обработка страницы {page + 1}/{pages}...", end=" ", flush=True)

            try:
                response = self.session.get(f"{self.base_url}/vacancies", params=params)

                if response.status_code != 200:
                    print(f"❌ Ошибка {response.status_code}")
                    break

                data = response.json()
                vacancies = data.get("items", [])

                if not vacancies:
                    print("⚠️ Вакансии не найдены")
                    break

                print(f"✅ Найдено {len(vacancies)} вакансий")

                # Обрабатываем каждую вакансию
                for vac in vacancies:
                    parsed_vac = self._parse_vacancy(vac)
                    all_vacancies.append(parsed_vac)

                # Проверяем, есть ли еще страницы
                if page >= data.get("pages", 0) - 1:
                    break

                # Задержка между запросами
                time.sleep(REQUEST_DELAY)

            except Exception as e:
                print(f"❌ Ошибка: {e}")
                break

        print(f"\n📊 Всего собрано вакансий: {len(all_vacancies)}")
        return all_vacancies

    def _parse_vacancy(self, vacancy_data):
        """
        Парсинг отдельной вакансии

        Args:
            vacancy_data (dict): Данные вакансии из API

        Returns:
            dict: Обработанные данные вакансии
        """
        # Обработка зарплаты
        salary = vacancy_data.get("salary")
        salary_from = None
        salary_to = None
        salary_currency = None

        if salary:
            salary_from = salary.get("from")
            salary_to = salary.get("to")
            salary_currency = salary.get("currency")

        # Расчет средней зарплаты
        avg_salary = None
        if salary_from and salary_to:
            avg_salary = (salary_from + salary_to) / 2
        elif salary_from:
            avg_salary = salary_from
        elif salary_to:
            avg_salary = salary_to

        # Обработка требований (удаляем HTML теги)
        requirement = vacancy_data.get("snippet", {}).get("requirement", "")
        if requirement:
            import re
            requirement = re.sub(r'<[^>]+>', '', requirement)

        return {
            "id": vacancy_data.get("id"),
            "name": vacancy_data.get("name"),
            "salary_from": salary_from,
            "salary_to": salary_to,
            "salary_currency": salary_currency,
            "salary_avg": avg_salary,
            "employer_name": vacancy_data.get("employer", {}).get("name"),
            "employer_url": vacancy_data.get("employer", {}).get("url"),
            "experience": vacancy_data.get("experience", {}).get("name"),
            "employment": vacancy_data.get("employment", {}).get("name"),
            "schedule": vacancy_data.get("schedule", {}).get("name"),
            "requirement": requirement,
            "responsibility": vacancy_data.get("snippet", {}).get("responsibility", ""),
            "published_at": vacancy_data.get("published_at"),
            "url": vacancy_data.get("alternate_url")
        }

    def get_vacancy_details(self, vacancy_id):
        """
        Получение детальной информации о вакансии по ID

        Args:
            vacancy_id (str): ID вакансии

        Returns:
            dict: Детальная информация о вакансии
        """
        try:
            response = self.session.get(f"{self.base_url}/vacancies/{vacancy_id}")

            if response.status_code == 200:
                data = response.json()
                return {
                    "id": data.get("id"),
                    "description": data.get("description", ""),
                    "key_skills": [skill["name"] for skill in data.get("key_skills", [])],
                    "additional_info": data
                }
            else:
                return None
        except Exception as e:
            print(f"Ошибка получения деталей вакансии {vacancy_id}: {e}")
            return None


# ==================== АНАЛИЗ ДАННЫХ ====================
class VacancyAnalyzer:
    """Анализатор собранных вакансий"""

    @staticmethod
    def create_dataframe(vacancies):
        """Создание DataFrame из списка вакансий"""
        return pd.DataFrame(vacancies)

    @staticmethod
    def analyze_salaries(df):
        """Анализ зарплат"""
        print("\n" + "=" * 50)
        print("📊 АНАЛИЗ ЗАРПЛАТ")
        print("=" * 50)

        # Убираем вакансии без зарплаты
        df_with_salary = df[df['salary_avg'].notna()]

        if len(df_with_salary) == 0:
            print("⚠️ Нет данных о зарплатах для анализа")
            return

        print(f"📈 Вакансий с указанной зарплатой: {len(df_with_salary)} из {len(df)}")
        print(
            f"💰 Средняя зарплата: {df_with_salary['salary_avg'].mean():.0f} {df_with_salary['salary_currency'].mode()[0] if not df_with_salary['salary_currency'].empty else '₽'}")
        print(f"📊 Медианная зарплата: {df_with_salary['salary_avg'].median():.0f} ₽")
        print(f"🔽 Минимальная зарплата: {df_with_salary['salary_avg'].min():.0f} ₽")
        print(f"🔼 Максимальная зарплата: {df_with_salary['salary_avg'].max():.0f} ₽")

        # Распределение по диапазонам
        bins = [0, 50000, 100000, 150000, 200000, 300000, float('inf')]
        labels = ['<50k', '50-100k', '100-150k', '150-200k', '200-300k', '300k+']
        df_with_salary['salary_range'] = pd.cut(df_with_salary['salary_avg'], bins=bins, labels=labels)

        print("\n📊 Распределение зарплат:")
        salary_distribution = df_with_salary['salary_range'].value_counts().sort_index()
        for range_name, count in salary_distribution.items():
            percentage = (count / len(df_with_salary)) * 100
            print(f"  {range_name}: {count} вакансий ({percentage:.1f}%)")

    @staticmethod
    def analyze_employers(df):
        """Анализ работодателей"""
        print("\n" + "=" * 50)
        print("🏢 ТОП-10 РАБОТОДАТЕЛЕЙ")
        print("=" * 50)

        employer_counts = df['employer_name'].value_counts().head(10)
        for employer, count in employer_counts.items():
            print(f"  {employer}: {count} вакансий")

    @staticmethod
    def analyze_skills(df):
        """Анализ требований к навыкам"""
        print("\n" + "=" * 50)
        print("🔧 ЧАСТО ВСТРЕЧАЮЩИЕСЯ НАВЫКИ")
        print("=" * 50)

        # Простой анализ ключевых слов в требованиях
        skills_dict = {}
        common_skills = ['Python', 'Django', 'Flask', 'SQL', 'PostgreSQL',
                         'Git', 'Docker', 'Linux', 'API', 'REST', 'JavaScript',
                         'React', 'Pandas', 'NumPy', 'Machine Learning', 'AI']

        for skill in common_skills:
            count = df['requirement'].str.contains(skill, case=False, na=False).sum()
            if count > 0:
                skills_dict[skill] = count

        sorted_skills = sorted(skills_dict.items(), key=lambda x: x[1], reverse=True)

        for skill, count in sorted_skills[:10]:
            percentage = (count / len(df)) * 100
            print(f"  {skill}: {count} вакансий ({percentage:.1f}%)")

    @staticmethod
    def save_to_csv(vacancies, filename=None):
        """Сохранение данных в CSV"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"hh_vacancies_{timestamp}.csv"

        df = pd.DataFrame(vacancies)
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"\n💾 Данные сохранены в файл: {filename}")
        return filename

    @staticmethod
    def save_to_json(vacancies, filename=None):
        """Сохранение данных в JSON"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"hh_vacancies_{timestamp}.json"

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(vacancies, f, ensure_ascii=False, indent=2)
        print(f"💾 Данные сохранены в файл: {filename}")
        return filename


# ==================== ИНТЕРАКТИВНЫЙ РЕЖИМ ====================
def interactive_mode():
    """Интерактивный режим работы"""
    print("\n" + "=" * 60)
    print("🔍 ПАРСЕР ВАКАНСИЙ HEADHUNTER")
    print("=" * 60)

    # Инициализация парсера
    parser = HHParser()

    # Проверка подключения
    if not parser.test_connection():
        print("\n❌ Не удалось подключиться к API. Проверьте токен и интернет-соединение.")
        return

    # Ввод параметров поиска
    print("\n📝 Введите параметры поиска (или нажмите Enter для значений по умолчанию):")

    keyword = input(f"Поисковый запрос [{DEFAULT_KEYWORD}]: ").strip()
    if not keyword:
        keyword = DEFAULT_KEYWORD

    print("\nДоступные регионы:")
    for i, (region, code) in enumerate(REGIONS.items(), 1):
        print(f"  {i}. {region}")

    region_input = input(f"Выберите регион (1-{len(REGIONS)}) или название [{DEFAULT_REGION}]: ").strip()

    if region_input.isdigit() and 1 <= int(region_input) <= len(REGIONS):
        region_name = list(REGIONS.keys())[int(region_input) - 1]
    elif region_input:
        region_name = region_input
        if region_name not in REGIONS:
            print(f"⚠️ Регион '{region_name}' не найден, будет использована Россия")
            region_name = "Россия"
    else:
        region_name = DEFAULT_REGION

    pages_input = input(f"Количество страниц (1-20, каждая по 100 вакансий) [{DEFAULT_PAGES}]: ").strip()
    pages = int(pages_input) if pages_input.isdigit() else DEFAULT_PAGES
    pages = min(pages, 20)  # Ограничиваем максимум 20 страниц

    # Поиск вакансий
    vacancies = parser.search_vacancies(keyword, region_name, pages)

    if not vacancies:
        print("\n❌ Вакансии не найдены. Попробуйте изменить параметры поиска.")
        return

    # Создаем анализатор
    analyzer = VacancyAnalyzer()
    df = analyzer.create_dataframe(vacancies)

    # Анализ данных
    analyzer.analyze_salaries(df)
    analyzer.analyze_employers(df)
    analyzer.analyze_skills(df)

    # Сохранение результатов
    print("\n" + "=" * 50)
    print("💾 СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
    print("=" * 50)

    save_csv = input("Сохранить в CSV? (y/n) [y]: ").strip().lower()
    if save_csv != 'n':
        analyzer.save_to_csv(vacancies)

    save_json = input("Сохранить в JSON? (y/n) [n]: ").strip().lower()
    if save_json == 'y':
        analyzer.save_to_json(vacancies)

    # Показываем несколько примеров
    print("\n" + "=" * 50)
    print("📋 ПРИМЕРЫ ВАКАНСИЙ")
    print("=" * 50)

    for i, vac in enumerate(vacancies[:5], 1):
        print(f"\n{i}. {vac['name']}")
        print(f"   Компания: {vac['employer_name']}")
        if vac['salary_from'] or vac['salary_to']:
            salary_str = f"{vac['salary_from'] or ''} - {vac['salary_to'] or ''} {vac['salary_currency'] or '₽'}"
            print(f"   Зарплата: {salary_str.strip('- ')}")
        print(f"   Ссылка: {vac['url']}")

    print("\n✅ Парсинг завершен успешно!")


# ==================== БЫСТРЫЙ РЕЖИМ ====================
def quick_mode():
    """Быстрый режим с предустановленными параметрами"""
    parser = HHParser()

    if not parser.test_connection():
        print("❌ Ошибка подключения")
        return

    # Быстрый поиск Python разработчиков в Москве
    vacancies = parser.search_vacancies(
        keyword="Python developer",
        region_name="Москва",
        pages=3
    )

    if vacancies:
        analyzer = VacancyAnalyzer()
        df = analyzer.create_dataframe(vacancies)

        analyzer.analyze_salaries(df)
        analyzer.analyze_employers(df)
        analyzer.analyze_skills(df)

        # Сохраняем результаты
        filename = analyzer.save_to_csv(vacancies)
        print(f"\n✅ Результаты сохранены в {filename}")

        # Показываем 5 самых высокооплачиваемых вакансий
        df_with_salary = df[df['salary_avg'].notna()]
        if len(df_with_salary) > 0:
            print("\n🏆 ТОП-5 ВЫСОКООПЛАЧИВАЕМЫХ ВАКАНСИЙ:")
            top_salaries = df_with_salary.nlargest(5, 'salary_avg')[
                ['name', 'employer_name', 'salary_avg', 'salary_currency']]
            for idx, row in top_salaries.iterrows():
                print(
                    f"  • {row['name']} - {row['employer_name']}: {row['salary_avg']:.0f} {row['salary_currency'] or '₽'}")


# ==================== ТОЧКА ВХОДА ====================
if __name__ == "__main__":
    print("Выберите режим работы:")
    print("1. Интерактивный режим (с вводом параметров)")
    print("2. Быстрый режим (Python разработчики в Москве)")
    print("3. Тест подключения к API")

    choice = input("\nВаш выбор (1-3): ").strip()

    if choice == "1":
        interactive_mode()
    elif choice == "2":
        quick_mode()
    elif choice == "3":
        parser = HHParser()
        parser.test_connection()
    else:
        print("❌ Неверный выбор. Запустите программу снова.")
        sys.exit(1)