# app.py - Полностью исправленная версия
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import atexit
import logging
import asyncio
import sys
from pathlib import Path

# Добавляем корень проекта в путь
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# ==================== НАСТРОЙКИ ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="AI Рекомендательная Система Вакансий",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CSS СТИЛИ ====================
CSS_STYLES = """
<style>
    .main-header { font-size: 2.5rem; color: #1E3A8A; text-align: center; margin-bottom: 1rem; }
    .sub-header { font-size: 1.5rem; color: #3B82F6; margin-top: 1.5rem; margin-bottom: 1rem; }
    .metric-card { background-color: #F8FAFC; padding: 1rem; border-radius: 10px; border-left: 4px solid #3B82F6; margin-bottom: 1rem; }
    .vacancy-card { background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px; padding: 1rem; margin-bottom: 1rem; transition: all 0.3s ease; }
    .vacancy-card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.1); transform: translateY(-2px); }
    .skill-tag { background-color: #E0F2FE; color: #0369A1; padding: 0.2rem 0.5rem; border-radius: 15px; font-size: 0.8rem; display: inline-block; margin: 0.2rem; }
    .feedback-button { margin: 0.2rem; }
    .stButton>button { width: 100%; }
</style>
"""
st.markdown(CSS_STYLES, unsafe_allow_html=True)

# ==================== ГЛОБАЛЬНАЯ ПЕРЕМЕННАЯ ДЛЯ СЕРВИСОВ ====================
services = None


# ==================== УТИЛИТЫ ====================
def setup_session_state():
    """Инициализация состояния сессии"""
    defaults = {
        'current_user': None,
        'recommendations': [],
        'search_results': [],
        'feedback_history': []
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_header():
    """Отображение заголовка приложения"""
    st.markdown('<h1 class="main-header">💼 AI Рекомендательная Система Вакансий</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem; color: #64748B;'>
    Гибридная система рекомендаций на основе Neo4j, Mistral AI и графовых алгоритмов
    </div>
    """, unsafe_allow_html=True)


# ==================== СЕРВИСЫ ====================
@st.cache_resource
def init_services():
    """Инициализация всех сервисов"""
    global services

    try:
        from api.config import settings
        from src.database.neo4j_client import Neo4jClient
        from src.ai.embeddings import EmbeddingService
        from src.services.user_service import UserService
        from src.services.vacancy_service import VacancyService
        from src.services.feedback_service import FeedbackService
        from src.services.recommendation_service import RecommendationService
        from src.parsers.hh_parser import HHParser

        logger.info("🚀 Инициализация сервисов...")

        # 1. Neo4j
        logger.info("📡 Подключение к Neo4j...")
        neo4j_client = Neo4jClient()
        neo4j_client.connect()
        neo4j_client.initialize_database()
        logger.info("✅ Neo4j подключен")

        # 2. Embedding сервис
        logger.info("🧠 Инициализация Embedding сервиса...")
        embedding_service = EmbeddingService()
        logger.info("✅ Embedding сервис готов")

        # 3. Сервисы с правильными аргументами
        logger.info("👤 Создание UserService...")
        user_service = UserService(neo4j_client)  # ← 1 аргумент

        logger.info("💼 Создание VacancyService...")
        vacancy_service = VacancyService(neo4j_client, embedding_service)  # ← 2 аргумента

        logger.info("🎯 Создание RecommendationService...")
        recommendation_service = RecommendationService(neo4j_client, embedding_service)  # ← 2 аргумента

        logger.info("💬 Создание FeedbackService...")
        feedback_service = FeedbackService(neo4j_client)  # ← 1 аргумент
        feedback_service.init(neo4j_client)  # ← Инициализируем!

        logger.info("🤖 Создание HH Parser...")
        parser = HHParser()  # ← 0 аргументов

        logger.info("🎉 Все сервисы успешно инициализированы!")

        services = {
            'neo4j': neo4j_client,
            'embedding': embedding_service,
            'user_service': user_service,
            'vacancy_service': vacancy_service,
            'recommendation_service': recommendation_service,
            'feedback_service': feedback_service,
            'parser': parser
        }

        return services

    except ImportError as e:
        logger.error(f"Ошибка импорта: {e}")
        st.error(f"❌ Ошибка импорта модулей: {e}")
        st.info("Проверьте структуру проекта и пути импорта")
    except Exception as e:
        logger.error(f"Ошибка инициализации: {e}")
        st.error(f"❌ Не удалось инициализировать сервисы: {e}")
        st.info("""
        **Проверьте:**
        1. Запущен ли Neo4j (bolt://localhost:7687)
        2. Правильность пароля в .env
        3. Структуру проекта
        """)
    return None


# ==================== КОМПОНЕНТЫ ВАКАНСИЙ ====================
def render_vacancy_card(vacancy, user, context="search"):
    """Универсальный компонент отображения вакансии"""

    st.markdown('<div class="vacancy-card">', unsafe_allow_html=True)

    # Заголовок и зарплата
    col_title, col_salary = st.columns([3, 1])
    with col_title:
        title = getattr(vacancy, 'title', None) or 'Без названия'
        st.markdown(f"#### {title}")

        company_info = []
        company = getattr(vacancy, 'company_name', None)
        location = getattr(vacancy, 'location_name', None)
        experience = getattr(vacancy, 'experience', None)

        if company:
            company_info.append(f"🏢 {company}")
        if location:
            company_info.append(f"📍 {location}")
        if experience:
            company_info.append(f"🎓 {experience}")

        st.markdown(" • ".join(company_info) if company_info else "ℹ️ Информация не указана")

    with col_salary:
        salary_from = getattr(vacancy, 'salary_from', None)
        salary_to = getattr(vacancy, 'salary_to', None)
        currency = getattr(vacancy, 'currency', 'RUB')

        if salary_from or salary_to:
            salary_parts = []
            if salary_from:
                salary_parts.append(f"от {salary_from:,}")
            if salary_to:
                salary_parts.append(f"до {salary_to:,}")
            salary_parts.append(currency)
            st.markdown(f"**{' - '.join(salary_parts)}**")
        else:
            st.markdown("💰 Зарплата не указана")

    # Навыки
    skills = getattr(vacancy, 'skills', []) or []
    if skills:
        st.markdown("**Требуемые навыки:**")
        cols = st.columns(5)
        for i, skill in enumerate(skills[:10]):
            if skill:
                with cols[i % 5]:
                    st.markdown(f'<div class="skill-tag">{skill}</div>', unsafe_allow_html=True)

    # Описание
    description = getattr(vacancy, 'description', None)
    if description and len(description) > 100:
        with st.expander("📋 Описание вакансии"):
            preview = description[:500] + "..." if len(description) > 500 else description
            st.markdown(preview)

    # Кнопки обратной связи
    st.markdown("---")
    col_like, col_dislike, col_view, col_apply = st.columns(4)

    vacancy_id = getattr(vacancy, 'id', None)
    user_id = getattr(user, 'id', None)

    if vacancy_id and user_id:
        from src.database.models import UserFeedback, FeedbackType

        with col_like:
            if st.button("👍 Нравится", key=f"{context}_like_{vacancy_id}", use_container_width=True):
                feedback = UserFeedback(user_id=user_id, vacancy_id=vacancy_id, feedback_type=FeedbackType.LIKE)
                if services and services['feedback_service'].record_feedback(feedback):
                    st.success("✅ Спасибо за оценку!")
                    st.rerun()

        with col_dislike:
            if st.button("👎 Не нравится", key=f"{context}_dislike_{vacancy_id}", use_container_width=True):
                feedback = UserFeedback(user_id=user_id, vacancy_id=vacancy_id, feedback_type=FeedbackType.DISLIKE)
                if services and services['feedback_service'].record_feedback(feedback):
                    st.success("✅ Учтено!")
                    st.rerun()

        with col_view:
            if st.button("👁️ Подробнее", key=f"{context}_view_{vacancy_id}", use_container_width=True):
                feedback = UserFeedback(user_id=user_id, vacancy_id=vacancy_id, feedback_type=FeedbackType.VIEW)
                if services:
                    services['feedback_service'].record_feedback(feedback)
                if description:
                    with st.expander("📋 Полное описание", expanded=True):
                        st.markdown(description)

        with col_apply:
            if st.button("📨 Отклик", key=f"{context}_apply_{vacancy_id}", use_container_width=True):
                feedback = UserFeedback(user_id=user_id, vacancy_id=vacancy_id, feedback_type=FeedbackType.APPLY)
                if services and services['feedback_service'].record_feedback(feedback):
                    st.success("✅ Отклик записан!")
                    st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


def filter_vacancies(vacancies, min_salary, show_only_new):
    """Фильтрация списка вакансий"""
    if not vacancies:
        return []

    filtered = []
    for vacancy in vacancies:
        if not vacancy:
            continue

        # Фильтр по зарплате
        if min_salary > 0:
            salary_from = getattr(vacancy, 'salary_from', 0) or 0
            salary_to = getattr(vacancy, 'salary_to', 0) or 0

            if salary_to > 0 and salary_to < min_salary:
                continue
            if salary_from > 0 and salary_from < min_salary and salary_to == 0:
                continue

        # Фильтр по новизне
        if show_only_new:
            published = getattr(vacancy, 'published_at', None)
            if published:
                if hasattr(published, 'tzinfo') and published.tzinfo is not None:
                    published = published.replace(tzinfo=None)
                if datetime.now() - published > timedelta(days=30):
                    continue

        filtered.append(vacancy)
    return filtered


# ==================== ОБНОВЛЕНИЕ ДАННЫХ ====================
def update_feedback_history():
    """Обновление истории обратной связи"""
    if services and st.session_state.current_user:
        try:
            history = services['feedback_service'].get_user_feedback_history(
                st.session_state.current_user.id, 20
            )
            st.session_state.feedback_history = history if history else []
        except Exception as e:
            logger.error(f"Error updating feedback history: {e}")
            st.session_state.feedback_history = []


@st.cache_data(ttl=300)
def get_system_stats():
    """Получение статистики системы"""
    if not services:
        return None

    try:
        return {
            'user_count': services['neo4j'].execute_query("MATCH (u:User) RETURN COUNT(u) AS count")[0]['count'],
            'vacancy_count': services['neo4j'].execute_query("MATCH (v:Vacancy) RETURN COUNT(v) AS count")[0]['count'],
            'skill_count': services['neo4j'].execute_query("MATCH (s:Skill) RETURN COUNT(s) AS count")[0]['count']
        }
    except Exception as e:
        logger.warning(f"Ошибка получения статистики: {e}")
        return None


# ==================== СТРАНИЦА ПРОФИЛЯ ====================
def render_profile_page():
    """Страница управления профилем"""
    st.markdown('<h2 class="sub-header">👤 Управление профилем</h2>', unsafe_allow_html=True)

    # Функция для редактирования профиля
    def render_edit_form(user):
        """Форма редактирования профиля"""
        with st.expander("✏️ Редактировать профиль", expanded=False):
            with st.form("edit_user_form"):
                username = st.text_input("Имя пользователя", value=user.username)
                skills_input = st.text_area(
                    "Навыки через запятую",
                    value=", ".join(user.skills) if hasattr(user, 'skills') and user.skills else ""
                )
                resume_text = st.text_area(
                    "Резюме",
                    value=user.resume_text if hasattr(user, 'resume_text') else ""
                )

                if st.form_submit_button("💾 Сохранить изменения", type="primary"):
                    if not username or not skills_input or not resume_text:
                        st.error("⚠️ Заполните все обязательные поля")
                    else:
                        try:
                            skills = [s.strip() for s in skills_input.split(',') if s.strip()]
                            
                            # Создаем обновленный объект пользователя
                            updated_user = type(user)(
                                id=user.id,
                                username=username,
                                resume_text=resume_text,
                                skills=skills
                            )
                            
                            if services['user_service'].create_or_update_user(updated_user):
                                st.session_state.current_user = updated_user
                                st.success("✅ Профиль успешно обновлен!")
                                update_feedback_history()
                                st.rerun()
                            else:
                                st.error("❌ Не удалось обновить профиль")
                        except Exception as e:
                            st.error(f"❌ Ошибка при обновлении профиля: {e}")

    if not services:
        st.error("❌ Сервисы не инициализированы")
        return

    col_select, col_create = st.columns(2)

    # Загрузка профиля
    with col_select:
        st.markdown("### 📂 Загрузить существующий профиль")

        search_term = st.text_input("Поиск по имени", placeholder="Введите имя пользователя...", key="profile_search")

        if st.button("🔍 Поиск пользователей") or search_term:
            try:
                if search_term:
                    query = """
                    MATCH (u:User)
                    WHERE toLower(u.username) CONTAINS toLower($search)
                    RETURN u.id AS id, u.username AS username
                    ORDER BY u.username LIMIT 20
                    """
                    users = services['neo4j'].execute_query(query, {'search': search_term})
                else:
                    query = """
                    MATCH (u:User)
                    RETURN u.id AS id, u.username AS username
                    ORDER BY u.username LIMIT 20
                    """
                    users = services['neo4j'].execute_query(query)

                if users:
                    for user_data in users:
                        col_user, col_btn = st.columns([3, 1])
                        with col_user:
                            st.write(f"**{user_data.get('username', 'N/A')}**")
                            st.caption(f"ID: {user_data.get('id', 'N/A')}")
                        with col_btn:
                            if st.button("📥 Загрузить", key=f"load_{user_data['id']}"):
                                loaded_user = services['user_service'].get_user_by_id(user_data['id'])
                                if loaded_user:
                                    st.session_state.current_user = loaded_user
                                    st.success(f"✅ Профиль загружен!")
                                    update_feedback_history()
                                    st.rerun()
                else:
                    st.info("Пользователи не найдены")
            except Exception as e:
                st.error(f"Ошибка поиска: {e}")

    # Создание профиля
    with col_create:
        st.markdown("### 🆕 Создать новый профиль")

        with st.form("create_user_form", clear_on_submit=True):
            username = st.text_input("Имя пользователя*", placeholder="ivan_ivanov")
            skills_input = st.text_area("Навыки через запятую*", placeholder="Python, SQL, Docker...", height=100)
            resume_text = st.text_area("Резюме*", placeholder="Опытный разработчик...", height=150)

            if st.form_submit_button("✅ Создать профиль", type="primary"):
                if not username or not skills_input or not resume_text:
                    st.error("⚠️ Заполните все обязательные поля")
                else:
                    try:
                        skills = [s.strip() for s in skills_input.split(',') if s.strip()]
                        user_id = f"user_{int(datetime.now().timestamp())}"

                        from src.database.models import User
                        new_user = User(
                            id=user_id,
                            username=username,
                            resume_text=resume_text,
                            skills=skills
                        )

                        if services['user_service'].create_or_update_user(new_user):
                            st.session_state.current_user = new_user
                            st.success(f"🎉 Профиль {username} создан!")
                            update_feedback_history()
                            st.rerun()
                        else:
                            st.error("Не удалось создать профиль")
                    except Exception as e:
                        st.error(f"Ошибка: {e}")

    # Текущий профиль
    if st.session_state.current_user:
        user = st.session_state.current_user
        st.markdown("---")
        st.markdown('<h2 class="sub-header">📋 Текущий профиль</h2>', unsafe_allow_html=True)

        # Кнопка редактирования
        if hasattr(user, 'id') and user.id:
            render_edit_form(user)

        col_info, col_stats = st.columns([2, 1])
        with col_info:
            st.markdown(f"### {user.username}")
            st.markdown(f"**ID:** `{user.id}`")
            if hasattr(user, 'resume_text') and user.resume_text:
                with st.expander("📄 Просмотреть резюме"):
                    st.write(user.resume_text)

        with col_stats:
            if hasattr(user, 'skills') and user.skills:
                st.metric("🔧 Навыки", len(user.skills))

        if hasattr(user, 'skills') and user.skills:
            st.markdown("#### 🔧 Навыки")
            cols = st.columns(4)
            for i, skill in enumerate(user.skills):
                with cols[i % 4]:
                    st.markdown(f'<div class="skill-tag">{skill}</div>', unsafe_allow_html=True)


# ==================== СТРАНИЦА ПОИСКА ====================
def render_search_page():
    """Страница поиска вакансий"""
    st.markdown('<h2 class="sub-header">🔍 Поиск и анализ вакансий</h2>', unsafe_allow_html=True)

    if not services:
        st.error("❌ Сервисы не инициализированы")
        return

    if not st.session_state.current_user:
        st.warning("⚠️ Сначала создайте или загрузите профиль")
        return

    # Параметры поиска
    col_search, col_settings = st.columns([3, 1])
    with col_search:
        search_query = st.text_input("🔍 Поисковый запрос", value="Python разработчик",
                                     placeholder="Введите должность, технологию или компанию...")
    with col_settings:
        limit = st.slider("📊 Количество", 5, 30, 15)

    # Поиск
    if st.button("🚀 Начать поиск", type="primary", use_container_width=True):
        if not search_query.strip():
            st.error("⚠️ Введите поисковый запрос")
        else:
            with st.spinner("🔎 Ищем вакансии..."):
                try:
                    # Используем fetch_and_parse_vacancies для полной обработки
                    vacancies = services['parser'].fetch_and_parse_vacancies(
                        search_query=search_query,
                        limit=limit
                    )

                    if not vacancies:
                        st.warning("😕 Вакансий не найдено")
                    else:
                        st.session_state.search_results = vacancies
                        st.success(f"✅ Загружено {len(vacancies)} вакансий!")

                        # Сохраняем в базу
                        saved_count = 0
                        for vac in vacancies:
                            if services['vacancy_service'].save_vacancy(vac):
                                saved_count += 1

                        if saved_count > 0:
                            st.info(f"💾 Сохранено {saved_count} вакансий в базу данных")

                except Exception as e:
                    st.error(f"❌ Ошибка поиска: {e}")
                    logger.error(f"Search error: {e}")

    # Отображение результатов
    if st.session_state.search_results:
        st.markdown(f"### 📄 Результаты поиска ({len(st.session_state.search_results)})")

        col1, col2 = st.columns(2)
        with col1:
            min_salary = st.number_input("💰 Мин. зарплата", min_value=0, value=0, step=10000)
        with col2:
            show_only_new = st.checkbox("🆕 Только новые (30 дней)", value=False)

        filtered = filter_vacancies(st.session_state.search_results, min_salary, show_only_new)

        if filtered:
            for vacancy in filtered:
                render_vacancy_card(vacancy, st.session_state.current_user, "search")
        else:
            st.info("📭 Нет вакансий, соответствующих фильтрам")


# ==================== СТРАНИЦА РЕКОМЕНДАЦИЙ ====================
def render_recommendations_page():
    """Страница персональных рекомендаций"""
    st.markdown('<h2 class="sub-header">🎯 Персональные рекомендации</h2>', unsafe_allow_html=True)

    if not services:
        st.error("❌ Сервисы не инициализированы")
        return

    if not st.session_state.current_user:
        st.warning("⚠️ Сначала создайте или загрузите профиль")
        return

    user = st.session_state.current_user

    # Настройки
    col1, col2, col3 = st.columns(3)
    with col1:
        num_rec = st.slider("📊 Количество", 3, 20, 5)
    with col2:
        content_weight = st.slider("📝 Контентный вес", 0.0, 1.0, 0.3, 0.05)
    with col3:
        semantic_weight = st.slider("🧠 Семантический вес", 0.0, 1.0, 0.3, 0.05)

    # Получение рекомендаций
    if st.button("🚀 Получить рекомендации", type="primary", use_container_width=True):
        # Проверка на наличие навыков у пользователя
        if not hasattr(user, 'skills') or not user.skills:
            st.warning("⚠️ У пользователя нет навыков. Добавьте навыки в профиле для получения рекомендаций.")
            return
            
        with st.spinner("🧠 Анализируем предпочтения..."):
            try:
                recommendations = services['vacancy_service'].get_recommendations(
                    user.id, num_rec,
                    content_weight=content_weight,
                    semantic_weight=semantic_weight
                )
                # Обработка None от get_recommendations
                if recommendations is None:
                    recommendations = []
                st.session_state.recommendations = recommendations if recommendations else []

                if st.session_state.recommendations:
                    st.success(f"✅ Найдено {len(st.session_state.recommendations)} рекомендаций!")
                else:
                    st.info("📭 Рекомендаций не найдено. Попробуйте добавить больше навыков.")
            except Exception as e:
                st.error(f"❌ Ошибка получения рекомендаций: {e}")
                logger.error(f"Recommendation error: {e}")

    # Отображение рекомендаций
    if st.session_state.recommendations:
        st.markdown("### 📋 Рекомендованные вакансии")

        for i, rec in enumerate(st.session_state.recommendations, 1):
            vacancy = rec.vacancy if hasattr(rec, 'vacancy') else rec

            with st.expander(f"{i}. {getattr(vacancy, 'title', 'Без названия')}", expanded=i <= 3):
                render_vacancy_card(vacancy, user, f"rec_{i}")


# ==================== СТРАНИЦА АНАЛИТИКИ ====================
def render_analytics_page():
    """Страница аналитики"""
    st.markdown('<h2 class="sub-header">📊 Аналитика системы</h2>', unsafe_allow_html=True)

    if not services:
        st.error("❌ Сервисы не инициализированы")
        return

    if not st.session_state.current_user:
        st.warning("⚠️ Загрузите профиль")
        return

    update_feedback_history()

    # Статистика пользователя
    st.markdown("### 👤 Ваша статистика")

    try:
        stats = services['neo4j'].execute_query("""
        MATCH (u:User {id: $user_id})-[r:VIEWED|RATED]->(:Vacancy)
        RETURN 
            COUNT(CASE WHEN type(r) = 'RATED' AND r.rating >= 4 THEN 1 END) AS likes,
            COUNT(CASE WHEN type(r) = 'RATED' AND r.rating <= 2 THEN 1 END) AS dislikes,
            COUNT(CASE WHEN type(r) = 'VIEWED' THEN 1 END) AS views,
            COUNT(CASE WHEN type(r) = 'RATED' THEN 1 END) AS applies
        """, {'user_id': st.session_state.current_user.id})

        if stats:
            s = stats[0]
            cols = st.columns(4)
            metrics = [
                ("👍 Лайков", s.get('likes', 0)),
                ("👎 Дизлайков", s.get('dislikes', 0)),
                ("👁️ Просмотров", s.get('views', 0)),
                ("📨 Откликов", s.get('applies', 0))
            ]
            for col, (label, value) in zip(cols, metrics):
                with col:
                    st.metric(label, value)
    except Exception as e:
        logger.warning(f"Ошибка статистики пользователя: {e}")

    # Статистика системы
    st.markdown("### 🏢 Статистика системы")
    sys_stats = get_system_stats()
    if sys_stats:
        cols = st.columns(3)
        with cols[0]:
            st.metric("👥 Пользователи", sys_stats['user_count'])
        with cols[1]:
            st.metric("💼 Вакансии", sys_stats['vacancy_count'])
        with cols[2]:
            st.metric("🔧 Навыки", sys_stats['skill_count'])


# ==================== СТРАНИЦА НАСТРОЕК ====================
def render_settings_page():
    """Страница настроек"""
    st.markdown('<h2 class="sub-header">⚙️ Настройки системы</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎯 Настройки рекомендаций")
        from api.config import settings

        content_weight = st.slider("Вес контентной фильтрации", 0.0, 1.0, settings.content_weight, 0.05)
        graph_weight = st.slider("Вес графовой фильтрации", 0.0, 1.0, settings.graph_weight, 0.05)
        semantic_weight = st.slider("Вес семантической фильтрации", 0.0, 1.0, settings.semantic_weight, 0.05)

        total = content_weight + graph_weight + semantic_weight
        if abs(total - 1.0) > 0.01:
            st.warning(f"⚠️ Сумма весов: {total:.2f} (должна быть 1.0)")
        else:
            settings.content_weight = content_weight
            settings.graph_weight = graph_weight
            settings.semantic_weight = semantic_weight
            st.success("✅ Веса обновлены")

    with col2:
        st.markdown("### 🔧 Утилиты")

        if st.button("🗑️ Очистить кэш", use_container_width=True):
            st.cache_resource.clear()
            st.cache_data.clear()
            st.success("✅ Кэш очищен")
            st.rerun()

        if st.button("🔄 Перезагрузить сервисы", use_container_width=True):
            init_services.clear()
            st.success("✅ Сервисы перезагружены")
            st.rerun()


# ==================== САЙДБАР ====================
def render_sidebar():
    """Отображение боковой панели"""
    st.sidebar.title("🔍 Навигация")

    menu_options = {
        "👤 Профиль": render_profile_page,
        "🔍 Поиск вакансий": render_search_page,
        "🎯 Рекомендации": render_recommendations_page,
        "📊 Аналитика": render_analytics_page,
        "⚙️ Настройки": render_settings_page
    }

    selected = st.sidebar.radio("Выберите раздел:", list(menu_options.keys()))

    # Информация о пользователе
    if st.session_state.current_user:
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"### 👤 {st.session_state.current_user.username}")
        st.sidebar.info(f"ID: {st.session_state.current_user.id}")

        if hasattr(st.session_state.current_user, 'skills') and st.session_state.current_user.skills:
            st.sidebar.markdown("**Навыки:**")
            for skill in st.session_state.current_user.skills[:5]:
                st.sidebar.markdown(f"• {skill}")

        if st.sidebar.button("🚪 Выйти из профиля", use_container_width=True):
            st.session_state.current_user = None
            st.session_state.recommendations = []
            st.session_state.search_results = []
            st.rerun()

    # Статистика
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Статистика системы")
    stats = get_system_stats()
    if stats:
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("👥", stats['user_count'])
        with col2:
            st.metric("💼", stats['vacancy_count'])
        st.sidebar.metric("🔧 Навыки", stats['skill_count'])

    return menu_options[selected]


# ==================== ОСНОВНОЙ КОД ====================
def main():
    """Главная функция"""
    setup_session_state()
    render_header()

    # Загрузка сервисов
    global services
    services = init_services()

    if not services:
        st.error("""
        ⚠️ **Не удалось инициализировать сервисы.**

        **Проверьте:**
        1. Запущен ли Neo4j (bolt://localhost:7687)
        2. Правильность пароля в .env
        3. Структуру проекта
        """)
        st.stop()

    # Отображение выбранной страницы
    try:
        render_page = render_sidebar()
        render_page()
    except Exception as e:
        logger.error(f"Ошибка рендеринга: {e}")
        st.error(f"Произошла ошибка: {e}")

    # Футер
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #64748B; font-size: 0.9rem; margin-top: 2rem;'>
        <p>💡 <strong>AI Рекомендательная Система Вакансий</strong></p>
        <p>Архитектура: Neo4j + Mistral AI + Гибридные алгоритмы | Матвеев А.В.</p>
    </div>
    """, unsafe_allow_html=True)


def cleanup():
    """Очистка ресурсов при завершении"""
    global services
    if services and 'neo4j' in services:
        try:
            services['neo4j'].close()
            logger.info("Neo4j connection closed")
        except:
            pass


if __name__ == "__main__":
    atexit.register(cleanup)
    main()