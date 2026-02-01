import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np
import atexit
import logging
import asyncio
from typing import Optional, Dict, Any, List

# ==================== НАСТРОЙКИ ====================
logging.basicConfig(level=logging.INFO)
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
    Гибридная система рекомендаций на основе Neo4j, Mistral AI 
    </div>
    """, unsafe_allow_html=True)


# ==================== СЕРВИСЫ ====================
@st.cache_resource
def init_services():
    """Инициализация всех сервисов"""
    try:
        from config import settings
        from src.database.neo4j_client import Neo4jClient
        from src.ai.embeddings import EmbeddingService
        from src.services.user_service import UserService
        from src.services.vacancy_service import VacancyService
        from src.services.feedback_service import FeedbackService
        from src.parsers.hh_parser import HHParser

        neo4j_client = Neo4jClient()
        if not neo4j_client.execute_query("RETURN 'Connected' AS status"):
            raise ConnectionError("Не удалось подключиться к Neo4j")

        neo4j_client.initialize_database()
        embedding_service = EmbeddingService()

        return {
            'neo4j': neo4j_client,
            'embedding': embedding_service,
            'user_service': UserService(neo4j_client, embedding_service),
            'vacancy_service': VacancyService(neo4j_client, embedding_service),
            'feedback_service': FeedbackService(neo4j_client, UserService(neo4j_client, embedding_service)),
            'parser': HHParser()
        }
    except ImportError as e:
        logger.error(f"Ошибка импорта: {e}")
        st.error(f"Ошибка импорта модулей: {e}")
    except Exception as e:
        logger.error(f"Ошибка инициализации: {e}")
        st.error(f"Не удалось инициализировать сервисы: {e}")
    return None


# ==================== КОМПОНЕНТЫ ВАКАНСИЙ ====================
def render_vacancy_card(vacancy, user, context="search"):
    """Универсальный компонент отображения вакансии"""
    from src.database.models import UserFeedback, FeedbackType

    st.markdown('<div class="vacancy-card">', unsafe_allow_html=True)

    # Заголовок и зарплата
    col_title, col_salary = st.columns([3, 1])
    with col_title:
        st.markdown(f"#### {vacancy.title or 'Без названия'}")
        company_info = []
        if vacancy.company_name: company_info.append(f"🏢 {vacancy.company_name}")
        if vacancy.location_name: company_info.append(f"📍 {vacancy.location_name}")
        if vacancy.experience: company_info.append(f"🎓 {vacancy.experience}")
        st.markdown(" • ".join(company_info) or "ℹ️ Информация о компании не указана")

    with col_salary:
        if vacancy.salary_from or vacancy.salary_to:
            salary_parts = []
            if vacancy.salary_from: salary_parts.append(f"от {vacancy.salary_from:,}")
            if vacancy.salary_to: salary_parts.append(f"до {vacancy.salary_to:,}")
            if vacancy.currency: salary_parts.append(vacancy.currency)
            st.markdown(f"**{' - '.join(salary_parts)}**")
        else:
            st.markdown("💰 Зарплата не указана")

    # Навыки
    if vacancy.skills:
        st.markdown("**Требуемые навыки:**")
        cols = st.columns(5)
        for i, skill in enumerate(vacancy.skills[:10]):
            if skill:
                with cols[i % 5]:
                    st.markdown(f'<div class="skill-tag">{skill}</div>', unsafe_allow_html=True)

    # Краткое описание
    if vacancy.description and len(vacancy.description) > 100:
        with st.expander("📋 Краткое описание"):
            preview = vacancy.description[:500] + "..." if len(vacancy.description) > 500 else vacancy.description
            st.markdown(preview)

    # Кнопки обратной связи
    st.markdown("---")
    col_like, col_dislike, col_view, col_apply = st.columns(4)

    feedback_actions = [
        ("👍 Нравится", FeedbackType.LIKE, col_like),
        ("👎 Не нравится", FeedbackType.DISLIKE, col_dislike),
        ("👁️ Подробнее", FeedbackType.VIEW, col_view),
        ("📨 Откликнуться", FeedbackType.APPLY, col_apply)
    ]

    for label, fb_type, column in feedback_actions:
        with column:
            if st.button(label, key=f"{context}_{fb_type}_{vacancy.id}", use_container_width=True):
                feedback = UserFeedback(user_id=user.id, vacancy_id=vacancy.id, feedback_type=fb_type)
                if services['feedback_service'].record_feedback(feedback):
                    st.success("✅ Спасибо за оценку!")
                    update_feedback_history()
                    if fb_type == FeedbackType.VIEW:
                        with st.expander("📋 Полное описание", expanded=True):
                            st.markdown(vacancy.description or "Описание отсутствует")
                    st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


def filter_vacancies(vacancies, min_salary, experience_filter, show_only_new):
    """Фильтрация списка вакансий"""
    filtered = []
    for vacancy in vacancies:
        if not vacancy:
            continue

        # Фильтр по зарплате
        if min_salary > 0:
            if vacancy.salary_to and vacancy.salary_to < min_salary:
                if vacancy.salary_from and vacancy.salary_from < min_salary:
                    continue

        # Фильтр по новизне
        if show_only_new and vacancy.published_at:
            published = vacancy.published_at
            if hasattr(published, 'tzinfo') and published.tzinfo is not None:
                published = published.replace(tzinfo=None)
            if datetime.now() - published > timedelta(days=30):
                continue

        filtered.append(vacancy)
    return filtered


# ==================== ОБНОВЛЕНИЕ ДАННЫХ ====================
def update_feedback_history():
    """Обновление истории обратной связи"""
    if st.session_state.current_user:
        history = services['feedback_service'].get_user_feedback_history(
            st.session_state.current_user.id, 20
        )
        st.session_state.feedback_history = history


@st.cache_data(ttl=300)
def get_system_stats():
    """Получение статистики системы с кэшированием"""
    try:
        return {
            'user_count': services['neo4j'].execute_query("MATCH (u:User) RETURN COUNT(u) AS count")[0]['count'],
            'vacancy_count': services['neo4j'].execute_query("MATCH (v:Vacancy) RETURN COUNT(v) AS count")[0]['count'],
            'skill_count': services['neo4j'].execute_query("MATCH (s:Skill) RETURN COUNT(s) AS count")[0]['count']
        }
    except Exception as e:
        logger.warning(f"Ошибка статистики: {e}")
        return None


# ==================== СТРАНИЦА ПРОФИЛЯ ====================
def render_profile_page():
    """Страница управления профилем"""
    st.markdown('<h2 class="sub-header">👤 Управление профилем</h2>', unsafe_allow_html=True)

    col_select, col_create = st.columns(2)

    # Загрузка профиля
    with col_select:
        st.markdown("### 📂 Загрузить существующий профиль")
        search_term = st.text_input("Поиск по имени", placeholder="Введите имя пользователя...", key="profile_search")

        if st.button("🔍 Поиск пользователей") or search_term:
            query = """
            MATCH (u:User)
            WHERE toLower(u.username) CONTAINS toLower($search)
            RETURN u.id AS id, u.username AS username
            ORDER BY u.username
            LIMIT 20
            """ if search_term else """
            MATCH (u:User) RETURN u.id AS id, u.username AS username
            ORDER BY u.username LIMIT 20
            """

            users = services['neo4j'].execute_query(query, {'search': search_term})
            if users:
                for user_data in users:
                    col_user, col_btn = st.columns([3, 1])
                    with col_user:
                        st.write(f"**{user_data['username']}**")
                        st.caption(f"ID: {user_data['id']}")
                    with col_btn:
                        if st.button("📥 Загрузить", key=f"load_{user_data['id']}"):
                            loaded_user = services['user_service'].get_user_by_id(user_data['id'])
                            if loaded_user:
                                st.session_state.current_user = loaded_user
                                st.success(f"✅ Профиль {loaded_user.username} загружен!")
                                update_feedback_history()
                                st.rerun()

    # Создание профиля
    with col_create:
        st.markdown("### 🆕 Создать новый профиль")
        with st.form("create_user_form", clear_on_submit=True):
            username = st.text_input("Имя пользователя*", placeholder="john_doe")
            skills_input = st.text_area("Навыки через запятую*", placeholder="Python, SQL...", height=100)
            resume_text = st.text_area("Резюме*", placeholder="Опытный разработчик...", height=150)

            if st.form_submit_button("✅ Создать профиль", type="primary"):
                if not (username and skills_input and resume_text):
                    st.error("⚠️ Заполните все обязательные поля")
                else:
                    skills = [s.strip() for s in skills_input.split(',') if s.strip()]
                    user_id = f"user_{int(datetime.now().timestamp())}"

                    from src.database.models import User
                    new_user = User(id=user_id, username=username, resume_text=resume_text, skills=skills)

                    with st.spinner("🔄 Сохранение профиля..."):
                        if services['user_service'].create_or_update_user(new_user):
                            st.session_state.current_user = new_user
                            st.success(f"🎉 Профиль {username} создан!")
                            update_feedback_history()
                            st.rerun()

    # Отображение текущего профиля
    if st.session_state.current_user:
        user = st.session_state.current_user
        st.markdown("---")
        st.markdown('<h2 class="sub-header">📋 Текущий профиль</h2>', unsafe_allow_html=True)

        col_info, col_stats = st.columns([2, 1])
        with col_info:
            st.markdown(f"### {user.username}")
            st.markdown(f"**ID:** `{user.id}`")
            if user.resume_text:
                with st.expander("📄 Просмотреть резюме"):
                    st.write(user.resume_text)

        with col_stats:
            st.metric("🔧 Навыки", len(user.skills))
            if user.preferences:
                active_prefs = len([v for v in user.preferences.values() if v > 0.1])
                st.metric("⭐ Предпочтения", active_prefs)

        if user.skills:
            st.markdown("#### 🔧 Навыки")
            cols = st.columns(4)
            for i, skill in enumerate(user.skills):
                with cols[i % 4]:
                    st.markdown(f'<div class="skill-tag">{skill}</div>', unsafe_allow_html=True)


# ==================== СТРАНИЦА ПОИСКА ====================
def render_search_page():
    """Страница поиска вакансий"""
    st.markdown('<h2 class="sub-header">🔍 Поиск и анализ вакансий</h2>', unsafe_allow_html=True)

    if not st.session_state.current_user:
        st.warning("⚠️ Сначала создайте профиль")
        st.stop()

    # Тестирование подключения
    with st.expander("🔧 Проверка подключения к HH.ru"):
        if st.button("🔄 Проверить подключение"):
            try:
                if services['parser'].test_connection():
                    st.success("✅ Подключение успешно!")
                else:
                    st.warning("⚠️ Проблемы с подключением")
            except Exception as e:
                st.error(f"❌ Ошибка: {e}")

    # Параметры поиска
    col_search, col_settings = st.columns([3, 1])
    with col_search:
        search_query = st.text_input("🔍 Поисковый запрос", value="Python разработчик",
                                     placeholder="Введите должность, технологию или компанию...")
    with col_settings:
        limit = st.slider("📊 Количество", 5, 30, 150)

    # Поиск вакансий
    if st.button("🚀 Начать поиск", type="primary", use_container_width=True):
        if not search_query.strip():
            st.error("⚠️ Введите поисковый запрос")
        else:
            with st.spinner("🔎 Ищем вакансии..."):
                try:
                    @st.cache_data(ttl=300)
                    def get_vacancy_ids(query: str, limit: int):
                        parser = services['parser']
                        pages = (limit + 100 - 1) // 100
                        ids = []
                        for page in range(pages):
                            remaining = limit - len(ids)
                            if remaining <= 0: break
                            items = parser.search_vacancies(text=query, per_page=min(100, remaining), page=page)
                            ids.extend(item['id'] for item in items if len(ids) < limit)
                        return ids

                    vacancy_ids = get_vacancy_ids(search_query, limit)
                    if not vacancy_ids:
                        st.warning("😕 Вакансий не найдено")
                    else:
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        async def load_with_progress():
                            results = []
                            parser = services['parser']
                            batches = [vacancy_ids[i:i + 10] for i in range(0, len(vacancy_ids), 10)]

                            for i, batch in enumerate(batches):
                                batch_results = await parser.fetch_and_parse_vacancies_async(batch)
                                results.extend(batch_results)
                                progress = int((len(results) / len(vacancy_ids)) * 100)
                                progress_bar.progress(progress)
                                status_text.text(f"Загружено {len(results)}/{len(vacancy_ids)}...")

                            return results

                        detailed_vacancies = asyncio.run(load_with_progress())
                        st.session_state.search_results = detailed_vacancies
                        st.success(f"✅ Загружено {len(detailed_vacancies)} вакансий!")

                        # Сохранение в базу
                        saved_count = sum(
                            1 for vac in detailed_vacancies if services['vacancy_service'].save_vacancy(vac))
                        if saved_count > 0:
                            st.info(f"💾 Сохранено {saved_count} вакансий")

                except Exception as e:
                    st.error(f"❌ Ошибка поиска: {e}")
                    logger.error(f"Search error: {e}")

    # Отображение результатов
    if st.session_state.search_results:
        st.markdown(f"### 📄 Результаты поиска ({len(st.session_state.search_results)})")

        # Фильтры
        col1, col2, col3 = st.columns(3)
        with col1:
            min_salary = st.number_input("💰 Мин. зарплата", min_value=0, value=50000, step=10000)
        with col2:
            show_only_new = st.checkbox("🆕 Только новые", value=True)

        # Фильтрация и отображение
        filtered = filter_vacancies(st.session_state.search_results, min_salary, "", show_only_new)

        if filtered:
            for vacancy in filtered:
                render_vacancy_card(vacancy, st.session_state.current_user, "search")
        else:
            st.info("📭 Нет вакансий, соответствующих фильтрам")
    elif st.session_state.search_results == []:
        st.info("🔍 Начните поиск вакансий")


# ==================== СТРАНИЦА РЕКОМЕНДАЦИЙ ====================
# ==================== СТРАНИЦА РЕКОМЕНДАЦИЙ ====================
def render_recommendations_page():
    """Страница персональных рекомендаций"""
    st.markdown('<h2 class="sub-header">🎯 Персональные рекомендации</h2>', unsafe_allow_html=True)

    if not st.session_state.current_user:
        st.warning("⚠️ Сначала создайте профиль")
        st.stop()

    user = st.session_state.current_user  # Добавляем переменную пользователя

    # Настройки
    from config import settings
    col1, col2, col3 = st.columns(3)
    with col1:
        num_rec = st.slider("📊 Количество", 3, 20, 8)
    with col2:
        content_weight = st.slider("📝 Контентный вес", 0.0, 1.0, settings.content_weight, 0.05)
    with col3:
        semantic_weight = st.slider("🧠 Семантический вес", 0.0, 1.0, settings.semantic_weight, 0.05)

    # Получение рекомендаций
    if st.button("🚀 Получить рекомендации", type="primary", use_container_width=True):
        with st.spinner("🧠 Анализируем предпочтения..."):
            try:
                recommendations = services['vacancy_service'].get_recommendations(user.id, num_rec)
                st.session_state.recommendations = recommendations
                st.success(
                    f"✅ Найдено {len(recommendations)} рекомендаций!" if recommendations else "📭 Рекомендаций не найдено")
            except Exception as e:
                st.error(f"❌ Ошибка: {e}")
                logger.error(f"Recommendation error: {e}")

    # Отображение рекомендаций
    if st.session_state.recommendations:
        # Визуализация оценок
        scores_data = []
        for rec in st.session_state.recommendations:
            scores_data.append({
                'Вакансия': rec.vacancy.title[:40] + ('...' if len(rec.vacancy.title) > 40 else ''),
                'Контентный': rec.content_score,
                'Графовый': rec.graph_score,
                'Семантический': rec.semantic_score,
                'total': rec.total_score
            })

        scores_df = pd.DataFrame(scores_data).sort_values('total', ascending=True)
        fig = px.bar(scores_df, x=['Контентный', 'Графовый', 'Семантический'], y='Вакансия',
                     title="📊 Распределение оценок", orientation='h', barmode='stack',
                     color_discrete_sequence=['#3B82F6', '#10B981', '#8B5CF6'])
        fig.update_layout(height=400, showlegend=True, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

        # Детали рекомендаций
        st.markdown("### 📋 Детали рекомендаций")
        for i, rec in enumerate(st.session_state.recommendations, 1):
            vacancy = rec.vacancy

            with st.expander(f"{i}. {vacancy.title} | 🎯 Score: {rec.total_score:.3f}", expanded=i <= 3):
                col_details, col_scores = st.columns([3, 1])

                with col_details:
                    # Отображение информации о вакансии
                    info = []
                    if vacancy.company_name:
                        info.append(f"**🏢 Компания:** {vacancy.company_name}")
                    if vacancy.location_name:
                        info.append(f"**📍 Локация:** {vacancy.location_name}")
                    if vacancy.experience:
                        info.append(f"**🎓 Опыт:** {vacancy.experience}")
                    if vacancy.employment:
                        info.append(f"**💼 Занятость:** {vacancy.employment}")

                    for line in info:
                        st.markdown(line)

                    # Зарплата
                    if vacancy.salary_from or vacancy.salary_to:
                        salary_text = "**💰 Зарплата:** "
                        if vacancy.salary_from:
                            salary_text += f"от {vacancy.salary_from:,}"
                        if vacancy.salary_to:
                            if vacancy.salary_from:
                                salary_text += " - "
                            salary_text += f"до {vacancy.salary_to:,}"
                        if vacancy.currency:
                            salary_text += f" {vacancy.currency}"
                        st.markdown(salary_text)

                    # Навыки с маркировкой совпадений
                    if vacancy.skills:
                        st.markdown("**🔧 Требуемые навыки:**")
                        skill_match = []
                        user_skills_set = set(user.skills)

                        for skill in vacancy.skills[:15]:
                            if skill in user_skills_set:
                                skill_match.append(f"✅ **{skill}**")
                            else:
                                skill_match.append(f"❌ {skill}")

                        cols = st.columns(3)
                        for j, skill_item in enumerate(skill_match):
                            with cols[j % 3]:
                                st.markdown(skill_item)

                with col_scores:
                    # Визуализация score
                    st.metric("🎯 Общий score", f"{rec.total_score:.3f}")

                    # Progress bar для общего score
                    st.progress(min(rec.total_score, 1.0))

                    # Детальные scores
                    st.markdown("**📊 Компоненты:**")
                    st.markdown(f"📝 Контентный: `{rec.content_score:.3f}`")
                    st.markdown(f"🕸️ Графовый: `{rec.graph_score:.3f}`")
                    st.markdown(f"🧠 Семантический: `{rec.semantic_score:.3f}`")

                    # Кнопки обратной связи для рекомендаций
                    st.markdown("---")

                    from src.database.models import UserFeedback, FeedbackType

                    col_like_small, col_dislike_small = st.columns(2)

                    with col_like_small:
                        if st.button("👍", key=f"rec_like_{vacancy.id}", use_container_width=True,
                                     help="Нравится вакансия"):
                            feedback = UserFeedback(
                                user_id=user.id,
                                vacancy_id=vacancy.id,
                                feedback_type=FeedbackType.LIKE
                            )
                            if services['feedback_service'].record_feedback(feedback):
                                st.success("✅ Спасибо! Учтем ваши предпочтения")
                                update_feedback_history()
                                st.rerun()

                    with col_dislike_small:
                        if st.button("👎", key=f"rec_dislike_{vacancy.id}", use_container_width=True,
                                     help="Не нравится вакансия"):
                            feedback = UserFeedback(
                                user_id=user.id,
                                vacancy_id=vacancy.id,
                                feedback_type=FeedbackType.DISLIKE
                            )
                            if services['feedback_service'].record_feedback(feedback):
                                st.success("✅ Спасибо! Исключим из рекомендаций")
                                update_feedback_history()
                                st.rerun()

                    # Кнопка для просмотра полного описания
                    if st.button("👁️ Полное описание", key=f"rec_view_{vacancy.id}",
                                 use_container_width=True):
                        feedback = UserFeedback(
                            user_id=user.id,
                            vacancy_id=vacancy.id,
                            feedback_type=FeedbackType.VIEW
                        )
                        services['feedback_service'].record_feedback(feedback)
                        with st.expander("📋 Полное описание вакансии", expanded=True):
                            st.markdown(vacancy.description if vacancy.description else "Описание отсутствует")


# ==================== СТРАНИЦА АНАЛИТИКИ ====================
def render_analytics_page():
    """Страница аналитики"""
    st.markdown('<h2 class="sub-header">📊 Аналитика системы</h2>', unsafe_allow_html=True)

    if not st.session_state.current_user:
        st.warning("⚠️ Загрузите профиль")
        st.stop()

    update_feedback_history()

    # Статистика пользователя
    st.markdown("### 👤 Ваша статистика")
    feedback_stats = services['neo4j'].execute_query("""
    MATCH (u:User {id: $user_id})-[r]->(:Vacancy)
    RETURN 
        COUNT(CASE WHEN type(r) = 'LIKED' THEN 1 END) AS likes,
        COUNT(CASE WHEN type(r) = 'DISLIKED' THEN 1 END) AS dislikes,
        COUNT(CASE WHEN type(r) = 'VIEWED' THEN 1 END) AS views,
        COUNT(CASE WHEN type(r) = 'APPLIED' THEN 1 END) AS applies
    """, {'user_id': st.session_state.current_user.id})

    if feedback_stats:
        stats = feedback_stats[0]
        cols = st.columns(4)
        metrics = [("👍 Лайков", 'likes'), ("👎 Дизлайков", 'dislikes'),
                   ("👁️ Просмотров", 'views'), ("📨 Откликов", 'applies')]

        for (label, key), col in zip(metrics, cols):
            with col:
                st.metric(label, stats[key])

    # История действий
    if st.session_state.feedback_history:
        history_data = []
        for item in st.session_state.feedback_history:
            title = item.get('vacancy_title', '')
            if isinstance(title, str):
                title = title[:50]
            history_data.append({
                'Дата': item.get('timestamp'),
                'Тип': item.get('feedback_type'),
                'Вакансия': title
            })

        df_history = pd.DataFrame(history_data)
        if 'Дата' in df_history.columns:
            df_history['Дата'] = pd.to_datetime(df_history['Дата'])
            df_history = df_history.sort_values('Дата', ascending=False)
            st.dataframe(df_history, use_container_width=True)

    # Статистика системы
    st.markdown("### 🏢 Статистика системы")
    stats = get_system_stats()
    if stats:
        cols = st.columns(3)
        with cols[0]: st.metric("👥 Пользователи", stats['user_count'])
        with cols[1]: st.metric("💼 Вакансии", stats['vacancy_count'])
        with cols[2]: st.metric("🔧 Навыки", stats['skill_count'])


# ==================== СТРАНИЦА НАСТРОЕК ====================
def render_settings_page():
    """Страница настроек"""
    st.markdown('<h2 class="sub-header">⚙️ Настройки системы</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🎯 Настройки рекомендаций")
        from config import settings

        content_weight = st.slider("Вес контентной фильтрации", 0.0, 1.0, settings.content_weight, 0.05)
        graph_weight = st.slider("Вес графовой фильтрации", 0.0, 1.0, settings.graph_weight, 0.05)
        semantic_weight = st.slider("Вес семантической фильтрации", 0.0, 1.0, settings.semantic_weight, 0.05)

        total = content_weight + graph_weight + semantic_weight
        if abs(total - 1.0) > 0.01:
            st.warning(f"Сумма весов: {total:.2f} (должна быть 1.0)")
        else:
            settings.content_weight, settings.graph_weight, settings.semantic_weight = content_weight, graph_weight, semantic_weight
            st.success("✅ Веса обновлены")

    with col2:
        st.markdown("### 🔧 Утилиты")
        if st.button("🗑️ Очистить кэш"):
            st.cache_resource.clear()
            st.success("✅ Кэш очищен")
            st.rerun()

        if st.button("🔄 Перезагрузить сервисы"):
            init_services.clear()
            st.success("✅ Сервисы перезагружены")
            st.rerun()


# ==================== САЙДБАР ====================
def render_sidebar():
    """Отображение боковой панели"""
    st.sidebar.image("C:/Users/0\PycharmProjects\PythonProjectстатья\LOGOTIP-rasshifrovka-fioletovyy-rus.png", width=280)
    st.sidebar.image("C:/Users/0\PycharmProjects\PythonProjectстатья/1.jpg",
                     width=280)
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

        if st.session_state.current_user.skills:
            st.sidebar.markdown("**Навыки:**")
            for skill in st.session_state.current_user.skills[:3]:
                st.sidebar.markdown(f"• {skill}")

        if st.sidebar.button("🚪 Выйти из профиля"):
            st.session_state.current_user = None
            st.session_state.recommendations = []
            st.session_state.search_results = []
            st.rerun()

    # Статистика системы
    st.sidebar.image("C:/Users/0\PycharmProjects\PythonProjectстатья\9310f63baa64b4591f85d5d8978f1466.jpg",
                     width=280)

    st.sidebar.image("C:/Users/0\PycharmProjects\PythonProjectстатья\d95f3135c3ee5f6f7c08b17753a8dcea.png",
                     width=280)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Статистика системы")
    stats = get_system_stats()
    if stats:
        col1, col2 = st.sidebar.columns(2)
        with col1: st.metric("👥 Пользователи", stats['user_count'])
        with col2: st.metric("💼 Вакансии", stats['vacancy_count'])
        st.sidebar.metric("🔧 Навыки", stats['skill_count'])

    return menu_options[selected]






# ==================== ОСНОВНОЙ КОД ====================
if __name__ == "__main__":
    # Инициализация
    setup_session_state()
    render_header()

    # Загрузка сервисов
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
        <p>Архитектура: Neo4j + Mistral AI + Гибридные алгоритмы Матвеев А.В.</p>
    </div>
    """, unsafe_allow_html=True)


    # Очистка ресурсов
    def cleanup():
        if services and 'neo4j' in services:
            services['neo4j'].close()
            logger.info("Neo4j connection closed")


    atexit.register(cleanup)