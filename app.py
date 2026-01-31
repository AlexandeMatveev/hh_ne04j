import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import numpy as np
import atexit
import logging
import asyncio

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Настройка страницы
st.set_page_config(
    page_title="AI Рекомендательная Система Вакансий",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Стили CSS для улучшения внешнего вида
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3B82F6;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #3B82F6;
        margin-bottom: 1rem;
    }
    .vacancy-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    .vacancy-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    .skill-tag {
        background-color: #E0F2FE;
        color: #0369A1;
        padding: 0.2rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
        display: inline-block;
        margin: 0.2rem;
    }
    .feedback-button {
        margin: 0.2rem;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Заголовок приложения
st.markdown('<h1 class="main-header">💼 AI Рекомендательная Система Вакансий</h1>', unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; margin-bottom: 2rem; color: #64748B;'>
Гибридная система рекомендаций на основе Neo4j, Mistral AI и обучения с подкреплением
</div>
""", unsafe_allow_html=True)


# Инициализация сервисов
@st.cache_resource
def init_services():
    """Инициализация всех сервисов с обработкой ошибок"""
    try:
        from config import settings
        from src.database.neo4j_client import Neo4jClient
        from src.ai.embeddings import EmbeddingService
        from src.services.user_service import UserService
        from src.services.vacancy_service import VacancyService
        from src.services.feedback_service import FeedbackService
        from src.parsers.hh_parser import HHParser

        # Клиент Neo4j
        neo4j_client = Neo4jClient()

        # Проверка подключения
        test_result = neo4j_client.execute_query("RETURN 'Connected' AS status")
        if not test_result:
            raise ConnectionError("Не удалось подключиться к Neo4j")

        neo4j_client.initialize_database()

        # Сервис эмбеддингов
        embedding_service = EmbeddingService()

        # Сервисы
        user_service = UserService(neo4j_client, embedding_service)
        vacancy_service = VacancyService(neo4j_client, embedding_service)
        feedback_service = FeedbackService(neo4j_client, user_service)

        # Парсер
        parser = HHParser()

        logger.info("Все сервисы успешно инициализированы")

        return {
            'neo4j': neo4j_client,
            'embedding': embedding_service,
            'user_service': user_service,
            'vacancy_service': vacancy_service,
            'feedback_service': feedback_service,
            'parser': parser
        }

    except ImportError as e:
        logger.error(f"Ошибка импорта модулей: {e}")
        st.error(f"Ошибка импорта модулей. Проверьте структуру проекта: {e}")
        return None
    except Exception as e:
        logger.error(f"Ошибка инициализации сервисов: {e}")
        st.error(f"Не удалось инициализировать сервисы: {e}")
        return None


# Инициализация
services = init_services()

if not services:
    st.error("""
    ⚠️ **Не удалось инициализировать сервисы.**  

    **Проверьте:**
    1. Запущен ли Neo4j (bolt://localhost:7687)
    2. Правильность пароля в файле .env
    3. Структуру проекта и наличие всех файлов

    **Быстрое решение:**
    ```bash
    # Запустите Neo4j в Docker:
    docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password123 -d neo4j:latest

    # Или установите Neo4j Desktop
    ```
    """)
    st.stop()

# Состояние сессии
if 'current_user' not in st.session_state:
    st.session_state.current_user = None
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = []
if 'search_results' not in st.session_state:
    st.session_state.search_results = []
if 'feedback_history' not in st.session_state:
    st.session_state.feedback_history = []


# Функция для обновления истории обратной связи
def update_feedback_history():
    if st.session_state.current_user:
        history = services['feedback_service'].get_user_feedback_history(
            st.session_state.current_user.id, 20
        )
        st.session_state.feedback_history = history


# Боковая панель навигации
st.sidebar.image("https://img.icons8.com/color/96/000000/parse-from-clipboard.png", width=80)
st.sidebar.title("🔍 Навигация")

# Меню навигации
menu_options = {
    "👤 Профиль": "profile",
    "🔍 Поиск вакансий": "search",
    "🎯 Рекомендации": "recommendations",
    "📊 Аналитика": "analytics",
    "⚙️ Настройки": "settings"
}

selected_menu = st.sidebar.radio(
    "Выберите раздел:",
    list(menu_options.keys()),
    index=0
)

# Отображение текущего пользователя в сайдбаре
if st.session_state.current_user:
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"### 👤 {st.session_state.current_user.username}")
    st.sidebar.info(f"ID: {st.session_state.current_user.id}")

    if st.session_state.current_user.skills:
        st.sidebar.markdown("**Навыки:**")
        for skill in st.session_state.current_user.skills[:3]:
            st.sidebar.markdown(f"• {skill}")
        if len(st.session_state.current_user.skills) > 3:
            st.sidebar.caption(f"и ещё {len(st.session_state.current_user.skills) - 3}...")

    if st.sidebar.button("🚪 Выйти из профиля"):
        st.session_state.current_user = None
        st.session_state.recommendations = []
        st.session_state.search_results = []
        st.rerun()

# Информация о системе
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Статистика системы")

try:
    # Получаем статистику
    user_count = services['neo4j'].execute_query("MATCH (u:User) RETURN COUNT(u) AS count")[0]['count']
    vacancy_count = services['neo4j'].execute_query("MATCH (v:Vacancy) RETURN COUNT(v) AS count")[0]['count']
    skill_count = services['neo4j'].execute_query("MATCH (s:Skill) RETURN COUNT(s) AS count")[0]['count']

    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("👥 Пользователи", user_count)
    with col2:
        st.metric("💼 Вакансии", vacancy_count)

    st.sidebar.metric("🔧 Навыки", skill_count)

except Exception as e:
    logger.warning(f"Не удалось загрузить статистику: {e}")

# ==================== СТРАНИЦА ПРОФИЛЯ ====================
if menu_options[selected_menu] == "profile":
    st.markdown('<h2 class="sub-header">👤 Управление профилем</h2>', unsafe_allow_html=True)

    # Две колонки для выбора и создания профиля
    col_select, col_create = st.columns(2)

    with col_select:
        st.markdown("### 📂 Загрузить существующий профиль")

        # Поиск пользователей
        search_term = st.text_input("Поиск по имени", placeholder="Введите имя пользователя...")

        if st.button("🔍 Поиск пользователей") or search_term:
            if search_term:
                query = """
                MATCH (u:User)
                WHERE toLower(u.username) CONTAINS toLower($search)
                RETURN u.id AS id, u.username AS username
                ORDER BY u.username
                LIMIT 20
                """
            else:
                query = """
                MATCH (u:User)
                RETURN u.id AS id, u.username AS username
                ORDER BY u.username
                LIMIT 20
                """

            users = services['neo4j'].execute_query(query, {'search': search_term})

            if users:
                st.markdown("#### Найденные пользователи:")
                for user in users:
                    col_user, col_btn = st.columns([3, 1])
                    with col_user:
                        st.write(f"**{user['username']}**")
                        st.caption(f"ID: {user['id']}")
                    with col_btn:
                        if st.button("📥 Загрузить", key=f"load_{user['id']}"):
                            loaded_user = services['user_service'].get_user_by_id(user['id'])
                            if loaded_user:
                                st.session_state.current_user = loaded_user
                                st.success(f"✅ Профиль {loaded_user.username} загружен!")
                                update_feedback_history()
                                st.rerun()
            else:
                st.info("👤 Пользователи не найдены")

    with col_create:
        st.markdown("### 🆕 Создать новый профиль")

        with st.form("create_user_form", clear_on_submit=True):
            username = st.text_input("Имя пользователя*", placeholder="john_doe")
            email = st.text_input("Email (опционально)", placeholder="john@example.com")

            st.markdown("#### 🎯 Навыки")
            skills_input = st.text_area(
                "Перечислите ваши навыки через запятую*",
                placeholder="Python, Machine Learning, SQL, Docker, FastAPI, ...",
                height=100
            )

            st.markdown("#### 📄 Резюме")
            resume_text = st.text_area(
                "Опишите ваш опыт и цели*",
                placeholder="Опытный Python разработчик с 5+ годами опыта...",
                height=150
            )

            submitted = st.form_submit_button("✅ Создать профиль", type="primary")

            if submitted:
                if not username:
                    st.error("⚠️ Имя пользователя обязательно")
                elif not skills_input:
                    st.error("⚠️ Укажите хотя бы один навык")
                elif not resume_text:
                    st.error("⚠️ Заполните резюме")
                else:
                    # Подготовка данных
                    skills = [skill.strip() for skill in skills_input.split(',') if skill.strip()]

                    # Создание ID
                    user_id = f"user_{int(datetime.now().timestamp())}"

                    # Создание объекта пользователя
                    from src.database.models import User

                    new_user = User(
                        id=user_id,
                        username=username,
                        resume_text=resume_text,
                        skills=skills
                    )

                    # Сохранение пользователя
                    with st.spinner("🔄 Сохранение профиля..."):
                        if services['user_service'].create_or_update_user(new_user):
                            st.session_state.current_user = new_user
                            st.success(f"🎉 Профиль {username} успешно создан!")
                            update_feedback_history()
                            st.rerun()
                        else:
                            st.error("❌ Ошибка при создании профиля")

    # Отображение текущего профиля
    if st.session_state.current_user:
        st.markdown("---")
        st.markdown('<h2 class="sub-header">📋 Текущий профиль</h2>', unsafe_allow_html=True)

        user = st.session_state.current_user

        # Основная информация
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

        # Навыки
        st.markdown("#### 🔧 Навыки")
        if user.skills:
            cols = st.columns(4)
            for i, skill in enumerate(user.skills):
                with cols[i % 4]:
                    st.markdown(f'<div class="skill-tag">{skill}</div>', unsafe_allow_html=True)
        else:
            st.info("📝 Навыки не указаны")

        # Предпочтения (если есть)
        if user.preferences and any(v > 0.1 for v in user.preferences.values()):
            st.markdown("#### ⭐ Веса предпочтений")
            prefs_data = [(k.replace('_', ' ').title(), v)
                          for k, v in user.preferences.items()
                          if v > 0.1]

            if prefs_data:
                prefs_df = pd.DataFrame(prefs_data, columns=['Навык', 'Вес'])
                prefs_df = prefs_df.sort_values('Вес', ascending=False)

                # Визуализация предпочтений
                fig = px.bar(prefs_df.head(10), x='Навык', y='Вес',
                             title="Топ-10 предпочтений",
                             color='Вес',
                             color_continuous_scale='Blues')
                st.plotly_chart(fig, use_container_width=True)

        # Кнопка редактирования
        if st.button("✏️ Редактировать профиль"):
            st.info("Функция редактирования в разработке")

# ==================== СТРАНИЦА ПОИСКА ВАКАНСИЙ ====================
# ==================== СТРАНИЦА ПОИСКА ВАКАНСИЙ ====================
elif menu_options[selected_menu] == "search":
    st.markdown('<h2 class="sub-header">🔍 Поиск и анализ вакансий</h2>', unsafe_allow_html=True)

    if not st.session_state.current_user:
        st.warning("""
        ⚠️ **Сначала создайте или загрузите профиль**  
        Перейдите на страницу "👤 Профиль" чтобы начать работу
        """)
        st.stop()

    user = st.session_state.current_user

    # Тестирование подключения к HH.ru
    with st.expander("🔧 Проверка подключения к HH.ru"):
        if st.button("🔄 Проверить подключение к HH.ru"):
            try:
                if services['parser'].test_connection():
                    st.success("✅ Подключение к HH.ru API успешно!")
                else:
                    st.warning("⚠️ Проблемы с подключением к HH.ru API")
            except Exception as e:
                st.error(f"❌ Ошибка проверки подключения: {e}")

    # Панель поиска
    col_search, col_settings = st.columns([3, 1])

    with col_search:
        search_query = st.text_input(
            "🔍 Поисковый запрос",
            value="Python разработчик",
            placeholder="Введите должность, технологию или компанию..."
        )

    with col_settings:
        limit = st.slider("📊 Количество", 5, 30, 150)
        area = st.selectbox("📍 Регион", ["Москва", "Санкт-Петербург", "Удалённо", "Все"], index=0)

    # Кнопка поиска
    # === ПОИСК ВАКАНСИЙ С КЭШЕМ И АСИНХРОННОСТЬЮ ===
    if st.button("🚀 Начать поиск", type="primary", use_container_width=True):
        if not search_query.strip():
            st.error("⚠️ Введите поисковый запрос")
        else:
            with st.spinner("🔎 Ищем вакансии на HH.ru..."):
                try:
                    # Шаг 1: Получаем ID вакансий через синхронный кэшируемый поиск
                    @st.cache_data(ttl=300)  # 5 минут
                    def get_vacancy_ids(query: str, limit: int):
                        parser = services['parser']
                        pages = (limit + 100 - 1) // 100
                        ids = []
                        for page in range(pages):
                            remaining = limit - len(ids)
                            if remaining <= 0:
                                break
                            per_page = min(100, remaining)
                            items = parser.search_vacancies(text=query, per_page=per_page, page=page)
                            for item in items:
                                if len(ids) < limit:
                                    ids.append(item['id'])
                            if len(items) < per_page:
                                break
                        return ids


                    vacancy_ids = get_vacancy_ids(search_query, limit)
                    st.session_state.search_results = []  # временно пусто

                    if not vacancy_ids:
                        st.warning("😕 По вашему запросу не найдено вакансий.")
                    else:
                        st.info(f"📥 Загружаем детали {len(vacancy_ids)} вакансий...")

                        # Шаг 2: Асинхронная загрузка деталей
                        parser = services['parser']

                        # Создаём progress bar
                        progress_bar = st.progress(0)
                        status_text = st.empty()


                        def update_progress(current, total):
                            progress = int((current / total) * 100)
                            progress_bar.progress(progress)
                            status_text.text(f"Загружено {current}/{total} вакансий...")


                        # Обёртка для запуска асинхронного кода
                        async def load_with_progress():
                            results = []
                            for i, batch_ids in enumerate(
                                    [vacancy_ids[i:i + 10] for i in range(0, len(vacancy_ids), 10)]):
                                batch_results = await parser.fetch_and_parse_vacancies_async(batch_ids)
                                results.extend(batch_results)
                                update_progress(len(results), len(vacancy_ids))
                            return results


                        # Запуск асинхронной загрузки
                        detailed_vacancies = asyncio.run(load_with_progress())

                        # Сохранение в сессию
                        st.session_state.search_results = detailed_vacancies
                        st.success(f"✅ Успешно загружено {len(detailed_vacancies)} вакансий!")

                        # Сохранение в базу (опционально)
                        saved_count = 0
                        for vac in detailed_vacancies:
                            if services['vacancy_service'].save_vacancy(vac):
                                saved_count += 1
                        if saved_count > 0:
                            st.info(f"💾 Сохранено {saved_count} вакансий в Neo4j")

                except Exception as e:
                    st.error(f"❌ Ошибка при поиске: {str(e)}")
                    logger.error(f"Search error: {e}", exc_info=True)

    # Отображение результатов поиска
    if st.session_state.search_results:
        st.markdown(f"### 📄 Результаты поиска ({len(st.session_state.search_results)})")

        # Фильтры
        col_filter1, col_filter2, col_filter3 = st.columns(3)
        with col_filter1:
            min_salary = st.number_input("💰 Мин. зарплата", min_value=0, value=50000, step=10000)
        with col_filter2:
            experience_filter = st.selectbox("🎓 Опыт", ["Любой", "Нет опыта", "1-3 года", "3-6 лет", "Более 6 лет"])
        with col_filter3:
            show_only_new = st.checkbox("🆕 Только новые", value=True)

        # Отображение вакансий
        displayed_count = 0

        for i, vacancy in enumerate(st.session_state.search_results):
            if not vacancy:
                continue

            # Фильтрация по зарплате
            if min_salary > 0:
                if vacancy.salary_to and vacancy.salary_to < min_salary:
                    if vacancy.salary_from and vacancy.salary_from < min_salary:
                        continue
                elif not vacancy.salary_from and not vacancy.salary_to:
                    pass  # Если зарплата не указана, пропускаем фильтр

            # Фильтрация по новизне
            if show_only_new and vacancy.published_at:
                from datetime import datetime, timedelta

                now = datetime.now()
                published = vacancy.published_at

                # Убираем временную зону если она есть
                if hasattr(published, 'tzinfo') and published.tzinfo is not None:
                    published = published.replace(tzinfo=None)

                # Проверяем что published не в будущем
                if published > now:
                    published = now - timedelta(days=1)

                if now - published > timedelta(days=30):
                    continue

            # Карточка вакансии
            with st.container():
                st.markdown(f'<div class="vacancy-card">', unsafe_allow_html=True)

                col_title, col_salary = st.columns([3, 1])

                with col_title:
                    st.markdown(f"#### {vacancy.title if vacancy.title else 'Без названия'}")

                    # Информация о компании и локации
                    company_info = []
                    if vacancy.company_name:
                        company_info.append(f"🏢 {vacancy.company_name}")
                    if vacancy.location_name:
                        company_info.append(f"📍 {vacancy.location_name}")
                    if vacancy.experience:
                        company_info.append(f"🎓 {vacancy.experience}")

                    if company_info:
                        st.markdown(" • ".join(company_info))
                    else:
                        st.markdown("ℹ️ Информация о компании не указана")

                with col_salary:
                    if vacancy.salary_from or vacancy.salary_to:
                        salary_display = ""
                        if vacancy.salary_from:
                            salary_display += f"от {vacancy.salary_from:,}"
                        if vacancy.salary_to:
                            if salary_display:
                                salary_display += " - "
                            salary_display += f"до {vacancy.salary_to:,}"
                        if vacancy.currency:
                            salary_display += f" {vacancy.currency}"

                        st.markdown(f"**{salary_display}**")
                    else:
                        st.markdown("💰 Зарплата не указана")

                # Навыки
                if vacancy.skills:
                    st.markdown("**Требуемые навыки:**")
                    skill_cols = st.columns(5)
                    for j, skill in enumerate(vacancy.skills[:10]):
                        if skill:
                            with skill_cols[j % 5]:
                                st.markdown(f'<div class="skill-tag">{skill}</div>', unsafe_allow_html=True)
                else:
                    st.markdown("🔧 Навыки не указаны")

                # Описание (кратко)
                if vacancy.description and len(vacancy.description) > 100:
                    with st.expander("📋 Краткое описание"):
                        st.markdown(vacancy.description[:500] + "..." if len(
                            vacancy.description) > 500 else vacancy.description)

                # Кнопки обратной связи
                st.markdown("---")
                col_like, col_dislike, col_view, col_apply = st.columns(4)

                from src.database.models import UserFeedback, FeedbackType

                with col_like:
                    if st.button("👍 Нравится", key=f"search_like_{vacancy.id}", use_container_width=True):
                        feedback = UserFeedback(
                            user_id=user.id,
                            vacancy_id=vacancy.id,
                            feedback_type=FeedbackType.LIKE
                        )
                        if services['feedback_service'].record_feedback(feedback):
                            st.success("✅ Спасибо за оценку!")
                            update_feedback_history()
                            st.rerun()

                with col_dislike:
                    if st.button("👎 Не нравится", key=f"search_dislike_{vacancy.id}", use_container_width=True):
                        feedback = UserFeedback(
                            user_id=user.id,
                            vacancy_id=vacancy.id,
                            feedback_type=FeedbackType.DISLIKE
                        )
                        if services['feedback_service'].record_feedback(feedback):
                            st.success("✅ Спасибо за оценку!")
                            update_feedback_history()
                            st.rerun()

                with col_view:
                    if st.button("👁️ Подробнее", key=f"search_view_{vacancy.id}", use_container_width=True):
                        feedback = UserFeedback(
                            user_id=user.id,
                            vacancy_id=vacancy.id,
                            feedback_type=FeedbackType.VIEW
                        )
                        services['feedback_service'].record_feedback(feedback)
                        with st.expander("📋 Полное описание", expanded=True):
                            st.markdown(vacancy.description if vacancy.description else "Описание отсутствует")

                with col_apply:
                    if st.button("📨 Откликнуться", key=f"search_apply_{vacancy.id}", use_container_width=True):
                        feedback = UserFeedback(
                            user_id=user.id,
                            vacancy_id=vacancy.id,
                            feedback_type=FeedbackType.APPLY
                        )
                        if services['feedback_service'].record_feedback(feedback):
                            st.success("🎉 Отклик отправлен!")
                            update_feedback_history()
                            st.rerun()

                st.markdown('</div>', unsafe_allow_html=True)

        if displayed_count == 0:
            st.info("📭 Нет вакансий, соответствующих фильтрам. Попробуйте изменить критерии поиска.")
    elif 'search_results' in st.session_state and st.session_state.search_results == []:
        st.info("🔍 Начните поиск вакансий, чтобы увидеть результаты здесь.")

# ==================== СТРАНИЦА РЕКОМЕНДАЦИЙ ====================
elif menu_options[selected_menu] == "recommendations":
    st.markdown('<h2 class="sub-header">🎯 Персональные рекомендации</h2>', unsafe_allow_html=True)

    if not st.session_state.current_user:
        st.warning("""
        ⚠️ **Сначала создайте или загрузите профиль**  
        Перейдите на страницу "👤 Профиль" чтобы начать работу
        """)
        st.stop()

    user = st.session_state.current_user

    # Настройки рекомендаций
    st.markdown("### ⚙️ Настройки рекомендаций")

    col_settings1, col_settings2, col_settings3 = st.columns(3)

    with col_settings1:
        num_recommendations = st.slider("📊 Количество рекомендаций", 3, 20, 8)

    with col_settings2:
        from config import settings

        content_weight = st.slider("📝 Контентный вес", 0.0, 1.0, settings.content_weight, 0.05)
        settings.content_weight = content_weight

    with col_settings3:
        semantic_weight = st.slider("🧠 Семантический вес", 0.0, 1.0, settings.semantic_weight, 0.05)
        settings.semantic_weight = semantic_weight
        settings.graph_weight = 1.0 - content_weight - semantic_weight

    # Кнопка получения рекомендаций
    if st.button("🚀 Получить рекомендации", type="primary", use_container_width=True):
        with st.spinner("🧠 Анализируем ваши предпочтения и ищем подходящие вакансии..."):
            try:
                recommendations = services['vacancy_service'].get_recommendations(
                    user.id, num_recommendations
                )
                st.session_state.recommendations = recommendations

                if recommendations:
                    st.success(f"✅ Найдено {len(recommendations)} персональных рекомендаций!")
                else:
                    st.info("📭 Рекомендаций не найдено. Попробуйте оценить больше вакансий")

            except Exception as e:
                st.error(f"❌ Ошибка при получении рекомендаций: {e}")
                logger.error(f"Recommendation error: {e}")

    # Отображение рекомендаций
    if st.session_state.recommendations:
        st.markdown(f"### 🏆 Топ-{len(st.session_state.recommendations)} рекомендаций")

        # Визуализация распределения оценок
        scores_data = []
        for rec in st.session_state.recommendations:
            scores_data.append({
                'Вакансия': rec.vacancy.title[:40] + ('...' if len(rec.vacancy.title) > 40 else ''),
                'Контентный': rec.content_score,
                'Графовый': rec.graph_score,
                'Семантический': rec.semantic_score,
                'total': rec.total_score
            })

        scores_df = pd.DataFrame(scores_data)
        scores_df = scores_df.sort_values('total', ascending=True)

        fig = px.bar(scores_df,
                     x=['Контентный', 'Графовый', 'Семантический'],
                     y='Вакансия',
                     title="📊 Распределение оценок по компонентам",
                     orientation='h',
                     barmode='stack',
                     color_discrete_sequence=['#3B82F6', '#10B981', '#8B5CF6'])

        fig.update_layout(
            height=400,
            showlegend=True,
            legend_title="Компоненты",
            yaxis={'categoryorder': 'total ascending'}
        )

        st.plotly_chart(fig, use_container_width=True)

        # Детализированный список рекомендаций
        st.markdown("### 📋 Детали рекомендаций")

        for i, rec in enumerate(st.session_state.recommendations, 1):
            vacancy = rec.vacancy

            with st.expander(
                    f"{i}. {vacancy.title} | 🎯 Score: {rec.total_score:.3f}",
                    expanded=i <= 3  # Первые 3 развернуты по умолчанию
            ):
                col_details, col_scores = st.columns([3, 1])

                with col_details:
                    # Основная информация
                    info_lines = []
                    if vacancy.company_name:
                        info_lines.append(f"**🏢 Компания:** {vacancy.company_name}")
                    if vacancy.location_name:
                        info_lines.append(f"**📍 Локация:** {vacancy.location_name}")
                    if vacancy.experience:
                        info_lines.append(f"**🎓 Опыт:** {vacancy.experience}")
                    if vacancy.employment:
                        info_lines.append(f"**💼 Занятость:** {vacancy.employment}")

                    for line in info_lines:
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

                    # Навыки
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

                    # Кнопки обратной связи
                    st.markdown("---")

                    from src.database.models import UserFeedback, FeedbackType

                    col_like_small, col_dislike_small = st.columns(2)

                    with col_like_small:
                        if st.button("👍", key=f"rec_like_{vacancy.id}", use_container_width=True):
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
                        if st.button("👎", key=f"rec_dislike_{vacancy.id}", use_container_width=True):
                            feedback = UserFeedback(
                                user_id=user.id,
                                vacancy_id=vacancy.id,
                                feedback_type=FeedbackType.DISLIKE
                            )
                            if services['feedback_service'].record_feedback(feedback):
                                st.success("✅ Спасибо! Исключим из рекомендаций")
                                update_feedback_history()
                                st.rerun()

# ==================== СТРАНИЦА АНАЛИТИКИ ====================
elif menu_options[selected_menu] == "analytics":
    st.markdown('<h2 class="sub-header">📊 Аналитика системы</h2>', unsafe_allow_html=True)

    if not st.session_state.current_user:
        st.warning("⚠️ Загрузите профиль для просмотра аналитики")
        st.stop()

    user = st.session_state.current_user

    # Обновляем историю обратной связи
    update_feedback_history()

    # Статистика пользователя
    st.markdown("### 👤 Ваша статистика")

    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)

    # Считаем статистику пользователя
    feedback_stats = services['neo4j'].execute_query("""
    MATCH (u:User {id: $user_id})-[r]->(:Vacancy)
    RETURN 
        COUNT(CASE WHEN type(r) = 'LIKED' THEN 1 END) AS likes,
        COUNT(CASE WHEN type(r) = 'DISLIKED' THEN 1 END) AS dislikes,
        COUNT(CASE WHEN type(r) = 'VIEWED' THEN 1 END) AS views,
        COUNT(CASE WHEN type(r) = 'APPLIED' THEN 1 END) AS applies
    """, {'user_id': user.id})

    if feedback_stats:
        stats = feedback_stats[0]

        with col_stat1:
            st.metric("👍 Лайков", stats['likes'])
        with col_stat2:
            st.metric("👎 Дизлайков", stats['dislikes'])
        with col_stat3:
            st.metric("👁️ Просмотров", stats['views'])
        with col_stat4:
            st.metric("📨 Откликов", stats['applies'])

    # История действий
        # История действий
        st.markdown("### 📜 История ваших действий")

        if st.session_state.feedback_history:
            history_data = []
            for item in st.session_state.feedback_history:
                # Безопасное извлечение названия вакансии
                vacancy_title = item.get('vacancy_title')
                if vacancy_title is None:
                    vacancy_title = ''
                elif isinstance(vacancy_title, str):
                    vacancy_title = vacancy_title[:50]
                else:
                    vacancy_title = str(vacancy_title)[:50]

                history_data.append({
                    'Дата': item.get('timestamp'),
                    'Тип': item.get('feedback_type'),
                    'Вакансия': vacancy_title
                })

            df_history = pd.DataFrame(history_data)

        if 'Дата' in df_history.columns:
            df_history['Дата'] = pd.to_datetime(df_history['Дата'])
            df_history = df_history.sort_values('Дата', ascending=False)

            # Визуализация активности
            df_history['Дата_день'] = df_history['Дата'].dt.date
            daily_activity = df_history.groupby('Дата_день').size().reset_index(name='Активность')

            fig_activity = px.line(daily_activity,
                                   x='Дата_день', y='Активность',
                                   title="📈 Активность по дням",
                                   markers=True)
            st.plotly_chart(fig_activity, use_container_width=True)

        st.dataframe(df_history, use_container_width=True)
    else:
        st.info("📭 История действий пуста. Начните оценивать вакансии!")

    # Анализ эффективности системы
    st.markdown("### 📈 Эффективность рекомендаций")

    # Симуляция роста точности (в реальной системе здесь были бы реальные данные)
    days = np.arange(1, 31)
    base_accuracy = 0.5
    improvement = 0.35
    learning_rate = 0.15

    # Симулируем разные сценарии
    scenario_slow = base_accuracy + improvement * (1 - np.exp(-days / 15))
    scenario_fast = base_accuracy + improvement * (1 - np.exp(-days / 7))
    scenario_ideal = base_accuracy + improvement * (1 - np.exp(-days / 10))

    accuracy_data = pd.DataFrame({
        'День': np.concatenate([days, days, days]),
        'Точность': np.concatenate([scenario_slow, scenario_fast, scenario_ideal]),
        'Сценарий': ['Медленное обучение'] * 30 + ['Быстрое обучение'] * 30 + ['Наша система'] * 30
    })

    fig_accuracy = px.line(accuracy_data,
                           x='День', y='Точность',
                           color='Сценарий',
                           title="🎯 Рост точности рекомендаций (симуляция)",
                           markers=True,
                           color_discrete_sequence=['#EF4444', '#F59E0B', '#10B981'])

    fig_accuracy.update_layout(
        yaxis_range=[0.4, 1.0],
        yaxis_tickformat=".0%",
        hovermode="x unified"
    )

    st.plotly_chart(fig_accuracy, use_container_width=True)

    # Статистика системы
    st.markdown("### 🏢 Статистика системы")

    try:
        system_stats = {}

        # Общая статистика
        total_stats = services['neo4j'].execute_query("""
        MATCH (u:User)
        WITH COUNT(u) AS user_count

        MATCH (v:Vacancy)
        WITH user_count, COUNT(v) AS vacancy_count

        MATCH (s:Skill)
        WITH user_count, vacancy_count, COUNT(s) AS skill_count

        MATCH ()-[r:LIKED|DISLIKED]->()
        RETURN user_count, vacancy_count, skill_count, COUNT(r) AS interaction_count
        """)

        if total_stats:
            stats = total_stats[0]

            col_sys1, col_sys2, col_sys3, col_sys4 = st.columns(4)

            with col_sys1:
                st.metric("👥 Всего пользователей", stats['user_count'])
            with col_sys2:
                st.metric("💼 Всего вакансий", stats['vacancy_count'])
            with col_sys3:
                st.metric("🔧 Всего навыков", stats['skill_count'])
            with col_sys4:
                st.metric("🔄 Взаимодействий", stats['interaction_count'])

        # Популярные навыки
        popular_skills = services['neo4j'].execute_query("""
        MATCH (v:Vacancy)-[:REQUIRES]->(s:Skill)
        RETURN s.name AS skill_name, COUNT(v) AS demand
        ORDER BY demand DESC
        LIMIT 10
        """)

        if popular_skills:
            st.markdown("#### 🏆 Самые востребованные навыки")

            skills_data = []
            for skill in popular_skills:
                skills_data.append({
                    'Навык': skill['skill_name'],
                    'Спрос': skill['demand']
                })

            df_skills = pd.DataFrame(skills_data)

            fig_skills = px.bar(df_skills,
                                x='Спрос', y='Навык',
                                orientation='h',
                                title="Топ-10 востребованных навыков",
                                color='Спрос',
                                color_continuous_scale='Blues')

            st.plotly_chart(fig_skills, use_container_width=True)

    except Exception as e:
        logger.warning(f"Ошибка при получении статистики системы: {e}")

# ==================== СТРАНИЦА НАСТРОЕК ====================
elif menu_options[selected_menu] == "settings":
    st.markdown('<h2 class="sub-header">⚙️ Настройки системы</h2>', unsafe_allow_html=True)

    col_set1, col_set2 = st.columns(2)

    with col_set1:
        st.markdown("### 🎯 Настройки рекомендаций")

        from config import settings

        content_weight = st.slider(
            "Вес контентной фильтрации",
            0.0, 1.0, settings.content_weight, 0.05,
            help="Влияние совпадения навыков пользователя и вакансии"
        )

        graph_weight = st.slider(
            "Вес графовой фильтрации",
            0.0, 1.0, settings.graph_weight, 0.05,
            help="Влияние поведения похожих пользователей"
        )

        semantic_weight = st.slider(
            "Вес семантической фильтрации",
            0.0, 1.0, settings.semantic_weight, 0.05,
            help="Влияние смысловой близости резюме и описания вакансии"
        )

        # Проверка суммы весов
        total_weight = content_weight + graph_weight + semantic_weight
        if abs(total_weight - 1.0) > 0.01:
            st.warning(f"Сумма весов должна быть равна 1.0 (сейчас: {total_weight:.2f})")
        else:
            settings.content_weight = content_weight
            settings.graph_weight = graph_weight
            settings.semantic_weight = semantic_weight
            st.success("✅ Веса обновлены")

    with col_set2:
        st.markdown("### 📚 Настройки обучения")

        learning_rate = st.slider(
            "Скорость обучения",
            0.01, 0.5, settings.learning_rate, 0.01,
            help="Скорость обновления предпочтений на основе обратной связи"
        )

        regularization = st.slider(
            "Регуляризация",
            0.0, 0.1, settings.regularization_lambda, 0.001,
            help="Предотвращение переобучения предпочтений"
        )

        settings.learning_rate = learning_rate
        settings.regularization_lambda = regularization

        st.markdown("### 🔧 Утилиты")

        if st.button("🗑️ Очистить кэш", help="Очистить кэшированные данные Streamlit"):
            st.cache_resource.clear()
            st.success("✅ Кэш очищен")
            st.rerun()

        if st.button("🔄 Перезагрузить сервисы", help="Перезагрузить все сервисы системы"):
            init_services.clear()
            st.success("✅ Сервисы перезагружены")
            st.rerun()

    # Информация о системе
    st.markdown("---")
    st.markdown("### ℹ️ Информация о системе")

    try:
        from config import settings
        import streamlit as st

        info_col1, info_col2 = st.columns(2)

        with info_col1:
            st.markdown("#### 📊 Конфигурация")
            st.markdown(f"- **Neo4j URI:** `{settings.neo4j_uri}`")
            st.markdown(f"- **Mistral AI:** {'✅ Настроен' if settings.mistral_api_key else '❌ Не настроен'}")
            st.markdown(f"- **Контентный вес:** `{settings.content_weight:.2f}`")
            st.markdown(f"- **Графовый вес:** `{settings.graph_weight:.2f}`")
            st.markdown(f"- **Семантический вес:** `{settings.semantic_weight:.2f}`")

        with info_col2:
            st.markdown("#### 🚀 Производительность")

            # Получаем статистику
            user_count = services['neo4j'].execute_query("MATCH (u:User) RETURN COUNT(u) AS count")[0]['count']
            vacancy_count = services['neo4j'].execute_query("MATCH (v:Vacancy) RETURN COUNT(v) AS count")[0]['count']

            st.markdown(f"- **Пользователей:** `{user_count}`")
            st.markdown(f"- **Вакансий:** `{vacancy_count}`")
            st.markdown(f"- **Скорость обучения:** `{settings.learning_rate:.3f}`")
            st.markdown(f"- **Регуляризация:** `{settings.regularization_lambda:.3f}`")

    except Exception as e:
        st.error(f"Ошибка при получении информации: {e}")

# Футер
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #64748B; font-size: 0.9rem; margin-top: 2rem;'>
    <p>💡 <strong>AI Рекомендательная Система Вакансий</strong> | Архитектура: Neo4j + Mistral AI + Гибридные алгоритмы</p>
    <p>📚 Гибридная рекомендательная система с обучением на основе обратной связи</p>
</div>
""", unsafe_allow_html=True)


# Очистка ресурсов при завершении
def cleanup():
    if services and 'neo4j' in services:
        services['neo4j'].close()
        logger.info("Neo4j connection closed")


atexit.register(cleanup)