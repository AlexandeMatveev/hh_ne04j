"""
Обновленный app.py для взаимодействия с FastAPI
Изменения: Streamlit делает HTTP запросы к FastAPI
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import atexit
import logging
import sys
from pathlib import Path
import requests

# Добавляем корень проекта в путь
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== НАСТРОЙКИ ====================
FASTAPI_BASE_URL = "http://localhost:8000"
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

# ==================== ГЛОБАЛЬНАЯ ПЕРЕМЕННАЯ ====================
services = None
fastapi_client = None


# ==================== FASTAPI CLIENT ====================
class FastAPIClient:
    """Клиент для взаимодействия с FastAPI"""
    
    def __init__(self, base_url: str = FASTAPI_BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
    
    def request(self, method: str, endpoint: str, **kwargs):
        """Выполнить HTTP запрос"""
        try:
            url = f"{self.base_url}{endpoint}"
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"FastAPI error: {e}")
            return None
    
    def get_vacancies(self, query: str = "", limit: int = 20):
        """Получить вакансии"""
        return self.request("GET", f"/api/v1/vacancies/search", 
                          params={"query": query, "limit": limit})
    
    def get_recommendations(self, user_id: str, top_n: int = 10):
        """Получить рекомендации"""
        return self.request("POST", "/api/v1/recommendations/", 
                          json={"user_id": user_id, "top_n": top_n})
    
    def add_feedback(self, user_id: str, vacancy_id: str, feedback_type: str):
        """Добавить обратную связь"""
        return self.request("POST", "/api/v1/feedback/", 
                          json={"user_id": user_id, "vacancy_id": vacancy_id, 
                                "feedback_type": feedback_type})
    
    def get_health(self):
        """Проверка здоровья"""
        return self.request("GET", "/health")


# ==================== УТИЛИТЫ ====================
def setup_session_state():
    """Инициализация состояния сессии"""
    defaults = {'current_user': None, 'recommendations': [], 
                'search_results': [], 'feedback_history': []}
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ==================== ИНИЦИАЛИЗАЦИЯ ====================
@st.cache_resource
def init_services():
    """Инициализация через FastAPI"""
    global services, fastapi_client
    services = {}
    
    try:
        logger.info("🚀 Инициализация FastAPI client...")
        fastapi_client = FastAPIClient()
        
        # Проверка соединения
        health = fastapi_client.get_health()
        if health:
            logger.info(f"✅ FastAPI connected: {health['status']}")
            services['fastapi'] = fastapi_client
            st.success(f"✅ Подключено к FastAPI: {health['status']}")
        else:
            logger.warning("⚠️ FastAPI connection failed")
            st.warning("⚠️ Не удалось подключиться к FastAPI")
        
        # Локальные сервисы (резервные)
        from src.database.neo4j_client import Neo4jClient
        from src.services.feedback_service import FeedbackService
        
        services['neo4j'] = Neo4jClient()
        services['neo4j'].connect()
        
        feedback_service = FeedbackService(services['neo4j'])
        feedback_service.init(services['neo4j'])
        services['feedback_service'] = feedback_service
        
        return services
        
    except Exception as e:
        logger.error(f"❌ Init error: {e}")
        st.error(f"Ошибка инициализации: {e}")
        return None


# ==================== ВАКАНСИИ ====================
def render_vacancy_card(vacancy, user, context="search"):
    """Отображение вакансии"""
    st.markdown('<div class="vacancy-card">', unsafe_allow_html=True)
    
    title = getattr(vacancy, 'title', None) or getattr(vacancy, 'title', 'Без названия')
    if isinstance(title, dict):
        title = title.get('title', 'Без названия')
    st.markdown(f"#### {title}")
    
    # Зарплата
    salary_from = getattr(vacancy, 'salary_from', None) or getattr(vacancy, 'salary_from', None)
    salary_to = getattr(vacancy, 'salary_to', None) or getattr(vacancy, 'salary_to', None)
    currency = getattr(vacancy, 'currency', None) or getattr(vacancy, 'salary_currency', 'RUB')
    
    if salary_from or salary_to:
        parts = []
        if salary_from: parts.append(f"от {salary_from:,}")
        if salary_to: parts.append(f"до {salary_to:,}")
        if currency: parts.append(currency)
        st.markdown(f"**{' - '.join(parts)}**")
    
    # Навыки
    skills = getattr(vacancy, 'skills', []) or getattr(vacancy, 'skills', []) or []
    if skills and isinstance(skills, list):
        st.markdown("**Навыки:**")
        cols = st.columns(5)
        for i, skill in enumerate(skills[:10]):
            if skill:
                with cols[i % 5]:
                    st.markdown(f'<div class="skill-tag">{skill}</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)


# ==================== ОСНОВНЫЕ ФУНКЦИИ ====================
def render_search_page():
    """Страница поиска"""
    st.markdown('<h2 class="sub-header">🔍 Поиск вакансий</h2>', unsafe_allow_html=True)
    
    # Статус подключения
    if not fastapi_client:
        st.error("❌ FastAPI не подключен!")
        st.info("⚠️ Убедитесь, что FastAPI запущен на порту 8000")
        return
    
    search_query = st.text_input("🔍 Поисковый запрос", value="Python разработчик")
    limit = st.slider("Количество", 5, 30, 15)
    
    if st.button("🚀 Искать"):
        if not search_query.strip():
            st.error("⚠️ Введите запрос")
            return
        
        with st.spinner("Ищем вакансии..."):
            vacancies = fastapi_client.get_vacancies(search_query, limit)
            if vacancies:
                st.session_state.search_results = vacancies
                st.success(f"✅ Загружено {len(vacancies)} вакансий")
            else:
                st.error("❌ Ошибка получения вакансий")


def render_recommendations_page():
    """Страница рекомендаций"""
    st.markdown('<h2 class="sub-header">🎯 Рекомендации</h2>', unsafe_allow_html=True)
    
    # Статус подключения
    if not fastapi_client:
        st.error("❌ FastAPI не подключен!")
        st.info("⚠️ Убедитесь, что FastAPI запущен на порту 8000")
        return
    
    if not st.session_state.current_user:
        st.warning("⚠️ Загрузите профиль")
        return
    
    user_id = st.session_state.current_user.id
    num_rec = st.slider("Количество", 3, 20, 5)
    
    if st.button("🚀 Получить рекомендации"):
        with st.spinner("Анализируем..."):
            recommendations = fastapi_client.get_recommendations(user_id, num_rec)
            if recommendations:
                st.session_state.recommendations = recommendations
                st.success(f"✅ Найдено {len(recommendations)} рекомендаций")
            else:
                st.error("❌ Ошибка рекомендаций")


def render_feedback():
    """Обработка обратной связи через FastAPI"""
    if fastapi_client:
        return fastapi_client.add_feedback(
            st.session_state.current_user.id,

        )
    return False


# ==================== MAIN ====================
def main():
    setup_session_state()
    
    global services
    services = init_services()
    
    if not services:
        st.error("❌ Сервисы не инициализированы")
        return
    
    st.markdown('<h1 class="main-header">💼 AI Рекомендательная Система Вакансий</h1>', unsafe_allow_html=True)
    
    menu = st.sidebar.radio("Раздел", ["Поиск", "Рекомендации", "Профиль"])
    
    if menu == "Поиск":
        render_search_page()
    elif menu == "Рекомендации":
        render_recommendations_page()


def cleanup():
    """Очистка"""
    if fastapi_client:
        fastapi_client.session.close()


if __name__ == "__main__":
    atexit.register(cleanup)
    main()
