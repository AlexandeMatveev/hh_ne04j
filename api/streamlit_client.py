"""
Обновление app.py для взаимодействия с FastAPI
Вариант: Streamlit делает HTTP запросы к FastAPI
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import atexit
import logging
import sys
from pathlib import Path

# Добавляем корень проекта в путь
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== КОНФИГУРАЦИЯ FASTAPI ====================
FASTAPI_BASE_URL = "http://localhost:8000"


# ==================== HTTP CLIENT ДЛЯ FASTAPI ====================
class FastAPIClient:
    """Клиент для взаимодействия с FastAPI"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def _request(self, method: str, endpoint: str, **kwargs):
        """Выполнить HTTP запрос"""
        try:
            url = f"{self.base_url}{endpoint}"
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"FastAPI request error: {e}")
            return None
    
    def get_vacancies(self, search_query: str = "", limit: int = 20):
        """Получить вакансии"""
        return self._request("GET", f"/api/v1/vacancies/search", 
                           params={"query": search_query, "limit": limit})
    
    def get_recommendations(self, user_id: str, top_n: int = 10):
        """Получить рекомендации"""
        return self._request("POST", "/api/v1/recommendations/", 
                           json={"user_id": user_id, "top_n": top_n})
    
    def add_feedback(self, user_id: str, vacancy_id: str, feedback_type: str):
        """Добавить обратную связь"""
        return self._request("POST", "/api/v1/feedback/", 
                           json={"user_id": user_id, "vacancy_id": vacancy_id, 
                                 "feedback_type": feedback_type})
    
    def get_health(self):
        """Проверка здоровья FastAPI"""
        return self._request("GET", "/health")


# ==================== ГЛОБАЛЬНАЯ ПЕРЕМЕННАЯ ДЛЯ КЛИЕНТА ====================
fastapi_client = None


# ==================== УТИЛИТЫ ====================
def init_fastapi_client():
    """Инициализация FastAPI клиента"""
    global fastapi_client
    if fastapi_client is None:
        fastapi_client = FastAPIClient(FASTAPI_BASE_URL)
    return fastapi_client


# ==================== СЕРВИСЫ НА ОСНОВЕ FASTAPI ====================
class FastAPIVacancyService:
    """Обертка для вакансий через FastAPI"""
    
    def __init__(self, client: FastAPIClient):
        self.client = client
    
    def search(self, query: str, limit: int = 20):
        return self.client.get_vacancies(query, limit)
    
    def get_recommendations(self, user_id: str, top_n: int = 10):
        return self.client.get_recommendations(user_id, top_n)


class FastAPIUserService:
    """Обертка для пользователей через FastAPI"""
    
    def __init__(self, client: FastAPIClient):
        self.client = client


class FastAPIFeedbackService:
    """Обертка для обратной связи через FastAPI"""
    
    def __init__(self, client: FastAPIClient):
        self.client = client
    
    def add_like(self, user_id: str, vacancy_id: str):
        return self.client.add_feedback(user_id, vacancy_id, "LIKED")
    
    def add_dislike(self, user_id: str, vacancy_id: str):
        return self.client.add_feedback(user_id, vacancy_id, "DISLIKED")
    
    def record_view(self, user_id: str, vacancy_id: str):
        return self.client.add_feedback(user_id, vacancy_id, "VIEWED")


# ==================== ИНИЦИАЛИЗАЦИЯ ====================
@st.cache_resource
def init_services():
    """Инициализация сервисов через FastAPI"""
    global fastapi_client
    
    logger.info("🚀 Инициализация через FastAPI...")
    
    try:
        fastapi_client = init_fastapi_client()
        
        # Проверка соединения
        health = fastapi_client.get_health()
        if health:
            logger.info(f"✅ FastAPI connected: {health}")
        else:
            logger.warning("⚠️ FastAPI connection failed, using local services")
        
        return {
            'fastapi': fastapi_client,
            'vacancy_service': FastAPIVacancyService(fastapi_client),
            'feedback_service': FastAPIFeedbackService(fastapi_client)
        }
    except Exception as e:
        logger.error(f"❌ Failed to init FastAPI services: {e}")
        return None


# ==================== ИСПОЛЬЗОВАНИЕ В app.py ====================
# Вместо:
# services['vacancy_service'].get_recommendations(user_id, top_n)

# Используем:
# fastapi_client.get_recommendations(user_id, top_n)

# ИЛИ через обертку:
# services['vacancy_service'].get_recommendations(user_id, top_n)
