# api/dependencies.py
from src.database.neo4j_client import Neo4jClient
from src.services.vacancy_service import VacancyService
from src.services.user_service import UserService
from src.services.recommendation_service import RecommendationService
from src.services.feedback_service import FeedbackService
from src.ai.embeddings import EmbeddingService
from src.parsers.hh_parser import HHParser
import logging

logger = logging.getLogger(__name__)

# Глобальные переменные для синглтонов
_neo4j_client = None
_embedding_service = None
_vacancy_service = None
_user_service = None
_recommendation_service = None
_feedback_service = None
_hh_parser = None

def get_neo4j_client():
    global _neo4j_client
    if _neo4j_client is None:
        _neo4j_client = Neo4jClient()
        _neo4j_client.connect()
        _neo4j_client.initialize_database()
    return _neo4j_client

def get_embedding_service():
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service

def get_vacancy_service():
    global _vacancy_service
    if _vacancy_service is None:
        neo4j_client = get_neo4j_client()
        embedding_service = get_embedding_service()
        _vacancy_service = VacancyService(neo4j_client, embedding_service)
    return _vacancy_service

def get_user_service():
    global _user_service
    if _user_service is None:
        neo4j_client = get_neo4j_client()
        _user_service = UserService(neo4j_client)
    return _user_service

def get_recommendation_service():
    global _recommendation_service
    if _recommendation_service is None:
        neo4j_client = get_neo4j_client()
        embedding_service = get_embedding_service()
        _recommendation_service = RecommendationService(neo4j_client, embedding_service)
    return _recommendation_service

def get_feedback_service():
    global _feedback_service
    if _feedback_service is None:
        neo4j_client = get_neo4j_client()
        _feedback_service = FeedbackService(neo4j_client)
    return _feedback_service

def get_hh_parser():
    global _hh_parser
    if _hh_parser is None:
        _hh_parser = HHParser()
    return _hh_parser




                                   # 1 аргумент  → __init__(self)