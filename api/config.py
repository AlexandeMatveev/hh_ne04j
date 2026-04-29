import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    def __init__(self):
        # Neo4j
        self.neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD", "")

        # Mistral AI
        self.mistral_api_key = os.getenv("MISTRAL_API_KEY", "")

        # HH.ru API (опционально)
        self.hh_api_client_id = os.getenv("HH_API_CLIENT_ID", "")
        self.hh_api_client_secret = os.getenv("HH_API_CLIENT_SECRET", "")

        # Recommendation weights
        self.content_weight = 0.3
        self.graph_weight = 0.4
        self.semantic_weight = 0.3

        # Learning rate for feedback
        self.learning_rate = 0.1
        self.regularization_lambda = 0.01


# Создаем экземпляр настроек
settings = Settings()