from mistralai import Mistral
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging
from config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self):
        self.api_key = settings.mistral_api_key  # строчная буква
        self.client = None

        if self.api_key:
            try:
                self.client = Mistral(api_key=self.api_key)
                self.model = "mistral-embed"
                logger.info("Mistral AI client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Mistral AI: {e}")
        else:
            logger.warning("Mistral API key not set, using dummy embeddings")

    def get_embedding(self, text):
        """Получение эмбеддинга текста"""
        if not text or not text.strip():
            return None

        # Если нет API ключа, возвращаем фиктивный эмбеддинг
        if not self.client:
            return self._get_dummy_embedding(text)

        try:
            embeddings_batch_response = self.client.embeddings.create(
                model=self.model,
                inputs=[text]
            )
            return embeddings_batch_response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return self._get_dummy_embedding(text)

    def _get_dummy_embedding(self, text):
        """Фиктивный эмбеддинг для тестирования"""
        # Создаем детерминированный эмбеддинг на основе текста
        np.random.seed(hash(text) % (2 ** 32))
        return list(np.random.randn(1024))

    def get_similarity(self, embedding1, embedding2):
        """Вычисление косинусной близости"""
        if not embedding1 or not embedding2:
            return 0.0

        try:
            emb1 = np.array(embedding1).reshape(1, -1)
            emb2 = np.array(embedding2).reshape(1, -1)
            similarity = cosine_similarity(emb1, emb2)[0][0]
            return float(similarity)
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0