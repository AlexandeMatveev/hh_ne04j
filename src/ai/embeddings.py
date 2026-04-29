import numpy as np
from typing import List, Optional
import aiohttp
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
import os

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self, model: str = "mistral-embed"):
        self.model = model
        self.base_url = "https://api.mistral.ai/v1"
        # Загружаем ключ из переменных окружения
        self.api_key = os.getenv("MISTRAL_API_KEY")

        if self.api_key:
            logger.info(f"EmbeddingService initialized with API key: {self.api_key[:10]}...")
        else:
            logger.warning("MISTRAL_API_KEY not found in environment variables!")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def get_embeddings(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Get embeddings for list of texts"""
        if not self.api_key:
            logger.error("No API key found in environment variables!")
            return None

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                        f"{self.base_url}/embeddings",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": self.model,
                            "input": texts
                        }
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"Successfully generated {len(data['data'])} embeddings")
                        return [item["embedding"] for item in data["data"]]
                    else:
                        error_text = await response.text()
                        logger.error(f"API Error {response.status}: {error_text}")
                        return None
        except Exception as e:
            logger.error(f"Exception in get_embeddings: {e}")
            return None

    async def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get single embedding"""
        result = await self.get_embeddings([text])
        return result[0] if result else None

    def get_embedding_sync(self, text: str) -> Optional[List[float]]:
        """Синхронное получение эмбеддинга для Streamlit"""
        if not self.api_key:
            logger.error("Cannot generate embedding: No API key")
            return None

        if not text or len(text.strip()) < 10:
            return None

        try:
            import asyncio
            import nest_asyncio
            nest_asyncio.apply()

            # Создаем новый event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self.get_embeddings([text]))
            loop.close()
            return result[0] if result else None
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None