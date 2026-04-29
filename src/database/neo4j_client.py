# src/database/neo4j_client.py
from neo4j import GraphDatabase, AsyncGraphDatabase
import logging
from typing import Dict, Any, List, Optional
import asyncio

logger = logging.getLogger(__name__)


class Neo4jClient:
    def __init__(self, uri: str = "bolt://localhost:7687", user: str = "neo4j", password: str = "1234567890"):
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None
        self._async_driver = None

    def connect(self):
        """Синхронное подключение к Neo4j"""
        if not self.driver:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password)
            )
            logger.info(f"Connected to Neo4j at {self.uri}")
        return self.driver

    async def connect_async(self):
        """Асинхронное подключение к Neo4j"""
        if not self._async_driver:
            self._async_driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password)
            )
            logger.info(f"Async connected to Neo4j at {self.uri}")
        return self._async_driver

    def execute_query(self, query: str, parameters: Dict[str, Any] = None) -> Optional[List[Dict]]:
        """Синхронное выполнение запроса"""
        try:
            with self.connect().session() as session:
                result = session.run(query, parameters or {})
                records = [record.data() for record in result]
                return records if records else None
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return None

    async def execute_query_async(self, query: str, parameters: Dict[str, Any] = None) -> Optional[List[Dict]]:
        """Асинхронное выполнение запроса"""
        try:
            driver = await self.connect_async()
            async with driver.session() as session:
                result = await session.run(query, parameters or {})
                records = await result.data()
                return records if records else None
        except Exception as e:
            logger.error(f"Error executing async query: {e}")
            return None

    def close(self):
        """Закрыть соединение"""
        if self.driver:
            self.driver.close()
            self.driver = None
            logger.info("Neo4j connection closed")

    async def close_async(self):
        """Закрыть асинхронное соединение"""
        if self._async_driver:
            await self._async_driver.close()
            self._async_driver = None
            logger.info("Async Neo4j connection closed")

    def initialize_database(self):
        """Инициализация базы данных (создание индексов)"""
        queries = [
            "CREATE INDEX IF NOT EXISTS FOR (u:User) ON (u.id)",
            "CREATE INDEX IF NOT EXISTS FOR (v:Vacancy) ON (v.id)",
            "CREATE INDEX IF NOT EXISTS FOR (s:Skill) ON (s.name)",
            "CREATE INDEX IF NOT EXISTS FOR (c:Company) ON (c.name)",
            "CREATE INDEX IF NOT EXISTS FOR (l:Location) ON (l.name)",
        ]

        for query in queries:
            try:
                self.execute_query(query)
                logger.info(f"Index created: {query[:50]}...")
            except Exception as e:
                logger.warning(f"Could not create index: {e}")

        logger.info("Database initialized")


# Создаем глобальный экземпляр
_neo4j_client = None


def get_neo4j_client():
    """Получить клиент Neo4j (синглтон)"""
    global _neo4j_client
    if _neo4j_client is None:
        _neo4j_client = Neo4jClient()
        _neo4j_client.connect()
        _neo4j_client.initialize_database()
    return _neo4j_client