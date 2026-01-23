from neo4j import GraphDatabase
import logging

logger = logging.getLogger(__name__)


class Neo4jClient:
    def __init__(self, uri=None, user=None, password=None):
        from config import settings

        # Используем строчные атрибуты
        self.uri = uri or settings.neo4j_uri
        self.user = user or settings.neo4j_user
        self.password = password or settings.neo4j_password

        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password)
            )
            logger.info(f"Connected to Neo4j at {self.uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            self.driver = None

    def close(self):
        if self.driver:
            self.driver.close()

    def execute_query(self, query, parameters=None):
        if not self.driver:
            logger.error("No Neo4j connection")
            return []

        try:
            with self.driver.session() as session:
                result = session.run(query, parameters or {})
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return []

    def initialize_database(self):
        """Создание индексов"""
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
            except Exception as e:
                logger.warning(f"Error creating index: {e}")

        logger.info("Database initialized")