class GraphQueries:
    """Коллекция Cypher-запросов для рекомендательной системы"""

    @staticmethod
    def get_similar_users(user_id: str, limit: int = 10):
        return """
        MATCH (u1:User {id: $user_id})-[:HAS_SKILL]->(s:Skill)<-[:HAS_SKILL]-(u2:User)
        WHERE u1 <> u2
        WITH u2, COUNT(DISTINCT s) AS common_skills
        ORDER BY common_skills DESC
        LIMIT $limit
        RETURN u2.id AS user_id, 
               u2.username AS username,
               common_skills
        """

    @staticmethod
    def get_skill_graph(limit: int = 50):
        return """
        MATCH (s1:Skill)<-[:REQUIRES]-(:Vacancy)-[:REQUIRES]->(s2:Skill)
        WHERE s1 <> s2
        WITH s1, s2, COUNT(*) AS cooccurrence
        WHERE cooccurrence >= 2
        RETURN s1.name AS skill1,
               s2.name AS skill2,
               cooccurrence
        ORDER BY cooccurrence DESC
        LIMIT $limit
        """

    @staticmethod
    def get_popular_skills(limit: int = 20):
        return """
        MATCH (v:Vacancy)-[:REQUIRES]->(s:Skill)
        RETURN s.name AS skill_name,
               COUNT(DISTINCT v) AS vacancy_count
        ORDER BY vacancy_count DESC
        LIMIT $limit
        """