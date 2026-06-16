#!/usr/bin/env python
"""Проверка данных в Neo4j"""

from src.database.neo4j_client import get_neo4j_client

def main():
    client = get_neo4j_client()
    
    print("=" * 60)
    print("ПРОВЕРКА ДАННЫХ В БАЗЕ")
    print("=" * 60)
    
    # Проверка вакансий
    print("\n1. Вакансии в базе (первые 3):")
    result = client.execute_query('MATCH (v:Vacancy) RETURN v.id, v.title, v.skills LIMIT 3')
    print(f"   Найдено {len(result or [])} вакансий")
    for v in result or []:
        print(f"   - {v.get('v.title', 'N/A')}: skills={v.get('v.skills', [])}")
    
    # Проверка пользователей
    print("\n2. Пользователи в базе (первые 3):")
    result = client.execute_query('MATCH (u:User) RETURN u.id, u.skills, u.resume_text LIMIT 3')
    print(f"   Найдено {len(result or [])} пользователей")
    for u in result or []:
        print(f"   - {u.get('u.id', 'N/A')}: skills={u.get('u.skills', [])}")
    
    # Проверка связей
    print("\n3. Связи в базе:")
    result = client.execute_query('MATCH ()-[r]->() RETURN DISTINCT type(r) as relationship_type')
    print(f"   Найдено типов связей: {len(result or [])}")
    for r in result or []:
        print(f"   - {r.get('relationship_type', 'N/A')}")
    
    # Количество связей каждого типа
    print("\n4. Количество связей:")
    for rel_type in ['VIEWED', 'RATED', 'FAVORITED']:
        result = client.execute_query(f'MATCH ()-[r:{rel_type}]->() RETURN COUNT(r) as count')
        count = result[0].get('count', 0) if result else 0
        print(f"   - {rel_type}: {count}")
    
    # Проверка рекомендаций
    print("\n5. Проверка рекомендаций для пользователя 'user1':")
    result = client.execute_query("""
        MATCH (u:User {id: 'user1'})-[:VIEWED]->(v:Vacancy)
        RETURN v.id as vacancy_id, v.title as title, v.skills as skills
    """)
    print(f"   Пользователь просмотрел {len(result or [])} вакансий")
    
    # Попытка получить рекомендации
    print("\n6. Попытка рекомендаций (по навыкам):")
    result = client.execute_query("""
        MATCH (v:Vacancy)
        WHERE v.skills IS NOT NULL AND size(v.skills) > 0
        RETURN v.id, v.title, v.skills
        LIMIT 5
    """)
    print(f"   Вакансий с навыками: {len(result or [])}")
    for v in result or []:
        print(f"   - {v.get('v.title')}: {v.get('v.skills')}")
    
    print("\n" + "=" * 60)

if __name__ == '__main__':
    main()
