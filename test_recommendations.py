#!/usr/bin/env python
"""Добавление навыков пользователю и тест рекомендаций"""

from src.database.neo4j_client import get_neo4j_client
from src.services.user_service import UserService
from src.services.recommendation_service import RecommendationService
from src.database.models import User

def main():
    client = get_neo4j_client()
    user_service = UserService(client)
    
    # Берем первого пользователя из базы
    result = client.execute_query('MATCH (u:User) RETURN u.id as id LIMIT 1')
    if not result:
        print("❌ Нет пользователей в базе! Создадим нового.")
        user_id = 'test_user_1'
        user_service.create_user(user_id, name='Test User', email='test@example.com')
    else:
        user_id = result[0]['id']
        print(f"✓ Используем пользователя: {user_id}")
    
    # Добавляем навыки пользователю
    user_skills = ['Python', 'Git', 'Docker', 'Linux']
    print(f"✓ Добавляем навыки пользователю: {user_skills}")
    
    # Создаем объект User с навыками
    user = User(id=user_id, username='Test User', skills=user_skills)
    user_service.create_or_update_user(user)
    
    print("\n" + "="*60)
    print("ПРОВЕРКА ПОСЛЕ ДОБАВЛЕНИЯ НАВЫКОВ")
    print("="*60)
    
    # Проверяем, что навыки добавились
    user_db = user_service.get_user(user_id)
    print(f"\nПользователь после обновления:")
    print(f"  ID: {user_db.get('id')}")
    print(f"  Навыки: {user_db.get('skills', [])}")
    
    # Получаем рекомендации
    print("\n" + "="*60)
    print("РЕКОМЕНДАЦИИ ДЛЯ ПОЛЬЗОВАТЕЛЯ")
    print("="*60)
    
    from src.ai.embeddings import EmbeddingService
    embedding_service = EmbeddingService()
    recommendation_service = RecommendationService(client, embedding_service)
    
    recommendations = recommendation_service.get_recommendations_for_user(user_id, limit=5)
    
    print(f"\nНайдено рекомендаций: {len(recommendations)}")
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec.get('title')}")
        print(f"   ID: {rec.get('id')}")
        print(f"   Навыки: {rec.get('skills', [])}")
        print(f"   Компания: {rec.get('company_name', 'N/A')}")
        if 'matching_skills_count' in rec:
            print(f"   Совпадающих навыков: {rec['matching_skills_count']}")
    
    if not recommendations:
        print("\n❌ Рекомендации не найдены!")
        print("   Возможно, навыки пользователя не совпадают с вакансиями.")
    else:
        print("\n✅ Рекомендации получены успешно!")
    
    # Проверка рекомендаций по навыкам напрямую
    print("\n" + "="*60)
    print("ТЕСТ ПОИСКА ВАКАНСИЙ ПО НАВЫКАМ")
    print("="*60)
    recommendations_direct = recommendation_service.get_recommendations_by_skills(user_skills, limit=5)
    print(f"Рекомендаций по навыкам {user_skills}: {len(recommendations_direct)}")
    for rec in recommendations_direct:
        print(f"  - {rec.get('title')}: {rec.get('matching_skills', [])}")

if __name__ == '__main__':
    main()
