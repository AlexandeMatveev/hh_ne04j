from datetime import datetime
from enum import Enum
import json


# Enums
class FeedbackType(Enum):
    LIKE = "LIKED"
    DISLIKE = "DISLIKED"
    VIEW = "VIEWED"
    APPLY = "APPLIED"


# Класс Vacancy
class Vacancy:
    def __init__(self, id, title, description, **kwargs):
        self.id = id
        self.title = title
        self.description = description
        self.external_id = kwargs.get('external_id')
        self.salary_from = kwargs.get('salary_from')
        self.salary_to = kwargs.get('salary_to')
        self.currency = kwargs.get('currency')
        self.experience = kwargs.get('experience')
        self.employment = kwargs.get('employment')
        self.skills = kwargs.get('skills', [])
        self.company_name = kwargs.get('company_name')
        self.location_name = kwargs.get('location_name')
        self.published_at = kwargs.get('published_at')
        self.embedding = kwargs.get('embedding')

    def to_dict(self):
        return {
            'id': self.id,
            'external_id': self.external_id,
            'title': self.title,
            'description': self.description,
            'salary_from': self.salary_from,
            'salary_to': self.salary_to,
            'currency': self.currency,
            'experience': self.experience,
            'employment': self.employment,
            'skills': self.skills,
            'company_name': self.company_name,
            'location_name': self.location_name,
            'published_at': self.published_at.isoformat() if self.published_at else None,
            'embedding': self.embedding
        }

    @classmethod
    def from_dict(cls, data):
        published_at = None
        if data.get('published_at'):
            try:
                published_at = datetime.fromisoformat(data['published_at'].replace('Z', '+00:00'))
            except:
                pass

        return cls(
            id=data.get('id', ''),
            title=data.get('title', ''),
            description=data.get('description', ''),
            external_id=data.get('external_id'),
            salary_from=data.get('salary_from'),
            salary_to=data.get('salary_to'),
            currency=data.get('currency'),
            experience=data.get('experience'),
            employment=data.get('employment'),
            skills=data.get('skills', []),
            company_name=data.get('company_name'),
            location_name=data.get('location_name'),
            published_at=published_at,
            embedding=data.get('embedding')
        )


# Класс User
class User:
    def __init__(self, id, username, **kwargs):
        self.id = id
        self.username = username
        self.resume_text = kwargs.get('resume_text')
        self.skills = kwargs.get('skills', [])
        self.preferences = kwargs.get('preferences', {})
        self.embedding = kwargs.get('embedding')

    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'resume_text': self.resume_text or "",
            'skills': self.skills,
            'preferences': json.dumps(self.preferences) if self.preferences else "{}",
            'embedding': self.embedding
        }

    @classmethod
    def from_dict(cls, data):
        preferences = {}
        if data.get('preferences'):
            if isinstance(data['preferences'], str):
                try:
                    preferences = json.loads(data['preferences'])
                except:
                    preferences = {}
            elif isinstance(data['preferences'], dict):
                preferences = data['preferences']

        return cls(
            id=data.get('id', ''),
            username=data.get('username', ''),
            resume_text=data.get('resume_text'),
            skills=data.get('skills', []),
            preferences=preferences,
            embedding=data.get('embedding')
        )


# Класс UserFeedback
class UserFeedback:
    def __init__(self, user_id, vacancy_id, feedback_type, **kwargs):
        self.user_id = user_id
        self.vacancy_id = vacancy_id
        self.feedback_type = feedback_type
        self.timestamp = kwargs.get('timestamp', datetime.now())
        self.interaction_time = kwargs.get('interaction_time')

    def to_dict(self):
        return {
            'user_id': self.user_id,
            'vacancy_id': self.vacancy_id,
            'feedback_type': self.feedback_type.value,
            'timestamp': self.timestamp.isoformat(),
            'interaction_time': self.interaction_time
        }


# Класс RecommendationScore
class RecommendationScore:
    def __init__(self, vacancy, content_score, graph_score, semantic_score, total_score):
        self.vacancy = vacancy
        self.content_score = content_score
        self.graph_score = graph_score
        self.semantic_score = semantic_score
        self.total_score = total_score