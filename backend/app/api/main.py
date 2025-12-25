from fastapi import APIRouter

# every file in routes
from app.api.routes import users, login, user_feedback
from app.core.config import settings

api_router = APIRouter()
api_router.include_router(users.router)
api_router.include_router(login.router)
api_router.include_router(user_feedback.router)