
from fastapi import APIRouter, HTTPException
from google import genai

from app import crud
from app.api.deps import SessionDep
from app.models import Message

router = APIRouter(prefix="/projects", tags=["projects"])


@router.post("", response_model=Message)
def xxx_projects():
    pass
