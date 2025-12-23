from typing import Any
from fastapi import APIRouter, Depends, HTTPException
from app import crud
from app.api.deps import SessionDep
from app.models import (
    UpdatePassword,
    # User,
    UserCreate,
    UserPublic,
    UserRegister,
    UsersPublic,
    UserUpdate,
    UserUpdateMe,
)
from app.emails import send_email, generate_new_account_email
from app.core.config import settings

router = APIRouter(prefix="/users", tags=["users"])
page_name = "users"

@router.post(
    "/create", response_model=UserPublic
)
def create_user(*, session: SessionDep, user_in: UserCreate) -> Any:
    """
    Create new user.
    """
    # 检查用户是否存在
    user = crud.get_user_by_email(session=session, email=user_in.email)
    if user:
        raise HTTPException(
            status_code=400,
            detail="The user with this email already exists in the system.",
        )
    # 创建用户 默认权限0 激活后权限1
    user = crud.create_user(session=session, user_create=user_in)
    print(f"user data inserted, sending email\n{user}")
    if settings.emails_enabled and user_in.email:
        email_data = generate_new_account_email(
            email_to=user_in.email, username=user_in.email, password=user_in.password
        )
        send_email(
            email_to=user_in.email,
            subject=email_data.subject,
            html_content=email_data.html_content,
        )
    return user