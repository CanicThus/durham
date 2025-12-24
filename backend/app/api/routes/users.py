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
    UserUpdateMe, Message,
)
from app.utils import send_email, generate_new_account_email, generate_email_token, verify_email_token
from app.core.config import settings

router = APIRouter(prefix="/users", tags=["users"])
page_name = "users"

@router.post(
    "/create", response_model=Message
)
def create_user(*, session: SessionDep, user_in: UserCreate) -> Message:
    """
    Create new user.
    """
    # 检查用户是否存在
    user = crud.get_user_by_email(session=session, email=user_in.email)
    is_user_active = crud.check_user_active(session=session, email=user_in.email)
    if user and not is_user_active:
        raise HTTPException(
            status_code=400,
            detail="The user with this email already exists in the system.",
        )
    # 发送邮件
    if not user:
        crud.create_user(session=session, user_create=user_in)
    create_account_token = generate_email_token(user_in.email)
    if settings.emails_enabled and user_in.email:
        email_data = generate_new_account_email(
            email_to=user_in.email, username=user_in.email,
            password=user_in.password, token=create_account_token
        )
        send_email(
            email_to=user_in.email,
            subject=email_data.subject,
            html_content=email_data.html_content,
        )
    return Message(message="user activate email sent")


@router.get("/verify_email", response_model=Message)
def verify_email(email_token: str, session: SessionDep):
    """
    验证登入邮箱
    """
    # 验证 Token
    user_email = verify_email_token(email_token)
    if not user_email:
        raise HTTPException(status_code=400, detail="invalid token")
    # check is activate
    if crud.check_user_active(session=session, email=user_email):
        raise HTTPException(status_code=400, detail="user is active")
    # 激活用户
    crud.activate_user(session=session, email=user_email)

    return Message(message="activate user successfully")