from typing import Any
from fastapi import APIRouter, Depends, HTTPException


from app import crud
from app.api.deps import SessionDep, CurrentUser
from app.core.security import get_password_hash, verify_password
from app.models import (
    UpdatePassword,
    UpdateProfile,
    UpdatePreference,
    UserCreate,
    Message,
    UserPublic,
    UpdateName,
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
    if user or is_user_active:
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
            email_to=user_in.email, username=user_in.name,
            password=user_in.password, token=create_account_token
        )
        send_email(
            email_to=user_in.email,
            subject=email_data.subject,
            html_content=email_data.html_content,
        )
    return Message(message="user activate email sent")


@router.get("/verify_email/create", response_model=Message)
def verify_email_create(email_token: str, session: SessionDep):
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

@router.patch("/me/password", response_model=Message)
def update_password_me(
    *, session: SessionDep, body: UpdatePassword, current_user: CurrentUser
) -> Any:
    """
    Update own password.
    """
    if not verify_password(body.current_password, current_user.hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect password")
    if body.current_password == body.new_password:
        raise HTTPException(
            status_code=400, detail="New password cannot be the same as the current one"
        )
    hashed_password = get_password_hash(body.new_password)
    current_user.hashed_password = hashed_password
    session.add(current_user)
    session.commit()
    return Message(message="Password update successfully")

@router.patch("/me/profile_photo", response_model=Message)
def update_Profile_me(
    *, session: SessionDep, body: UpdateProfile, current_user: CurrentUser
) -> Any:
    """
    更新用户头像
    """
    current_user.profile_photo = body.mew_profile
    session.add(current_user)
    session.commit()
    return Message(message="Update Profile photo successfully")

@router.patch("/me/preference", response_model=Message)
def update_Preference_me(
    *, session: SessionDep, body: UpdatePreference, current_user: CurrentUser
) -> Any:
    """
    跟新用户信息
    目前用户信息在chat中无应用
    """
    current_user.preference = body.new_preference
    session.add(current_user)
    session.commit()
    return Message(message="Update preference successfully")

@router.patch("/me/name", response_model=Message)
def update_Name_me(
    *, session: SessionDep, body: UpdateName, current_user: CurrentUser
) -> Any:
    """
    update name
    """
    current_user.name = body.name
    session.add(current_user)
    session.commit()
    return Message(message="Update name successfully")

@router.delete("/me", response_model=Message)
def delete_user_me(session: SessionDep, current_user: CurrentUser) -> Any:
    """
    Delete own user.
    """
    # 检查用户存在
    user = crud.get_user_by_email(session=session, email=current_user.email)
    if not user:
        raise HTTPException(
            status_code=400,
            detail="The user with this email does not exists in the system.",
        )
    crud.delete_user(session=session, email=current_user.email)
    return Message(message="User deleted successfully")

@router.get("/me", response_model=UserPublic)
def read_user_me(current_user: CurrentUser) -> Any:
    """
    Get current user.
    """
    return current_user