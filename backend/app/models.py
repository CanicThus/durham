import uuid
from typing import List

from pydantic import EmailStr
from sqlmodel import Field, Relationship, SQLModel
from sqlalchemy import Column, Integer, String, table, LargeBinary
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from app.core.config import settings

# Shared properties
class UserBase(SQLModel):
    __table_args__ = {"schema": settings.POSTGRES_SCHEMA}
    __tablename__ = "account"

    email: EmailStr = Field(unique=True, index=True, max_length=255)
    name: str = "default_name"
    # 权限等级 0最低
    privilege: int | None = None
    # json str
    preference: dict | None = Field(
        default=None,
        sa_column=Column(JSONB)
    )
    # 头像
    profile_photo: bytes |None = Field(
        default=None,
        sa_column=Column(LargeBinary)
    )

# Properties to receive via API on creation
class UserCreate(UserBase):
    password: str = Field(min_length=8, max_length=128)

class UserRegister(SQLModel):
    email: EmailStr = Field(max_length=255)
    password: str = Field(min_length=8, max_length=128)
    full_name: str | None = Field(default=None, max_length=255)

# Properties to receive via API on update, all are optional
class UpdatePassword(SQLModel):
    current_password: str = Field(min_length=8, max_length=128)
    new_password: str = Field(min_length=8, max_length=128)

class UpdatePreference(SQLModel):
    new_preference: dict | None = Field(default=None, sa_column=Column(JSONB))

class UpdateProfile(SQLModel):
    mew_profile: bytes | None = Field(default=None, sa_column=Column(LargeBinary))

# Database model, database table inferred from class name
class User(UserBase, table=True):
    id: int | None = Field(primary_key=True, default=None)
    password: str

# Properties to return via API, id is always required
class UserPublic(UserBase):
    id: int


# class UsersPublic(SQLModel):
#     data: list[UserPublic]
#     count: int


# Generic message
class Message(SQLModel):
    message: str

# JSON payload containing access token
class Token(SQLModel):
    access_token: str
    token_type: str = "bearer"

# Contents of JWT token
class TokenPayload(SQLModel):
    sub: str | None = None

class NewPassword(SQLModel):
    token: str
    new_password: str = Field(min_length=8, max_length=128)


class FeedbackBase(SQLModel):
    __table_args__ = {"schema": settings.POSTGRES_SCHEMA}
    __tablename__ = "feedback"

    user_id: int | None = None
    content:str | None = None
    photos:List[bytes] | None = Field(
        default=None,
        sa_column=Column(ARRAY(LargeBinary))
    )

class Feedback(FeedbackBase, table=True):
    id: int = Field(default=None, primary_key=True)

class ProjectBase(SQLModel):
    __table_args__ = {"schema": settings.POSTGRES_SCHEMA}
    __tablename__ = "projects"

    user_id: int | None = None
    content: dict | None = Field(
        default=None,
        sa_column=Column(JSONB)
    )

class Project(ProjectBase, table=True):
    id: int = Field(default=None, primary_key=True)

class UpdateProject(SQLModel):
    id: int | None = None
    content: dict | None = Field(
        default=None,
        sa_column=Column(JSONB)
    )