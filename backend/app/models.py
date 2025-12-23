import uuid

from pydantic import EmailStr
from sqlmodel import Field, Relationship, SQLModel
from sqlalchemy import Column, Integer, String
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
    preference: str | None = None
    # 头像
    profile_photo: bytes

# Properties to receive via API on creation
class UserCreate(UserBase):
    password: str = Field(min_length=8, max_length=128)

class UserRegister(SQLModel):
    email: EmailStr = Field(max_length=255)
    password: str = Field(min_length=8, max_length=128)
    full_name: str | None = Field(default=None, max_length=255)

# Properties to receive via API on update, all are optional
class UserUpdate(UserBase):
    email: EmailStr | None = Field(default=None, max_length=255)  # type: ignore
    password: str | None = Field(default=None, min_length=8, max_length=128)
    preference: str | None = Field(default=None)


class UserUpdateMe(SQLModel):
    full_name: str | None = Field(default=None, max_length=255)
    email: EmailStr | None = Field(default=None, max_length=255)


class UpdatePassword(SQLModel):
    current_password: str = Field(min_length=8, max_length=128)
    new_password: str = Field(min_length=8, max_length=128)


# Database model, database table inferred from class name
class User(UserBase, table=True):
    id: int | None = Field(primary_key=True, default=None)
    password: str

# Properties to return via API, id is always required
class UserPublic(UserBase):
    id: int


class UsersPublic(SQLModel):
    data: list[UserPublic]
    count: int