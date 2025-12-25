import uuid
from sqlmodel import Session, select, update
from typing import Any
from app.core.security import get_password_hash, verify_password
from app.models import User, UserCreate, Projects, Feedback

def create_user(*, session: Session, user_create: UserCreate) -> User:
    db_obj = User.model_validate(
        user_create, update={"password": get_password_hash(user_create.password), "privilege": 0}
    )
    print(db_obj)
    session.add(db_obj)
    session.commit()
    session.refresh(db_obj)
    return db_obj

# def update_user(*, session: Session, db_user: User, user_in: UserUpdate) -> Any:
#     user_data = user_in.model_dump(exclude_unset=True)
#     extra_data = {}
#     if "password" in user_data:
#         password = user_data["password"]
#         hashed_password = get_password_hash(password)
#         extra_data["hashed_password"] = hashed_password
#     db_user.sqlmodel_update(user_data, update=extra_data)
#     session.add(db_user)
#     session.commit()
#     session.refresh(db_user)
#     return db_user


def get_user_by_email(*, session: Session, email: str) -> User | None:
    statement = select(User).where(User.email == email)
    session_user = session.exec(statement).first()
    return session_user


def authenticate(*, session: Session, email: str, password: str) -> User | None:
    db_user = get_user_by_email(session=session, email=email)
    if not db_user:
        return None
    if not verify_password(password, db_user.hashed_password):
        return None
    return db_user

def check_user_active(*, session: Session, email: str) -> bool | None:
    statement = select(User).where(User.email == email, User.privilege != 0)
    if session.exec(statement).first():
        return True
    return False

def activate_user(*, session: Session, email: str) -> User | None:
    base_privilege: int = 1

    statement = select(User).where(User.email == email)
    session_user = session.exec(statement).first()
    session_user.privilege = base_privilege
    session.add(session_user)
    session.commit()
    session.refresh(session_user)

def delete_user(*, session: Session, email: str) -> bool | None:
    # get user id
    u_statement = select(User).where(User.email == email)
    u_result = session.exec(u_statement).first()
    user_id = u_result.id

    # 删除project
    p_statement = select(Projects).where(Projects.user_id == user_id)
    p_results = session.exec(p_statement)
    session.delete(p_results)
    session.commit()
    session.refresh(p_results)

    # 删除feedback
    f_statement = select(Feedback).where(Feedback.user_id == user_id)
    f_results = session.exec(f_statement)
    session.delete(f_results)
    session.commit()
    session.refresh(f_results)

    # 删除user
    session.delete(u_result)
    session.commit()
    session.refresh(u_result)

    return True
