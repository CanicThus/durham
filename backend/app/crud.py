import uuid
from sqlmodel import Session, select, update, delete
from typing import Any
from app.core.security import get_password_hash, verify_password
from app.models import User, UserCreate, Project, Feedback, FeedbackBase, ProjectBase, ProjectContentBase, ProjectContent


def create_user(*, session: Session, user_create: UserCreate) -> User:
    db_obj = User.model_validate(
        user_create, update={"password": get_password_hash(user_create.password), "privilege": 0}
    )
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

def get_user_by_id(*, session: Session, user_id: int) -> User | None:
    statement = select(User).where(User.id == user_id)
    session_user = session.exec(statement).first()
    return session_user

def authenticate(*, session: Session, email: str, password: str) -> User | None:
    db_user = get_user_by_email(session=session, email=email)
    if not db_user:
        return None
    if not verify_password(password, db_user.password):
        return None
    return db_user

def check_user_active(*, session: Session, email: str) -> bool | None:
    statement = select(User).where(User.email == email, User.privilege != 0)
    if session.exec(statement).first():
        return True
    return False

def get_user_email_by_id(*, session: Session, user_id: int) -> str | None:
    statement = select(User).where(User.id == user_id)
    result = session.exec(statement).first()
    if result:
        return result.email
    return None


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

    # 删除project content 和 project
    p_statement = select(Project).where(Project.user_id == user_id)
    p_results = session.exec(p_statement)
    for result in p_results:
        p_content_statement = delete(ProjectContent).where(ProjectContent.project_id == result.id)
        session.exec(p_content_statement)
        session.commit()

    p_d_statement = delete(Project).where(Project.user_id == user_id)
    session.exec(p_d_statement)
    session.commit()

    # 删除feedback
    f_statement = delete(Feedback).where(Feedback.user_id == user_id)
    session.exec(f_statement)
    session.commit()

    # 删除user
    session.delete(u_result)
    session.commit()

    return True

def create_feedback(*, session: Session, feedback_create: FeedbackBase) -> Feedback:
    db_obj = Feedback.model_validate(feedback_create)
    session.add(db_obj)
    session.commit()
    session.refresh(db_obj)
    return db_obj

def create_project(*, session: Session, project_create: ProjectBase) -> Project:
    db_obj = Project.model_validate(project_create)
    session.add(db_obj)
    session.commit()
    session.refresh(db_obj)
    return db_obj

def get_project_next_sequence(*, session: Session, project_id: int) -> int:
    statement = select(ProjectContent).where(ProjectContent.project_id == project_id)
    result = session.exec(statement)
    if result:
        return len(result.all()) + 1
    return 1

def add_project_content(*, session: Session, content_create: ProjectContentBase) -> ProjectContent:
    db_obj = ProjectContent.model_validate(content_create, update={"sequence": get_project_next_sequence(session=session, project_id=content_create.project_id)})
    session.add(db_obj)
    session.commit()
    session.refresh(db_obj)
    return db_obj

def delete_project(*, session: Session, project_id: int) -> bool:
    # 删除content
    p_content_statement = delete(ProjectContent).where(ProjectContent.project_id == project_id)
    session.exec(p_content_statement)
    session.commit()

    # 删除project
    p_statement = delete(Project).where(Project.id == project_id)
    session.exec(p_statement)
    session.commit()

    return True