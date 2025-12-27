from typing import List

from fastapi import APIRouter, HTTPException

from sqlmodel import select

from app import crud
from app.api.deps import SessionDep, CurrentUser
from app.models import Message, ProjectBase, Project, ProjectContentBase, ProjectContent, ProjectCreate, ContentTypeEnum
from app.api.utils import check_project_access_right, get_gemini_response

router = APIRouter(prefix="/projects", tags=["projects"])


@router.get("/get_all_projects", response_model=List[Project])
def get_all_projects(session: SessionDep, current_user: CurrentUser) -> [Project]:
    """
    get all projects of current user
    """
    user_id = current_user.id
    statement = select(Project).where(Project.user_id == user_id)
    return session.exec(statement).all()

@router.get("/get_project_content", response_model=List[ProjectContentBase])
def get_project_content(session: SessionDep, current_user: CurrentUser, project_id: int) -> [ProjectContentBase]:
    """
    get all contents of one project
    """
    # 检查用户权限
    if not check_project_access_right(session=session, user_id=current_user.id, project_id=project_id):
        raise HTTPException(
            status_code=400,
            detail="This user have no rights to access this project.",
        )

    # 获取contents
    statement = select(ProjectContent).where(ProjectContent.project_id == project_id)
    return session.exec(statement).all()


@router.post("/create_project", response_model=Project)
def create_project(*, session: SessionDep, project_in: ProjectCreate) -> Project:
    # 检查用户是否存在
    user_email = project_in.email
    if project_in.user_id:
        user_email = crud.get_user_email_by_id(session=session, user_id=project_in.user_id)

    user = crud.get_user_by_email(session=session, email=user_email)
    is_user_active = crud.check_user_active(session=session, email=user_email)
    if not user or not is_user_active:
        raise HTTPException(
            status_code=400,
            detail="The user is not exists in the system.",
        )

    project_in.user_id = user.id
    # 创建project
    result = crud.create_project(session=session, project_create=project_in)
    return result

@router.post("/insert_project_content", response_model=Message)
def insert_project_content(*, session: SessionDep, content_in: ProjectContentBase) -> Message:
    """
    :param session: auto generated, no need to pass

    picture/text content only choose one;
    if choose picture, there will not an AI response, and the picture will be oad into db, and be used to when next text-content
    """
    # 插入对话
    result = crud.add_project_content(session=session, content_create=content_in)
    if not result:
        raise HTTPException(
            status_code=420,
            detail="Insert data error, connect with db administrator",
        )
    # 如果给的是图片直接插入db，并不回复
    if content_in.text_content == ContentTypeEnum.PICTURE:
        return Message(message=f"Insert picture content successfully")
    # 调取AI获得回复
    statement = select(ProjectContent).where(ProjectContent.project_id == content_in.project_id)
    all_contents = session.exec(statement).all()
    response = get_gemini_response(all_contents)
    # 插入AI回复内容
    crud.add_project_content(session=session,
                             content_create=ProjectContentBase(
                                project_id=content_in.project_id,
                                sequence=crud.get_project_next_sequence(session=session, project_id=content_in.project_id),
                                text_content=response)
                             )
    # 返回AI回复内容

    return Message(message=f"{response}")

@router.delete("/delete_project", response_model=Message)
def delete_project(*, session: SessionDep,current_user: CurrentUser, project_id: int) -> Message:
    # 检查用户权限
    if not check_project_access_right(session=session, user_id=current_user.id, project_id=project_id):
        raise HTTPException(
            status_code=400,
            detail="This user have no rights to delete this project.",
        )

    crud.delete_project(session=session, project_id=project_id)
    return Message(message=f"Delete project successfully")