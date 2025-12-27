from google import genai
from sqlmodel import select

from app.api.deps import SessionDep
from app.core.config import settings
from app.models import Project, ProjectContent, ContentTypeEnum


def check_project_access_right(*, session: SessionDep, user_id: int, project_id: int) -> bool:
    statement = select(Project).where(Project.id == project_id)
    result = session.exec(statement).first()
    if result.user_id == user_id:
        return True
    return False


def get_gemini_response(content_in: [ProjectContent]) -> str:
    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    """
    把所有的记录合并成一个上下文
    role:你是一个厨师和营养师，并掌握大量的医学常识。需要你给65岁以上的英国老人提供餐饮建议和推荐详细的食谱
    如果没有记录
        第一条是用户选择的，前端将第一句话和用户的话一起传入当作上下文
    # 直接整个发过去的也可以看看效果
    """
    context = []
    for content in content_in:
        if content.content_type == ContentTypeEnum.PICTURE:
            pictures = content.picture_content
            context.append(pictures)
            continue
        context.append(content.text_content)


    response = client.models.generate_content(
        model=settings.GEMINI_MODEL,
        contents=context,
    )

    client.close()
    return response.text

