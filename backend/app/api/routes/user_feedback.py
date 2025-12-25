from fastapi import APIRouter, HTTPException

from app import crud
from app.api.deps import SessionDep
from app.models import Message, FeedbackBase

router = APIRouter(prefix="/feedback", tags=["feedback"])

@router.post("/create", response_model=Message)
def create_feedback(*, session: SessionDep, feedback_in: FeedbackBase) -> Message:
    # 检查用户是否存在/激活
    user = crud.get_user_by_id(session=session, id=feedback_in.user_id)
    is_user_active = crud.check_user_active(session=session, email=user.email)
    if not user or not is_user_active:
        raise HTTPException(
            status_code=400,
            detail="The user with this id is not exist or active in system.",
        )
    # db中插入feedback数据
    feedback = crud.create_feedback(session=session, feedback_create=feedback_in)
    return Message(message="create feedback successfully")