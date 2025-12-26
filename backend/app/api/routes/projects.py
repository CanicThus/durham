"""
一个api，接受发来的文字/图片， 返回AI回复，同时更新DB数据
gimi
AIzaSyApUbNi6b5FxaM7785AJzSsfvwugh7DP5U

content内容为json[]
{
    “str”: sentence1
    "bytea": picture
    “str”: answer1
    ...
}

或者
content内容拆多列村
project表
id  user_id sequence content_type str_content bytea_content


单个project

用户和反馈的模块已经全部完成了
我看例子上我们做的chatAI也是和chatGPT差不多的，创建新对话，一个对话里有上下文。
目前我这边时打算做一个接口，你那边传整个对话的json，我这边AI处理返回回复
或者是传单个对话
"""

from fastapi import APIRouter, HTTPException
from google import genai

from app import crud
from app.api.deps import SessionDep
from app.models import Message

router = APIRouter(prefix="/projects", tags=["projects"])


@router.post("", response_model=Message)
def xxx_projects():
    pass
