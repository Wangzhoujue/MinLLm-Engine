
from pydantic import BaseModel,Field
from typing import Literal



class ChatMessage(BaseModel):
    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str

class ChatCompletionRequest(BaseModel):
    model:str
    messages:list[ChatMessage]
    max_tokens: int = Field(default=128, gt=0)
    temperature:float=Field(default=1.0, gt=0)

class ChatCompletionMessage(BaseModel):
    role:str
    content:str

class ChatCompletionChoice(BaseModel):
    index:int
    message:ChatCompletionMessage
    finish_reason:str

class UsageInfo(BaseModel):
    prompt_tokens:int
    completion_tokens:int
    total_tokens:int

class ChatCompletionResponse(BaseModel):
    id:str
    object:str
    created:int
    model:str
    choices:list[ChatCompletionChoice]
    usage:UsageInfo