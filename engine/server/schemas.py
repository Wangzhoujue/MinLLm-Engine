
from pydantic import BaseModel,Field
from typing import Literal
from typing import Optional



class ChatMessage(BaseModel):
    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str

class StreamOptions(BaseModel):
    include_usage: bool = False

class ChatCompletionRequest(BaseModel):
    model:str
    messages:list[ChatMessage]
    max_tokens: int = Field(default=128, gt=0)
    temperature:float=Field(default=1.0, gt=0)
    stream: bool = False
    stream_options: Optional[StreamOptions] = None


class ChatCompletionMessage(BaseModel):
    role: str
    content: str

class ChatCompletionDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None

class ChatCompletionChoice(BaseModel):
    index:int
    message:ChatCompletionMessage
    finish_reason: Optional[str] = None

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

class ChatCompletionChunkChoice(BaseModel):
    index: int
    delta: ChatCompletionDelta
    finish_reason: Optional[str] = None

class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionChunkChoice]
    usage: Optional[UsageInfo] = None

