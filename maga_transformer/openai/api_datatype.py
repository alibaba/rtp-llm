import time
import uuid
from enum import Enum
from pydantic import BaseModel, Field
from typing import Union, Optional, List, Dict, Literal, Any

class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "owner"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: Optional[list] = None

class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = []

class FunctionCall(BaseModel):
    name: str
    arguments: str

class ToolCall(BaseModel):
    id: str
    type: str
    function: FunctionCall

class RoleEnum(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"
    function = "function"
    tool = "tool"

class ChatMessage(BaseModel):
    role: RoleEnum
    content: Optional[Union[str, Dict[str, Any]]]                     # Dict is for function role
    tool_calls: Optional[List[ToolCall]] = None                       # tool call request from model
    tool_call_id: Optional[str] = None                                # tool call id for tool call response
    name: Optional[str] = None                                        # function name for tool call response

class ChatCompletionRequest(BaseModel):
    model: Optional[str]
    messages: List[ChatMessage]
    functions: Optional[List[Dict[str, Any]]] = None
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    max_tokens: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = Field(default_factory=list)
    stream: Optional[bool] = False
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None

class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0

class LogProbs(BaseModel):
    text_offset: List[int] = Field(default_factory=list)
    token_logprobs: List[Optional[float]] = Field(default_factory=list)
    tokens: List[str] = Field(default_factory=list)
    top_logprobs: Optional[List[Optional[Dict[int, float]]]] = None

class FinisheReason(str, Enum):
    stop = "stop"
    length = "length"
    function_call = "function_call"

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[FinisheReason] = None

class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chat-")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = ""
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo

class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = ""

class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[FinisheReason] = None

class ChatCompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chat")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: Optional[str] = None
    choices: List[ChatCompletionResponseStreamChoice]
    usage: Optional[UsageInfo] = Field(default=None)
