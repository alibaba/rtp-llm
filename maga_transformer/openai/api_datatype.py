import time
import uuid
from enum import Enum
from pydantic import BaseModel, Field
from typing import Union, Optional, List, Dict, Literal, Any
from maga_transformer.models.base_model import AuxInfo

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
    name: Optional[str]
    arguments: Optional[str]

class ToolCallFunction(BaseModel):
    name: Optional[str] = None
    arguments: Optional[str] = None

class ToolCall(BaseModel):
    index: int
    id: Optional[str] = None
    type: str
    function: FunctionCall

class RoleEnum(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"
    function = "function"
    tool = "tool"

class ContentPartTypeEnum(str, Enum):
    text = "text"
    image_url = "image_url"

class ImageURL(BaseModel):
    url: str
    detail: Optional[str] = "auto"

class ContentPart(BaseModel):
    type: ContentPartTypeEnum
    text: Optional[str] = None
    image_url: Optional[ImageURL] = None

class ChatMessage(BaseModel):
    role: RoleEnum
    content: Union[str, List[ContentPart]] = ""
    function_call: Optional[FunctionCall] = None
    tool_calls: Optional[List[ToolCall]] = None

# NOTE: according to openai api definition, `function_call` is deprecated, and replaced by `tool_calls`.
# see `openai/types/chat/chat_completion_chunk.py`

# TODO: maybe also implement Qwen Style function call.
# See https://github.com/QwenLM/Qwen/blob/35023b6f2a932bde6ed27f21ec03164ccf09a25f/examples/function_call_examples.py#L47

class GPTFunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]

    # These parameters are for qwen style function.
    name_for_model: Optional[str] = None
    name_for_human: Optional[str] = None
    description_for_model: Optional[str] = None

class ExtraConfigs(BaseModel):
    top_k: Optional[int] = None
    repitition_penalty: Optional[float] = None
    max_new_tokens: Optional[int] = None
    timeout_ms: Optional[int] = None

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    functions: Optional[List[GPTFunctionDefinition]] = None
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = Field(default_factory=list)
    stream: Optional[bool] = False
    user: Optional[str] = None
    seed: Optional[int] = None

    # ---- These functions are not implemented yet.
    # n: Optional[int] = 1
    # presence_penalty: Optional[float] = 0.0
    # frequency_penalty: Optional[float] = 0.0
    # logit_bias: Optional[Dict[str, float]] = None

    # ---- These params are hacked for our framework, not standard.
    extra_configs: Optional[ExtraConfigs] = None
    private_request: bool = False
    trace_id: Optional[str] = None
    chat_id: Optional[str] = None
    debug_info: Optional[bool] = False
    aux_info: Optional[bool] = False
    extend_fields: Optional[Dict[str, Any]] = None # This field is not effective, only for logging.

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

class RendererInfo(BaseModel):
    class_name: str
    renderer_model_type: str
    extra_stop_word_ids_list: List[List[int]]
    extra_stop_words_list: List[str]
    template: Optional[str] = None

class DebugInfo(BaseModel):
    input_prompt: str
    input_ids: List[int]
    input_images: List[str]
    tokenizer_info: str
    max_seq_len: int
    eos_token_id: Optional[int]
    stop_word_ids_list: List[List[int]]
    stop_words_list: List[str]
    renderer_info: RendererInfo

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
    debug_info: Optional[DebugInfo] = None
    aux_info: Optional[AuxInfo] = None

class DeltaMessage(BaseModel):
    role: Optional[RoleEnum] = None
    content: Optional[str] = ""
    function_call: Optional[FunctionCall] = None
    tool_calls: Optional[List[ToolCall]] = None

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
    debug_info: Optional[DebugInfo] = None
    aux_info: Optional[AuxInfo] = None
