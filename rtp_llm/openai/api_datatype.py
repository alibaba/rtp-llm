import time
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.utils.base_model_datatypes import AuxInfo


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


class ToolCall(BaseModel):
    # 参照 openai 官方api definition
    index: Optional[int] = None
    id: Optional[str] = None
    type: str
    function: FunctionCall


class RoleEnum(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"
    function = "function"
    tool = "tool"
    observation = "observation"


class ContentPartTypeEnum(str, Enum):
    text = "text"
    igraph = "igraph"
    image_url = "image_url"
    video_url = "video_url"
    audio_url = "audio_url"


class MMPreprocessConfigPart(BaseModel):
    resized_width: Optional[int] = None
    resized_height: Optional[int] = None
    min_pixels: Optional[int] = None
    max_pixels: Optional[int] = None
    fps: Optional[int] = None
    min_frames: Optional[int] = None
    max_frames: Optional[int] = None
    crop_positions: Optional[str] = None
    mm_timeout_ms: int = 30000


class IgraphInfo(BaseModel):
    table_name: str
    item_id: str


class ImageURL(BaseModel):
    url: str
    detail: Optional[str] = "auto"


class AudioURL(BaseModel):
    url: str


class ContentPart(BaseModel):
    type: ContentPartTypeEnum
    text: Optional[str] = None
    igraph: Optional[IgraphInfo] = None
    image_url: Optional[ImageURL] = None
    video_url: Optional[ImageURL] = None
    audio_url: Optional[AudioURL] = None
    preprocess_config: Optional[MMPreprocessConfigPart] = None


class ChatMessage(BaseModel):
    role: RoleEnum
    content: Union[str, None, List[ContentPart]] = ""
    reasoning_content: Optional[str] = None
    function_call: Optional[FunctionCall] = None
    tool_calls: Optional[List[ToolCall]] = None
    partial: Optional[bool] = False
    tool_call_id: Optional[str] = None


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


class GPTToolDefinition(BaseModel):
    # 目前仅考虑type为function的tool
    type: str = "function"
    function: GPTFunctionDefinition


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    functions: Optional[List[GPTFunctionDefinition]] = None
    tools: Optional[List[GPTToolDefinition]] = None
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = Field(default_factory=list)
    stream: Optional[bool] = False
    user: Optional[str] = None
    seed: Optional[int] = None
    n: Optional[int] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None

    # ---- These functions are not implemented yet.
    # presence_penalty: Optional[float] = 0.0
    # frequency_penalty: Optional[float] = 0.0
    # logit_bias: Optional[Dict[str, float]] = None

    # ---- These params are hacked for our framework, not standard.
    extra_configs: Optional[GenerateConfig] = None
    private_request: bool = False
    trace_id: Optional[str] = None
    chat_id: Optional[str] = None
    template_key: Optional[str] = None
    user_template: Optional[str] = None
    debug_info: Optional[bool] = False
    aux_info: Optional[bool] = True
    extend_fields: Optional[Dict[str, Any]] = (
        None  # This field is not effective, only for logging.
    )
    master_info: Optional[Dict[str, Any]] = None
    chat_template_kwargs: Optional[Dict[str, Any]] = None

    @staticmethod
    def is_openai_request(request: Dict[str, Any]):
        return "messages" in request

    def get_chat_template_kwargs(self):
        if (
            self.extra_configs is not None
            and self.extra_configs.chat_template_kwargs is not None
        ):
            return self.extra_configs.chat_template_kwargs
        else:
            return self.chat_template_kwargs

    def disable_thinking(self):
        if (
            self.get_chat_template_kwargs() is not None
            and self.get_chat_template_kwargs().get("enable_thinking", True) is False
        ):
            return True
        else:
            return False


class CompletionTokensDetails(BaseModel):
    audio_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None


class PromptTokensDetails(BaseModel):
    audio_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0
    completion_tokens_details: Optional[CompletionTokensDetails] = None
    prompt_tokens_details: Optional[PromptTokensDetails] = None


class TopLogprob(BaseModel):
    token: str
    bytes: Optional[List[int]] = None
    logprob: float


class ChatCompletionTokenLogprob(BaseModel):
    token: str
    bytes: Optional[List[int]] = None
    logprob: float
    top_logprobs: List[TopLogprob]


class ChoiceLogprobs(BaseModel):
    content: Optional[List[ChatCompletionTokenLogprob]] = None
    refusal: Optional[List[ChatCompletionTokenLogprob]] = None


class FinisheReason(str, Enum):
    stop = "stop"
    length = "length"
    function_call = "function_call"
    tool_calls = "tool_calls"


class RendererInfo(BaseModel):
    class_name: str
    renderer_model_type: str
    extra_stop_word_ids_list: List[List[int]]
    extra_stop_words_list: List[str]
    template: Optional[Union[str, Dict[str, str]]] = None


class DebugInfo(BaseModel):
    input_prompt: str
    input_ids: List[int]
    input_urls: List[str]
    tokenizer_info: str
    max_seq_len: int
    eos_token_id: Optional[int]
    stop_word_ids_list: List[List[int]]
    stop_words_list: List[str]
    renderer_info: RendererInfo
    generate_config: GenerateConfig
    output_ids: Optional[List[List[int]]] = None
    raw_output: Optional[List[str]] = None


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[FinisheReason] = None
    logprobs: Optional[ChoiceLogprobs] = None


class ChatCompletionExtraOutputs(BaseModel):
    hidden_states: Optional[Union[List[float], List[List[float]]]] = None
    all_hidden_states: Optional[Union[List[float], List[List[float]]]] = None
    loss: Optional[Union[float, List[float]]] = None
    logits: Optional[Union[List[float], List[List[float]]]] = None
    output_ids: Optional[List[List[int]]] = None
    input_ids: Optional[List[List[int]]] = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chat-")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = ""
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo
    debug_info: Optional[Union[DebugInfo, str]] = None
    aux_info: Optional[AuxInfo] = None
    extra_outputs: Optional[ChatCompletionExtraOutputs] = None


class DeltaMessage(BaseModel):
    role: Optional[RoleEnum] = None
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    function_call: Optional[FunctionCall] = None
    tool_calls: Optional[List[ToolCall]] = None


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[FinisheReason] = None
    logprobs: Optional[ChoiceLogprobs] = None


class ChatCompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chat")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: Optional[str] = None
    choices: List[ChatCompletionResponseStreamChoice]
    usage: Optional[UsageInfo] = Field(default=None)
    debug_info: Optional[Union[DebugInfo, str]] = None
    aux_info: Optional[AuxInfo] = None
    extra_outputs: Optional[ChatCompletionExtraOutputs] = None
