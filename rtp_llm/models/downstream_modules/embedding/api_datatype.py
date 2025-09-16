from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from rtp_llm.config.base_model_config import PyDanticModelBase
from rtp_llm.config.generate_config import GenerateConfig


class Usage(PyDanticModelBase):
    prompt_tokens: int = 0
    total_tokens: int = 0


class ExtraConfigs(PyDanticModelBase):
    tokenizer_config: Dict[str, Any] = {}


class ContentPartTypeEnum(str, Enum):
    text = "text"
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


class ImageURL(BaseModel):
    url: str
    detail: Optional[str] = "auto"


class AudioURL(BaseModel):
    url: str


class ContentPart(BaseModel):
    type: ContentPartTypeEnum
    text: Optional[str] = None
    image_url: Optional[ImageURL] = None
    video_url: Optional[ImageURL] = None
    audio_url: Optional[AudioURL] = None
    preprocess_config: Optional[MMPreprocessConfigPart] = None


class RoleEnum(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"
    function = "function"
    tool = "tool"
    observation = "observation"


class ChatMessage(BaseModel):
    role: RoleEnum
    content: Union[str, None, List[ContentPart]] = None
    partial: Optional[bool] = False


class GPTFunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]

    # These parameters are for qwen style function.
    name_for_model: Optional[str] = None
    name_for_human: Optional[str] = None
    description_for_model: Optional[str] = None


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
    aux_info: Optional[bool] = False
    extend_fields: Optional[Dict[str, Any]] = (
        None  # This field is not effective, only for logging.
    )


class OpenAIEmbeddingRequest(PyDanticModelBase):
    prompt: str = ""
    input: Union[
        str, List[str], ContentPart, List[ContentPart], ChatMessage, List[ChatMessage]
    ] = None
    model: str = ""
    encoding_format: str = "float"
    user: str = ""
    extra_configs: ExtraConfigs = ExtraConfigs()


class EmbeddingResponseType(str, Enum):
    DENSE = "embedding"
    SPARSE = "sparse_embedding"
    COLBERT = "colbert_embedding"


"""
List[float] -> dense response
List[List[float]] -> colbert response
Dict[str, float] -> sparse response
str -> for smoke, embedding file path
"""


class EmbeddingResponseFormat(PyDanticModelBase):
    object: EmbeddingResponseType
    embedding: Union[List[float], Dict[str, float], List[List[float]], str]
    index: int


class ALLEmbeddingResponseFormat(PyDanticModelBase):
    object: EmbeddingResponseType
    embedding: Union[List[float], Dict[str, float], List[List[float]], str]
    token_ids: List[int]
    index: int


class OpenAIEmbeddingResponse(PyDanticModelBase):
    object: str = "list"
    data: List[EmbeddingResponseFormat] = []
    model: str = ""
    usage: Usage = Usage()


class ALLEmbeddingResponse(PyDanticModelBase):
    object: str = "list"
    data: List[ALLEmbeddingResponseFormat] = []
    model: str = ""
    usage: Usage = Usage()


class SimilarityRequest(PyDanticModelBase):
    left: Union[List[str], List[ContentPart]]
    right: Union[List[str], List[ContentPart]]
    model: str = ""
    return_response: bool = False


class SimilarityResponse(PyDanticModelBase):
    model: str = ""
    similarity: List[List[float]]
    left_response: Optional[OpenAIEmbeddingResponse] = None
    right_response: Optional[OpenAIEmbeddingResponse] = None


class SparseEmbeddingRequest(OpenAIEmbeddingRequest):
    return_decoded: bool = True


class ColbertEmbeddingRequest(OpenAIEmbeddingRequest):
    pass


class AllEmbeddingRequest(OpenAIEmbeddingRequest):
    normalize: bool = True
