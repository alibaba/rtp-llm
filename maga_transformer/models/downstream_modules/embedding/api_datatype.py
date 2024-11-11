from typing import Union, List, Dict, Optional, Any
from enum import Enum
from maga_transformer.config.base_model_config import PyDanticModelBase
from pydantic import BaseModel

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

class OpenAIEmbeddingRequest(PyDanticModelBase):
    input: Union[str, List[str], ContentPart, List[ContentPart]]
    model: str = ""
    encoding_format: str = 'float'
    user: str = ""
    extra_configs: ExtraConfigs = ExtraConfigs()

class EmbeddingResponseType(str, Enum):
    DENSE = "embedding"
    SPARSE = "sparse_embedding"
    COLBERT = "colbert_embedding"

'''
List[float] -> dense response
List[List[float]] -> colbert response
Dict[str, float] -> sparse response
str -> for smoke, embedding file path
'''
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
    object: str = 'list'
    data: List[EmbeddingResponseFormat] = []
    model: str = ""
    usage: Usage = Usage()

class ALLEmbeddingResponse(PyDanticModelBase):
    object: str = 'list'
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