from typing import Union, List, Dict, Optional
from enum import Enum
from maga_transformer.config.base_model_config import PyDanticModelBase
from maga_transformer.embedding.embedding_config import EmbeddingGenerateConfig, EmbeddingType

class OpenAIEmbeddingRequest(PyDanticModelBase):
    input: Union[str, List[str]]
    model: str = ""
    encoding_format: str = 'float'
    user: str = ""
    embedding_config: EmbeddingGenerateConfig = EmbeddingGenerateConfig()

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

# not in use yet
class Usage(PyDanticModelBase):
    prompt_tokens: int = 0
    total_tokens: int = 0

class OpenAIEmbeddingResponse(PyDanticModelBase):
    object: str = 'list'
    data: List[EmbeddingResponseFormat] = []
    model: str = ""
    usage: Usage = Usage()

class SimilarityRequest(PyDanticModelBase):
    left: Union[str, List[str]]
    right: Union[str, List[str]]
    embedding_config: EmbeddingGenerateConfig = EmbeddingGenerateConfig()
    model: str = ""
    return_response: bool = False
    
class SimilarityResponse(PyDanticModelBase):
    model: str = ""
    type: EmbeddingType = EmbeddingType.DENSE
    similarity: List[List[float]]
    left_response: Optional[OpenAIEmbeddingResponse] = None
    right_response: Optional[OpenAIEmbeddingResponse] = None
