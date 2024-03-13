from typing import Union, List
from maga_transformer.config.base_model_config import PyDanticModelBase

class OpenAIEmbeddingRequestFormat(PyDanticModelBase):
    input: Union[str, List[str]]
    model: str
    encoding_format: str = 'float'
    user: str = ""

class SingleEmbeddingFormat(PyDanticModelBase):
    object: str = 'embedding'
    embedding: Union[List[float], str]
    index: int

# not in use yet
class Usage(PyDanticModelBase):
    prompt_tokens: int = 0
    total_tokens: int = 0

class OpenAIEmbeddingResponseFormat(PyDanticModelBase):
    object: str = 'list'
    data: List[SingleEmbeddingFormat]
    model: str
    usage: Usage = Usage()