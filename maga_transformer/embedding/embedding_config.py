from enum import Enum
from typing import Union, List, Dict, Optional
from maga_transformer.config.base_model_config import PyDanticModelBase

class EmbeddingType(str, Enum):
    DENSE = "dense"
    SPARSE = "sparse"
    COLBERT = "colbert"

class EmbeddingGenerateConfig(PyDanticModelBase):
    type: EmbeddingType = EmbeddingType.DENSE
    do_normalize: bool = True
    
    def __eq__(self, other: 'EmbeddingGenerateConfig'):
        return self.type == other.type and self.do_normalize == other.do_normalize