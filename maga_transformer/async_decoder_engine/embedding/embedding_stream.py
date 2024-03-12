import torch
import numpy
from typing import Any, List, Optional

from maga_transformer.config.generate_config import GenerateConfig
from maga_transformer.config.base_model_config import PyDanticModelBase

class EmbeddingInput(PyDanticModelBase):
    token_ids: List[int]
    token_type_ids: List[int]
    input_length: int
    generate_config: GenerateConfig

class EmbeddingOutput(PyDanticModelBase):
    sentence_embedding: Optional[torch.Tensor] = None
    sparse_embedding: Optional[List[torch.Tensor]] = None
    colbert_embedding: Optional[List[torch.Tensor]] = None
    finished: bool = False

class EmbeddingStream(PyDanticModelBase):
    input: EmbeddingInput
    output: EmbeddingOutput = EmbeddingOutput()
    error_info: str = ""

    def set_error(self, error: str):
        self.error_info = error

    def update(self,
               sentence_embedding: Optional[torch.Tensor],
               sparse_embedding: Optional[List[torch.Tensor]],
               colbert_embedding: Optional[List[torch.Tensor]]):
            self.output.finished = True
            self.output.sentence_embedding = sentence_embedding
            self.output.sparse_embedding = sparse_embedding
            self.output.colbert_embedding = colbert_embedding

    @property
    def finished(self):
        return self.output.finished == True