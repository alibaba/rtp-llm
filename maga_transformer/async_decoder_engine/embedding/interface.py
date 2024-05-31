import torch
from typing import List, Any, Dict, Union, Optional
from maga_transformer.config.base_model_config import PyDanticModelBase

class EngineInputs(PyDanticModelBase):
    token_ids: torch.Tensor
    token_type_ids: torch.Tensor
    input_lengths: torch.Tensor
    config: Dict[str, Any] = {}
    
    @property
    def input_length(self):
        return len(self.token_ids)
    
    @property
    def batch_size(self):
        return len(self.input_lengths)


class EngineOutputs(PyDanticModelBase):
    outputs: Optional[torch.Tensor]
    input_length: int
