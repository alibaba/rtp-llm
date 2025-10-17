from typing import Any, Dict, List, Optional, Union

import torch

from rtp_llm.config.base_model_config import PyDanticModelBase
from rtp_llm.utils.base_model_datatypes import MultimodalInput


class EngineInputs:
    token_ids: torch.Tensor
    token_type_ids: torch.Tensor
    input_lengths: torch.Tensor
    config: Dict[str, Any] = {}
    multimodal_inputs: List[MultimodalInput] = []

    def __init__(
        self,
        token_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        input_lengths: torch.Tensor,
        config: Dict[str, Any] = {},
        multimodal_inputs: List[MultimodalInput] = [],
    ):
        self.token_ids = token_ids
        self.token_type_ids = token_type_ids
        self.input_lengths = input_lengths
        self.config = config
        self.multimodal_inputs = multimodal_inputs

    @property
    def input_length(self):
        return len(self.token_ids)

    @property
    def batch_size(self):
        return len(self.input_lengths)


class EngineOutputs(PyDanticModelBase):
    outputs: Optional[
        Union[List[Dict[str, torch.Tensor]], torch.Tensor, List[torch.Tensor]]
    ]
    input_length: int
