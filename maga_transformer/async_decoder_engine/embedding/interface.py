import torch
from typing import List, Any, Dict, Union, Optional
from maga_transformer.config.base_model_config import PyDanticModelBase
from maga_transformer.utils.multimodal_util import MMUrlType, MultimodalInput

class EngineInputs():
    token_ids: torch.Tensor
    token_type_ids: torch.Tensor
    input_lengths: torch.Tensor
    config: Dict[str, Any] = {}
    multimodal_inputs: List[MultimodalInput] = []

    def __init__(self,
                 token_ids: torch.Tensor,
                 token_type_ids: torch.Tensor,
                 input_lengths: torch.Tensor,
                 config: Dict[str, Any] = {},
                 input_urls: List[str] = [],
                 input_urls_type: List[MMUrlType] = []):
        self.token_ids = token_ids
        self.token_type_ids = token_type_ids
        self.input_lengths = input_lengths
        self.config = config
        self.multimodal_inputs = []
        if len(input_urls_type) == 0:
            input_urls_type = [MMUrlType.DEFAULT] * len(input_urls)
        elif len(input_urls_type) != len(input_urls):
            raise Exception(f"the number of multimodal input types must match url, now types {len(input_urls_type)} urls {len(input_urls)}")
        for url, type in zip(input_urls, input_urls_type):
            self.multimodal_inputs.append(MultimodalInput(url, type))

    @property
    def input_length(self):
        return len(self.token_ids)

    @property
    def batch_size(self):
        return len(self.input_lengths)


class EngineOutputs(PyDanticModelBase):
    outputs: Optional[Union[List[Dict[str, torch.Tensor]], torch.Tensor, List[torch.Tensor]]]
    input_length: int
