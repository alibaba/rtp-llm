import copy
import torch
import logging
import numpy as np
from typing import Any, List, Optional
from maga_transformer.config.generate_config import GenerateConfig
from maga_transformer.tokenizer.tokenizer_base import TokenizerBase

#TODO(xinfei.sxf) add comment
# move raw query file
'''
input_tokens:    prompt tokens
images:          multimodal param for image
generate_config: config for sampling
'''

class RawQuery:
    def __init__(
            self,
            input_token_ids: torch.Tensor,
            input_lengths: Optional[torch.Tensor],
            images: List[List[str]],
            generate_config: GenerateConfig,
            tokenizer:  TokenizerBase
    ) -> None:
        self.input_token_ids = input_token_ids
        self.input_lengths = input_lengths
        self.images = images
        self.generate_config = generate_config
        self.tokenizer = tokenizer
        self.check()
    
    def check(self):
        if self.generate_config.adapter_name and not isinstance(self.generate_config.adapter_name, list):
            raise Exception(f"raw query generate config's adapter_name type error {type(self.generate_config.adapter_name)}")
    
    def validate(self, index):
        assert index < self.batch_size, f"index {index} is out of range"
    
    @property
    def batch_size(self) -> int:
        return self.input_token_ids.shape[0]
    
    def get_tokens_id(self, index) -> torch.Tensor:
        self.validate(index)
        return self.input_token_ids[index, :int(self.input_lengths[index])]
    
    def get_adapter_name(self, index) -> torch.Tensor:
        self.validate(index)
        adapter_name = ""
        if self.generate_config.adapter_name:
            adapter_name = self.generate_config.adapter_name[index]
        return adapter_name