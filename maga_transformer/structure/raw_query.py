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