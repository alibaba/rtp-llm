import torch
import logging
from typing import Optional, Iterator, List, Any, Generator, AsyncGenerator, Dict, Union

from maga_transformer.model_factory_register import register_model
from maga_transformer.models.base_model import BaseModel, GenerateOutput
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.config.generate_config import GenerateConfig

class FakeTokenizer(object):
    def encode(self, inputs: List[str]) -> List[int]:
        return [1,2,3,4]

    def decode(self, tokens: List[int]) -> str:
        return "fake output"

class FakeConfig(object):
    def __init__(self):
        self.eos_token_id = 1

class FakeModel(BaseModel):
    @property
    def model_runtime_meta(self) -> str:
        return ""

    def load_tokenizer(self):
        self.tokenizer = FakeTokenizer()

    def init_misc(self):
        pass

    def load(self, ckpt_path: str):
        pass

    def ready(self):
        return True
    
    def stop(self):
        pass

    @classmethod
    def create_config(cls, model_config: Any) -> GptInitModelParameters:
        config = GptInitModelParameters(head_num = 2, size_per_head = 128,
                                             layer_num = 2, max_seq_len = 2048, vocab_size = 500000, multi_task_prompt=None)
        config.lora_infos = None
        config.multi_task_prompt = None
        config.is_sparse_head = False
        config.tokenizer_path = model_config.tokenizer_path
        return config

    @classmethod
    def from_config(cls, config: Any) -> 'FakeModel':
        return cls(config)

register_model("fake_model", FakeModel)
