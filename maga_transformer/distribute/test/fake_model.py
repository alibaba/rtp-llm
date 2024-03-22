import torch
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
    def __init__(self, config: GptInitModelParameters):
        super().__init__()
        self.tokenizer = FakeTokenizer()
        self.config = config

    @staticmethod
    def create_config(ckpt_path: str, **kwargs: Any) -> Any:
        config = GptInitModelParameters(head_num = 2, size_per_head = 128,
                                             layer_num = 2, max_seq_len = 2048, vocab_size = 500000, multi_task_prompt=None)
        config.lora_infos = None
        config.multi_task_prompt = None
        config.is_sparse_head = False
        config.use_medusa = False
        return config

    @classmethod
    def from_config(cls, config: Any) -> 'FakeModel':
        return cls(config)

register_model("fake_model", FakeModel)
