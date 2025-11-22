from typing import Any, List

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.models.base_model import BaseModel


class FakeTokenizer(object):
    def encode(self, inputs: List[str]) -> List[int]:
        return [1, 2, 3, 4]

    def decode(self, tokens: List[int]) -> str:
        return "fake output"


class FakeConfig(object):
    def __init__(self):
        self.eos_token_id = 1


class FakeModel(BaseModel):
    def load_tokenizer(self):
        self.tokenizer = FakeTokenizer()

    def init_misc(self):
        pass

    def load(self, ckpt_path: str):
        pass

    def stop(self):
        pass

    @classmethod
    def _create_config(cls, ckpt_path: str) -> ModelConfig:
        config = ModelConfig()
        config.head_num_ = 2
        config.size_per_head_ = 128
        config.num_layers = 2
        config.max_seq_len = 2048
        config.vocab_size = 500000
        config.lora_infos = {}
        config.is_sparse_head_ = False
        return config

    @classmethod
    def from_config(cls, config: Any) -> "FakeModel":
        return cls(config)


register_model("fake_model", FakeModel)
