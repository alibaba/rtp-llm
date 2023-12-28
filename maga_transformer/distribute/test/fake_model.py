import torch
from typing import Any, List, Optional, Iterator

from maga_transformer.model_factory import register_model
from maga_transformer.models.base_model import BaseTokenizer, BaseModel, GenerateOutput
from maga_transformer.utils.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.config.generate_config import GenerateConfig

class FakeTokenizer(BaseTokenizer):
    def encode(self, inputs: List[str]) -> List[int]:
        return [1,2,3,4]

    def decode(self, tokens: List[int]) -> str:
        return ["fake output"]

class FakeConfig(object):
    def __init__(self):
        self.eos_token_id = 1

class FakeModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.tokenizer = FakeTokenizer()
        self.config = GptInitModelParameters(head_num = 2, size_per_head = 128,
                                             layer_num = 2, max_seq_len = 2048, vocab_size = 500000)
        self.config.special_tokens.eos_token_id = 1

    @staticmethod
    def create_config(ckpt_path: str, **kwargs: Any) -> Any:
        return GptInitModelParameters(head_num = 2, size_per_head = 128,
                                             layer_num = 2, max_seq_len = 2048, vocab_size = 500000)

    @classmethod
    def from_config(cls, config: Any) -> 'FakeModel':
        return cls()

    @torch.no_grad()
    async def generate_stream(
            self, # type: ignore
            inputs: torch.Tensor,
            input_lengths: Optional[torch.Tensor],
            images: List[List[str]],
            generate_config: GenerateConfig) -> Iterator[GenerateOutput]:

        yield GenerateOutput(torch.Tensor([[1,2,3,4]]), torch.Tensor([[1]]), torch.Tensor([True]))

register_model("fake_model", FakeModel)
