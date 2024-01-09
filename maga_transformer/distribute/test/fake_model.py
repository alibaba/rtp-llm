import torch
from typing import Any, List, Optional, Iterator

from maga_transformer.model_factory_register import register_model
from maga_transformer.models.base_model import BaseTokenizer, BaseModel, GenerateOutput
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
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

    @torch.no_grad()
    async def generate_stream(
            self, # type: ignore
            inputs: torch.Tensor,
            tokenizer: Any,
            input_lengths: Optional[torch.Tensor],
            images: List[List[str]],
            generate_config: GenerateConfig) -> Iterator[GenerateOutput]:

        yield GenerateOutput(torch.Tensor([[1,2,3,4]]), torch.Tensor([[1]]), torch.Tensor([True]))

register_model("fake_model", FakeModel)
