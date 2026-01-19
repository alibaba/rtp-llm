from typing import TYPE_CHECKING, Any, Dict, List, Union

import torch
from pydantic import BaseModel

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.embedding.interface import EngineInputs, EngineOutputs
from rtp_llm.embedding.render.custom_render import CustomRenderer
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.model_loader.weight_module import CustomAtomicWeight
from rtp_llm.ops import EmbeddingCppOutput
from rtp_llm.tokenizer_factory.tokenizers import BaseTokenizer

"""
用于多种多样的下游任务
"""


class CustomModule(object):
    renderer: "CustomRenderer"
    handler: "CustomHandler"

    def __init__(self, config: ModelConfig, tokenizer: BaseTokenizer):
        self.config_ = config
        self.tokenizer_ = tokenizer

    def create_cpp_handler(self) -> Any:
        raise NotImplementedError("not support cpp handler")

    def get_renderer(self, request: Dict[str, Any]) -> "CustomRenderer":
        return self.renderer

    def get_handler(self) -> "CustomHandler":
        return self.handler

    def get_custom_weight_info(self) -> List[CustomAtomicWeight]:
        return self.handler.custom_weight_info()

    def init(self, weight: ModelWeights):
        tensor_map: Dict[str, torch.Tensor] = {}
        for weight_info in self.get_custom_weight_info():
            if weight_info.name.startswith(CustomAtomicWeight.prefix):
                name = weight_info.name.replace(CustomAtomicWeight.prefix, "", 1)
            else:
                name = weight_info.name
            tensor_map[name] = weight.get_global_weight(weight_info.name)
        self.handler.init(tensor_map)


class CustomHandler(object):
    def __init__(self, config: ModelConfig):
        self.config_ = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.need_post_process = False

    def custom_weight_info(self) -> List[CustomAtomicWeight]:
        return []

    # for cpp
    def init_cpp_handler(self) -> None:
        pass

    def init(self, tensor_map: Dict[str, torch.Tensor]) -> None:
        pass

    # 输入:
    # input_ids: [token_len]
    # hidden_states: [token_len, hidden_size]
    # input_lengths: [batch_size]
    # 输出:
    # [batch_size], 由endpoint格式化返回
    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> Union[torch.Tensor, List[Any]]:
        raise NotImplementedError

    # specify required args for extended_forward
    def extend_forward_args(self) -> List[str]:
        return ["input_lengths", "input_ids", "hidden_states"]

    # extended_forward
    # input_lengths: [batch_size]
    # input_ids: [token_len]
    # hidden_states: [token_len, hidden_size]
    # moe_gating: list of Optional[Tensor], with length = layer_num, [token_len, expert_num]
    def extend_forward(self, **kwargs: Any) -> Union[torch.Tensor, List[Any]]:
        return self.forward(
            input_lengths=kwargs["input_lengths"],
            input_ids=kwargs["input_ids"],
            hidden_states=kwargs["hidden_states"],
        )

    def post_process(
        self, request: Any, batch_output: EmbeddingCppOutput
    ) -> EmbeddingCppOutput:
        return batch_output
