import torch
from typing import List, Dict, Any, Union, Callable
from pydantic import BaseModel

from transformers import PreTrainedTokenizerBase
from maga_transformer.utils.util import to_torch_dtype
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.async_decoder_engine.embedding.interface import EngineInputs, EngineOutputs

'''
用于多种多样的下游任务
'''
class CustomModule(object):
    renderer: 'CustomRenderer'
    handler: 'CustomHandler'
    def __init__(self, config: GptInitModelParameters, tokenizer: PreTrainedTokenizerBase):
        self.config_ = config
        self.tokenizer_ = tokenizer

    def create_cpp_handler(self) -> Any:
        raise NotImplementedError("not support cpp handler")

    def get_renderer(self, request: Dict[str, Any]) -> 'CustomRenderer':
        return self.renderer

    def get_handler(self) -> 'CustomHandler':
        return self.handler

class CustomHandler(object):
    def __init__(self, config: GptInitModelParameters):
        self.config_ = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def tensor_info(self) -> List[str]:
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
    def forward(self, input_ids: torch.Tensor, hidden_states: torch.Tensor, input_lengths: torch.Tensor) -> Union[torch.Tensor, List[Any]]:
        raise NotImplementedError

    def post_process(self, request: Any, batch_output: EngineOutputs) -> EngineOutputs:
        return batch_output

class CustomRenderer(object):
    def __init__(self, config: GptInitModelParameters, tokenizer: PreTrainedTokenizerBase):
        self.config_ = config
        self.tokenizer_ = tokenizer

    def render_request(self, request_json: Dict[str, Any]) -> BaseModel:
        raise NotImplementedError

    def create_input(self, request: BaseModel) -> EngineInputs:
        raise NotImplementedError

    async def render_response(self, request: BaseModel, inputs: EngineInputs, outputs: EngineOutputs) -> Dict[str, Any]:
        raise NotImplementedError

    async def render_log_response(self, response: Dict[str, Any]):
        return response
