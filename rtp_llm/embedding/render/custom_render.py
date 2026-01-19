from typing import Any, Dict

from pydantic import BaseModel

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.embedding.interface import EngineInputs, EngineOutputs
from rtp_llm.tokenizer_factory.tokenizers import BaseTokenizer


class CustomRenderer(object):
    def __init__(self, config: ModelConfig, tokenizer: BaseTokenizer):
        self.config_ = config
        self.tokenizer_ = tokenizer

    def render_request(self, request_json: Dict[str, Any]) -> BaseModel:
        raise NotImplementedError

    def create_input(self, request: BaseModel) -> EngineInputs:
        raise NotImplementedError

    async def render_response(
        self, request: BaseModel, inputs: EngineInputs, outputs: EngineOutputs
    ) -> Dict[str, Any]:
        raise NotImplementedError

    async def render_log_response(self, response: Dict[str, Any]):
        return response
