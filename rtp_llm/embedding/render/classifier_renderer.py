from typing import Any, Dict

from rtp_llm.async_decoder_engine.embedding.interface import EngineInputs, EngineOutputs
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.embedding.render.classifier.api_datatype import (
    ClassifierRequest,
    ClassifierResponse,
)
from rtp_llm.embedding.render.common_input_generator import CommonInputGenerator
from rtp_llm.embedding.render.custom_render import CustomRenderer
from rtp_llm.frontend.tokenizer_factory.tokenizers import BaseTokenizer


class ClassifierRenderer(CustomRenderer):
    def __init__(self, config: ModelConfig, tokenizer: BaseTokenizer):
        super().__init__(config, tokenizer)
        self.generator = CommonInputGenerator(tokenizer, config)

    def render_request(self, request: Dict[str, Any]):
        return ClassifierRequest(**request)

    def create_input(self, formated_request: ClassifierRequest):
        return self.generator.generate(formated_request.input)

    async def render_response(
        self,
        formated_request: ClassifierRequest,
        inputs: EngineInputs,
        outputs: EngineOutputs,
    ) -> Dict[str, Any]:
        return ClassifierResponse(
            score=[x.tolist() for x in outputs.outputs]
        ).model_dump()
