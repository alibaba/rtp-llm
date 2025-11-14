from typing import Any, Dict, List

import torch

from rtp_llm.async_decoder_engine.embedding.interface import EngineInputs, EngineOutputs
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.frontend.tokenizer_factory.tokenizers import BaseTokenizer
from rtp_llm.model_loader.weight_module import CustomAtomicWeight
from rtp_llm.models.downstream_modules.classifier.api_datatype import (
    ClassifierRequest,
    ClassifierResponse,
)
from rtp_llm.models.downstream_modules.common_input_generator import (
    CommonInputGenerator,
)
from rtp_llm.models.downstream_modules.custom_module import (
    CustomHandler,
    CustomModule,
    CustomRenderer,
)
from rtp_llm.utils.model_weight import CkptWeightInfo
from rtp_llm.utils.tensor_utils import (
    get_first_token_from_combo_tokens,
    get_last_token_from_combo_tokens,
)
from rtp_llm.utils.util import to_torch_dtype

from .util import load_num_labels


# Normal Classifier
class ClassifierModule(CustomModule):
    def __init__(self, config: ModelConfig, tokenizer: BaseTokenizer):
        super().__init__(config, tokenizer)
        self.renderer = ClassifierRenderer(self.config_, self.tokenizer_)
        self.handler = ClassifierHandler(self.config_)


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


class ClassifierHandler(CustomHandler):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        num_labels = load_num_labels(self.config_.ckpt_path)
        self.linear = torch.nn.Linear(self.config_.hidden_size, num_labels)

    def custom_weight_info(self) -> List[CustomAtomicWeight]:
        w_list = ["classifier.weight", "classifier.bias"]
        weights = []
        for k in w_list:
            weights.append(
                CustomAtomicWeight(CustomAtomicWeight.prefix + k, [CkptWeightInfo(k)])
            )
        return weights

    def init(self, tensor_map: Dict[str, torch.Tensor]):
        data_type = to_torch_dtype(self.config_.data_type)
        self.linear.weight.data = tensor_map["classifier.weight"]
        self.linear.bias.data = tensor_map["classifier.bias"]
        self.linear = self.linear.to(data_type).eval().to(self.device)

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> List[torch.Tensor]:
        # TODO test it
        if self.config_.attn_config.is_causal:
            last_tokens = get_last_token_from_combo_tokens(hidden_states, input_lengths)
            return self.linear(last_tokens)
        else:
            first_tokens = get_first_token_from_combo_tokens(
                hidden_states, input_lengths
            )
            return self.linear(first_tokens)
