"""Example custom head for LANGUAGE_MODEL context steps.

Configure CUSTOM_OUTPUT_PROCESSOR=custom_output_processor.py and
CUSTOM_OUTPUT_TOKEN_POSITION=-1. Relative processor paths use CHECKPOINT_PATH.
Alternatively, select the last occurrence of a model-specific token with
CUSTOM_OUTPUT_TRACKED_TOKEN_ID (mutually exclusive with position selection).

The checkpoint must contain this example's score_head.dense/out weights.
Choose the hidden-state stage used to train the head. Keep output tensors on
the device; the engine transfers them and returns custom_output unchanged.
OpenAI-compatible responses expose it as extra_outputs.custom_output.
Return a nonempty [context_batch] or [context_batch, width] tensor with dtype
float32, float16, bfloat16 or int32. Token selectors refer to the original input;
the engine preserves text-token identity across prefix insertion and multimodal
expansion. Selecting a multimodal placeholder that is replaced is unsupported.
"""

from typing import Any, Dict, List

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.weight_module import CustomAtomicWeight
from rtp_llm.models.downstream_modules.custom_module import (
    CustomHandler,
    CustomModule,
    HiddenStateStage,
    Trigger,
)
from rtp_llm.utils.model_weight import CkptWeightInfo
from rtp_llm.utils.util import to_torch_dtype


class ScoreHandler(CustomHandler):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        hidden_size = self.config_.hidden_size
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, 1),
        )

    def custom_weight_info(self) -> List[CustomAtomicWeight]:
        w_list = [
            "score_head.dense.weight",
            "score_head.dense.bias",
            "score_head.out.weight",
            "score_head.out.bias",
        ]
        return [
            CustomAtomicWeight(CustomAtomicWeight.prefix + k, [CkptWeightInfo(k)])
            for k in w_list
        ]

    def init(self, tensor_map: Dict[str, torch.Tensor]):
        self.mlp[0].weight.data = tensor_map["score_head.dense.weight"]
        self.mlp[0].bias.data = tensor_map["score_head.dense.bias"]
        self.mlp[2].weight.data = tensor_map["score_head.out.weight"]
        self.mlp[2].bias.data = tensor_map["score_head.out.bias"]
        data_type = to_torch_dtype(self.config_.data_type)
        self.mlp = self.mlp.to(data_type).eval().to(self.device)

    def extend_forward_args(self) -> List[str]:
        return ["selected_hidden_states"]

    def trigger_mode(self) -> Trigger:
        return Trigger.CONTEXT

    def hidden_state_stage(self) -> HiddenStateStage:
        # This example uses the final normalized output. Switch to PRE_FINAL_NORM
        # only when the head was trained on the last decoder block output.
        return HiddenStateStage.POST_FINAL_NORM

    def extend_forward(self, **kwargs: Any) -> torch.Tensor:
        # One selected hidden-state row per context request.
        selected_hidden = kwargs["selected_hidden_states"]
        with torch.no_grad():
            return self.mlp(selected_hidden)  # [context_batch, 1]


class ScoreModule(CustomModule):
    def __init__(self, config: ModelConfig, tokenizer: Any):
        super().__init__(config, tokenizer)
        self.handler = ScoreHandler(config)


def create_custom_module(config: ModelConfig, tokenizer: Any) -> CustomModule:
    return ScoreModule(config, tokenizer)
