from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.model_loader.model_weight_info import ModelWeights

from rtp_llm.models_py.module_base import GptModelBase


class GptModelExample(GptModelBase):
    def __init__(self, params: GptInitModelParameters, weight: ModelWeights) -> None:
        super().__init__(params, weight)
        print("GptModelExample initialized")

    def forward(self, hidden):
        return super().forward(hidden)

