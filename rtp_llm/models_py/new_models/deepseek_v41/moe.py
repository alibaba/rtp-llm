"""V4.1's optional vision-token router bias on the shared main MoE owner."""

import torch

from rtp_llm.models_py.new_models.deepseek_v4.moe import OwnedFp8Fp4MoE


class OwnedV41MoE(OwnedFp8Fp4MoE):
    def __init__(self, *args, has_vision_router_bias=False, **kwargs):
        if type(has_vision_router_bias) is not bool:
            raise TypeError(
                "Vision router presence must be determined from checkpoint metadata"
            )
        super().__init__(*args, **kwargs)
        if has_vision_router_bias:
            self._parameter("router_bias_vl", (self.num_experts,), torch.float32)
            self._expected.add("gate.bias_vl")

    def _target(self, name):
        if self._relative_name(name) == "gate.bias_vl" and hasattr(
            self, "router_bias_vl"
        ):
            return self.router_bias_vl
        return super()._target(name)
