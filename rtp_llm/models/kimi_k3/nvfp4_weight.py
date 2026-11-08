"""Bind online NVFP4 to the native K3.1 weight lifecycle."""

from rtp_llm.config.quant_config import ModelOptFp4Config
from rtp_llm.model_loader.weight_module import (
    AtomicWeight,
    CompositeWeight,
    QuantWeight,
)
from rtp_llm.models.kimi_k3.kimi_k3_weight import KimiK3WeightNames as K3W
from rtp_llm.models.kimi_k3.nvfp4_tensor import OnlineNvfp4Projection, nvfp4_scale_keys
from rtp_llm.utils.model_weight import W

ATTENTION_PROJECTION_MODULES = {
    K3W.KDA_INPUT: "kda.input",
    W.linear_attn_f_b_w: "kda.forget",
    W.linear_attn_out_w: "kda.output",
    W.mla_fusedqkrope_w: "dsa.input",
    W.mla_q_b_w: "dsa.query",
    W.attn_o_w: "dsa.output",
}


class KimiK31Nvfp4Weight(OnlineNvfp4Projection, CompositeWeight, QuantWeight):
    @classmethod
    def support(cls, quant_config, src_weight_info):
        return False  # Explicit K3.1 policy; never selected model-wide.

    def __init__(self, source, *, module):
        OnlineNvfp4Projection.__init__(self, source, module=module)
        CompositeWeight.__init__(
            self,
            {
                key: AtomicWeight(key, [])
                for key in (
                    source.name,
                    self.local_key,
                    self.global_key,
                    self.logical_n_key,
                )
            },
            name=source.name,
            quant_config=ModelOptFp4Config(
                bits=4, group_size=16, is_quanted=False, activation_scheme="dynamic"
            ),
        )
