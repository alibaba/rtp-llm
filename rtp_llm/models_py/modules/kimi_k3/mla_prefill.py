"""K3 expanded prefill MLA, including reads of cached latent vectors."""

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashinfer_mla import MlaFlashInferPrefillOp
from rtp_llm.models_py.modules.kimi_k3.linear import KimiK3Bf16Linear
from rtp_llm.utils.model_weight import W
from rtp_llm.models_py.modules.kimi_k3.native_mla_prefill import KimiK3TokenspeedPrefill

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashinfer_mla_wrapper import (
    MlaFlashInferPrefillImpl,
)


class KimiK3MlaPrefillOp(MlaFlashInferPrefillOp):
    def _create_prefill_wrapper(self):
        return KimiK3TokenspeedPrefill()

    def _make_kv_b_proj(self, layer_id):
        weight = self.weights[layer_id][W.mla_kv_b_w]
        if weight.is_cuda and weight.dtype == torch.bfloat16:
            return KimiK3Bf16Linear(weight)
        return super()._make_kv_b_proj(layer_id)


class KimiK3MlaPrefillImpl(MlaFlashInferPrefillImpl):
    prefill_op_type = KimiK3MlaPrefillOp

    def __init__(
        self, config, parallelism, weights, inputs, fmha_config, is_cuda_graph
    ):
        if is_cuda_graph:
            raise ValueError(
                "K3 ordinary prefill requires eager planning; verify and draft "
                "updates use the separate paged MLA graph implementation"
            )
        attention = config.getAttentionConfigs(parallelism.get_attn_tp_size())
        inputs.headwise_config = getattr(config, "headwise_config", None)
        # Keep expanded attention for both full and reused prefixes. Native
        # TokenSpeed arithmetic runs under RTP's cache/state planning.
        super().__init__(
            attention,
            inputs,
            weights.weights,
            None,  # K3 uses NoPE.
            fmha_config,
            quant_config=config.quant_config,
            max_seq_len=config.max_seq_len,
            is_cuda_graph=False,
            parallelism_config=parallelism,
            allow_absorb=False,
        )
