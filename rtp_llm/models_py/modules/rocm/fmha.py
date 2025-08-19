import torch

from typing import Optional, Any, List
from rtp_llm.ops import PyAttentionInputs, FMHAType
from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from libth_transformer.rtp_llm_ops import CKAttnOp
from libth_transformer.rtp_llm_ops import FusedRopeKVCachePrefillOp

from aiter import flash_attn_func

class FMHAImplBase(object):
    fmha_impl: Any
    fmha_params: Any
    rope_params: Any
    rope_kvcache_impl: Any
    attn_inputs: PyAttentionInputs
    support: bool = False

    def __init__(self,
                 fmha_impl: Any,
                 rope_kvcache_impl: Any,
                 attn_inputs: PyAttentionInputs,
                 init_params: bool = True) -> None:
        self.fmha_impl = fmha_impl
        self.support: bool = self.fmha_impl.support(attn_inputs)
        self.fmha_params = None
        self.rope_params = None
        if self.support and init_params:
            self.rope_kvcache_impl = rope_kvcache_impl
            self.prepare(attn_inputs)
            self.attn_inputs = attn_inputs

    def forward(self, qkv: torch.Tensor, k_cache: Optional[torch.Tensor],
                v_cache: Optional[torch.Tensor]) -> torch.Tensor:
        print('z:', self.fmha_impl, self.fmha_params, flush=True)
        assert self.rope_kvcache_impl is not None and self.rope_params is not None
        fmha_input = self.rope_kvcache_impl.forward(qkv, self.fmha_type(),
                                                    k_cache, self.rope_params)
        assert self.fmha_impl is not None and self.fmha_params is not None
        return aiter.flash_attn_func(fmha_input, k_cache, v_cache, dropout_p=0.0, softmax_scale=None, causal=True)

    @staticmethod
    def fmha_type() -> FMHAType:
        return FMHAType.NONE

     def support(self):
        return True

    def prepare(self, attn_inputs: PyAttentionInputs):
        assert self.fmha_impl is not None
        self.fmha_params = self.fmha_impl.prepare(attn_inputs)
        assert self.rope_kvcache_impl is not None
        self.rope_params = self.rope_kvcache_impl.prepare(attn_inputs)


class FMHAPrefillImplBase(FMHAImplBase):

    def __init__(self, fmha_impl: Any, attn_inputs: PyAttentionInputs,
                 config: GptInitModelParameters) -> None:
        super().__init__(fmha_impl, FusedRopeKVCachePrefillOp(config),
                         attn_inputs)


class CKMHAImpl(FMHAPrefillImplBase):

    def __init__(self, config: GptInitModelParameters,
                 attn_inputs: PyAttentionInputs) -> None:
        super().__init__(CKAttnOp(config), attn_inputs, config)

    @staticmethod
    def fmha_type() -> FMHAType:
        return FMHAType.CK_MHA


PREFILL_MHA_IMPS: List[type[FMHAImplBase]] = [
    CKMHAImpl
]
