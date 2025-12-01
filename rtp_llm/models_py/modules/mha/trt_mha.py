import logging

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.ops import FMHAType
from rtp_llm.ops.compute_ops import PyAttentionInputs
from rtp_llm.models_py.modules.mha.base import (
    FMHAPrefillImplBase,
    PREFILL_MHA_IMPS,
)


try:
    from rtp_llm.ops.compute_ops import TRTAttnOp, FusedRopeKVCachePrefillOpQOut

    class TRTMHAImpl(FMHAPrefillImplBase):

        def __init__(
            self, config: GptInitModelParameters, attn_inputs: PyAttentionInputs
        ) -> None:
            super().__init__(
                TRTAttnOp(config.gpt_init_params),
                FusedRopeKVCachePrefillOpQOut(config.gpt_init_params),
                attn_inputs,
            )

        @staticmethod
        def fmha_type() -> FMHAType:
            return FMHAType.TRT_V2

        def support_cuda_graph(self) -> bool:
            return True

    PREFILL_MHA_IMPS.append(TRTMHAImpl)
    # PREFILL_MHA_IMPS.insert(0, TRTMHAImpl)

except ImportError:
    logging.info("TRTMHAImpl not available, skipped.")
