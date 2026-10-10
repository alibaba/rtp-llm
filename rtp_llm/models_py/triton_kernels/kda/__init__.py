"""KDA recurrent decode kernels; chunk prefill is supplied by cuLA."""

from rtp_llm.models_py.triton_kernels.kda.fused_recurrent import (
    fused_recurrent_kda,
)

__all__ = ["fused_recurrent_kda"]
