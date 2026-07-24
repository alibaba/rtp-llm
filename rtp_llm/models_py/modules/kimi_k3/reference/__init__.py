"""Pure-Torch Kimi K3 correctness and reference-model APIs."""

from rtp_llm.models_py.modules.kimi_k3.kda_state import (
    KDAExecutionMode,
    KimiKDAState,
)
from rtp_llm.models_py.modules.kimi_k3.reference.attn_residual import (
    KimiAttentionResidualMixer,
)
from rtp_llm.models_py.modules.kimi_k3.reference.common import (
    KimiRMSGated,
    KimiRMSNorm,
    SituAndMul,
)
from rtp_llm.models_py.modules.kimi_k3.reference.kda_attention import (
    CausalDepthwiseConv1dReference,
    KimiDeltaAttentionReference,
)
from rtp_llm.models_py.modules.kimi_k3.reference.kda_reference import (
    kimi_kda,
    kimi_kda_chunk,
    kimi_kda_recurrent,
    prepare_kimi_kda_inputs,
)
from rtp_llm.models_py.modules.kimi_k3.reference.mla_attention import (
    KimiMLAAttentionReference,
    KimiMLAState,
)
from rtp_llm.models_py.modules.kimi_k3.reference.mlp import (
    KimiBlockSparseMLPReference,
    KimiMLPReference,
    KimiMoEGateReference,
    KimiSparseMoeBlockReference,
)
from rtp_llm.models_py.modules.kimi_k3.reference.reference_model import (
    KimiK3ReferenceCache,
    KimiK3ReferenceConfig,
    KimiK3ReferenceForCausalLM,
    KimiK3ReferenceModel,
)

__all__ = [
    "CausalDepthwiseConv1dReference",
    "KimiAttentionResidualMixer",
    "KimiBlockSparseMLPReference",
    "KimiDeltaAttentionReference",
    "KimiK3ReferenceCache",
    "KimiK3ReferenceConfig",
    "KimiK3ReferenceForCausalLM",
    "KimiK3ReferenceModel",
    "KimiKDAState",
    "KimiMLAAttentionReference",
    "KimiMLAState",
    "KimiMLPReference",
    "KimiMoEGateReference",
    "KimiRMSGated",
    "KimiRMSNorm",
    "KimiSparseMoeBlockReference",
    "KDAExecutionMode",
    "SituAndMul",
    "kimi_kda",
    "kimi_kda_chunk",
    "kimi_kda_recurrent",
    "prepare_kimi_kda_inputs",
]
