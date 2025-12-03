from typing import Union

import torch

# Type alias for quantization dtype
QuantDtype = Union[None, torch.dtype, str]

from rtp_llm.models_py.modules.common.base.dense_mlp import (
    BertGeluActDenseMLP,
    DenseMLP,
)
from rtp_llm.models_py.modules.common.base.embedding import Embedding
from rtp_llm.models_py.modules.common.base.norm import (
    AddBiasResLayerNorm,
    AddBiasResLayerNormTorch,
    LayerNorm,
    LayerNormTorch,
    RMSNormTorch,
    RMSResNormTorch,
)
from rtp_llm.models_py.modules.common.kvcache_store import WriteCacheStoreOp
from rtp_llm.models_py.modules.common.mha import DECODE_MHA_IMPS, PREFILL_MHA_IMPS
from rtp_llm.models_py.modules.common.mha.attention import CausalAttention
from rtp_llm.models_py.modules.common.mha.base import FMHAImplBase
from rtp_llm.models_py.modules.common.mla import DECODE_MLA_IMPS, PREFILL_MLA_IMPS
from rtp_llm.models_py.modules.common.mla.mla_attention import MlaAttention
from rtp_llm.ops.compute_ops import DeviceType, get_device

device_type = get_device().get_device_type()

if device_type == DeviceType.ROCm:
    import rtp_llm.models_py.modules.rocm_registry
    from rtp_llm.models_py.modules.rocm.mlp import FusedSiluActDenseMLP
    from rtp_llm.models_py.modules.rocm.norm import FusedQKRMSNorm, QKRMSNorm, RMSNorm
    from rtp_llm.models_py.modules.rocm.select_topk import SelectTopk

else:
    import rtp_llm.models_py.modules.cuda_registry
    from rtp_llm.models_py.modules.cuda.mlp import FusedSiluActDenseMLP
    from rtp_llm.models_py.modules.cuda.norm import (
        FusedQKRMSNorm,
        QKRMSNorm,
        RMSNorm,
        RMSResNorm,
    )
    from rtp_llm.models_py.modules.cuda.select_topk import GroupTopK, SelectTopk

__all__ = [
    "BertGeluActDenseMLP",
    "DenseMLP",
    "Embedding",
    "AddBiasResLayerNorm",
    "AddBiasResLayerNormTorch",
    "LayerNorm",
    "LayerNormTorch",
    "RMSNormTorch",
    "RMSResNormTorch",
    "WriteCacheStoreOp",
    "FMHAImplBase",
    "DECODE_MHA_IMPS",
    "PREFILL_MHA_IMPS",
    "DECODE_MLA_IMPS",
    "PREFILL_MLA_IMPS",
    "CausalAttention",
    "MlaAttention",
    "FusedSiluActDenseMLP",
    "FusedQKRMSNorm",
    "QKRMSNorm",
    "RMSNorm",
    "RMSResNorm",
    "SelectTopk",
]
