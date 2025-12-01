from typing import Union

import torch

# Type alias for quantization dtype
QuantDtype = Union[None, torch.dtype, str]


from rtp_llm.ops.compute_ops import DeviceType, get_device

device_type = get_device().get_device_type()

if device_type == DeviceType.ROCm:
    import rtp_llm.models_py.modules.rocm_registry
    from rtp_llm.models_py.modules.rocm.mlp import FusedSiluActDenseMLP
    from rtp_llm.models_py.modules.rocm.norm import FusedQKRMSNorm, RMSNorm
    from rtp_llm.models_py.modules.rocm.select_topk import SelectTopk

else:
    from rtp_llm.models_py.modules.norm import FusedQKRMSNorm, RMSNorm
    from rtp_llm.models_py.modules.mlp import FusedSiluActDenseMLP
    from rtp_llm.models_py.modules.select_topk import SelectTopk, GroupTopK
    import rtp_llm.models_py.modules.cuda_registry

from rtp_llm.models_py.modules.common.mha import DECODE_MHA_IMPS, PREFILL_MHA_IMPS
from rtp_llm.models_py.modules.common.mha.attention import CausalAttention
from rtp_llm.models_py.modules.common.mha.base import FMHAImplBase
from rtp_llm.models_py.modules.common.mla import DECODE_MLA_IMPS, PREFILL_MLA_IMPS
from rtp_llm.models_py.modules.common.mla.mla_attention import MlaAttention
from rtp_llm.models_py.modules.embedding import Embedding
from rtp_llm.models_py.modules.kvcache_store import WriteCacheStoreOp
from rtp_llm.models_py.modules.mlp import DenseMLP
from rtp_llm.models_py.modules.norm import AddBiasResLayerNorm, LayerNorm, RMSNormTorch

__all__ = [
    "Linear",
    "FusedSiluActDenseMLP",
    "FusedQKRMSNorm",
    "RMSNorm",
    "SelectTopk",
    "GroupTopK",
    "DECODE_MHA_IMPS",
    "PREFILL_MHA_IMPS",
    "DECODE_MLA_IMPS",
    "PREFILL_MLA_IMPS",
    "FMHAImplBase",
    "WriteCacheStoreOp",
    "LinearTorch",
    "DenseMLP",
    "AddBiasResLayerNorm",
    "LayerNorm",
    "RMSNormTorch",
    "Embedding",
    "MlaAttention",
    "CausalAttention",
]
