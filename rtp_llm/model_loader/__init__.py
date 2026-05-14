import os as _os

# Phase-25 namespace merge: extend `rtp_llm.model_loader.__path__` with the
# sibling internal_source counterpart so internal model_loader content (currently
# `test/`) is reachable as `rtp_llm.model_loader.test`.
_internal_dir = _os.path.normpath(
    _os.path.join(
        _os.path.dirname(_os.path.abspath(__file__)),
        "..",
        "..",
        "internal_source",
        "rtp_llm",
        "model_loader",
    )
)
if _os.path.isdir(_internal_dir) and _internal_dir not in __path__:
    __path__.append(_internal_dir)
del _os, _internal_dir

from .attn_weight import AttnAtomicWeight, AttnConfig, MlaAttnAtomicWeight, MlaConfig
from .compressed_w4a8_int4_per_channel_weight import (
    LoadCompressedW4A8Int4PerGroupQuantWeight,
)
from .dynamic_fp8_quant_weight import LoadQuantDynamicPerTensorFp8Weight
from .ffn_weight import (
    FfnAtomicWeight,
    FfnConfig,
    FfnWeight,
    MoeAtomicWeight,
    MoeConfig,
)
from .group_wise_quant_weight import GroupWiseWeight
from .mixed_fp4_quant_weight import MixedFp4Weight
from .omni_quant_weight import OmniQuantWeightInfo
from .online_modelopt_fp4_quant_weight import OnlineModelOptFp4MoeWeight
from .per_block_fp8_quant_weight import PerBlockFp8Weight
from .per_channel_fp8_quant_weight import PerChannelFp8Weight
from .per_group_fp4_quant_weight import PerGroupFp4Weight
from .per_tensor_int8_quant_weight import PerTensorInt8QuantWeight
from .smooth_quant_weight import SmoothQuantWeightInfo
from .static_fp8_quant_weight import Fp8PerTensorCompressedWeight
from .w4a8_int4_per_channel_quant_weight import LoadW4a8Int4PerChannelQuantWeight
from .weight_only_quant_weight import WeightOnlyPerColWeight
