from typing import Union

import torch

# Type alias for quantization dtype
QuantDtype = Union[None, torch.dtype, str]


from rtp_llm.ops import DeviceExporter, DeviceType, get_device

device_type = get_device().get_device_type()

if device_type == DeviceType.ROCm:
    from rtp_llm.models_py.modules.rocm.linear import Linear
    from rtp_llm.models_py.modules.rocm.mlp import DenseMLP, FusedSiluActDenseMLP
    from rtp_llm.models_py.modules.rocm.norm import FusedQKRMSNorm, RMSNorm
    from rtp_llm.models_py.modules.rocm.fmha import (
        DECODE_MHA_IMPS,
        PREFILL_MHA_IMPS,
    )
else:
    from rtp_llm.models_py.modules.norm import FusedQKRMSNorm, RMSNorm
    from rtp_llm.models_py.modules.linear import Linear
    from rtp_llm.models_py.modules.mlp import DenseMLP, FusedSiluActDenseMLP
    from rtp_llm.models_py.modules.fmha import (
        DECODE_MHA_IMPS,
        PREFILL_MHA_IMPS,
    )

from rtp_llm.models_py.modules.kvcache_store import WriteCacheStoreOp
from rtp_llm.models_py.modules.embedding import Embedding
from rtp_llm.models_py.modules.fmha import FMHAImplBase
from rtp_llm.models_py.modules.linear import LinearTorch
from rtp_llm.models_py.modules.mlp import DenseMLP
from rtp_llm.models_py.modules.norm import AddBiasResLayerNorm, LayerNorm, RMSNormTorch
