# isort: skip_file

from .base import (
    MoeCfg,
    RoutedExpertsStrategy,
    select_strategy,
)

# Side-effect import — each module's ``@register_strategy`` decorator pushes
# the class into the priority list as the import lands. Order here = priority
# (high→low):
from .mega import MegaMoEStrategy  # noqa: F401  ep_size>1 + SM100 + dist
from .mega_se import MegaMoEStrategySE  # noqa: F401  explicit fused-SE opt-in
from .mega_fused import MegaMoEFusedStrategy  # noqa: F401  ep_size>1 + fused opt-in
from .grouped_fp4 import (  # noqa: F401  ep_size==1 + kernel
    GroupedFP4Strategy,
    _has_fp8_fp4_grouped_kernel,
)
from .deepep import DeepEPStrategy  # noqa: F401  actual DeepEP only
from .sm120_fused_moe import (  # noqa: F401  SM120 FusedMoe + collectives
    Sm120FusedMoeStrategy,
)
from .nccl_ep_mxfp8 import (  # noqa: F401  CP4EP4PP2 stage-local NCCL MXFP8
    NcclEpMxfp8Strategy,
)
from .sm120_decode import Sm120DecodeStrategy  # noqa: F401 explicit DP/EP decode
from .local_loop import LocalLoopStrategy  # noqa: F401  universal fallback

__all__ = [
    "MoeCfg",
    "RoutedExpertsStrategy",
    "select_strategy",
    "_has_fp8_fp4_grouped_kernel",
]
