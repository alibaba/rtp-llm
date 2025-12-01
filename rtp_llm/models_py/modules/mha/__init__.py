# Import base classes and lists first
from rtp_llm.models_py.modules.mha.base import (
    DECODE_MHA_IMPS,
    FMHADecodeImplBase,
    FMHAImplBase,
    FMHAPrefillImplBase,
    PREFILL_MHA_IMPS,
)

# Import implementations to register them
# These imports will trigger the registration of implementations
try:
    import rtp_llm.models_py.modules.mha.flashinfer_trtllm_gen  # noqa: F401
except ImportError:
    pass

try:
    import rtp_llm.models_py.modules.mha.flashinfer_prefill  # noqa: F401
except ImportError:
    pass

try:
    import rtp_llm.models_py.modules.mha.flashinfer_decode  # noqa: F401
except ImportError:
    pass

try:
    import rtp_llm.models_py.modules.mha.trt_mha  # noqa: F401
except ImportError:
    pass

try:
    import rtp_llm.models_py.modules.mha.xqa  # noqa: F401
except ImportError:
    pass

__all__ = [
    "PREFILL_MHA_IMPS",
    "FMHAPrefillImplBase",
    "FMHADecodeImplBase",
    "FMHAImplBase",
    "DECODE_MHA_IMPS",
]
