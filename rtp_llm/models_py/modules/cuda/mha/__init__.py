from rtp_llm.models_py.modules.common.mha import DECODE_MHA_IMPS, PREFILL_MHA_IMPS
from rtp_llm.models_py.modules.utils import is_cuda

from .flash_infer import FlashInferDecodeImpl, FlashInferPrefillImpl

# currently append early means impl has higher priority
if is_cuda():
    from .trt import TRTMHAImpl

    PREFILL_MHA_IMPS.append(TRTMHAImpl)
    from .xqa import XQAImpl

    DECODE_MHA_IMPS.append(XQAImpl)


PREFILL_MHA_IMPS.append(FlashInferPrefillImpl)
DECODE_MHA_IMPS.append(FlashInferDecodeImpl)
