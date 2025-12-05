import logging

from rtp_llm.models_py.modules.common.mha import DECODE_MHA_IMPS, PREFILL_MHA_IMPS
from rtp_llm.models_py.modules.rocm.mha.aiter import (
    AiterDecodeImplAsm,
    AiterDecodeImplNonAsm,
    AiterPrefillImpl,
)

PREFILL_MHA_IMPS.append(AiterPrefillImpl)
DECODE_MHA_IMPS.append(AiterDecodeImplAsm)
DECODE_MHA_IMPS.append(AiterDecodeImplNonAsm)
