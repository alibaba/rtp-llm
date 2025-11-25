from typing import List

from rtp_llm.models_py.modules.common.mha.base import (
    FMHADecodeImplBase,
    FMHAPrefillImplBase,
)

PREFILL_MLA_IMPS: List[type[FMHAPrefillImplBase]] = []
DECODE_MLA_IMPS: List[type[FMHADecodeImplBase]] = []
