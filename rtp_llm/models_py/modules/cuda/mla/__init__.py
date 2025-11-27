from rtp_llm.models_py.modules.common.mla import DECODE_MLA_IMPS, PREFILL_MLA_IMPS
from rtp_llm.models_py.modules.utils import is_cuda

if is_cuda():
    from .flashinfer_mla_wrapper import (
        MlaFlashInferDecodeImpl,
        MlaFlashInferPrefillImpl,
    )

    DECODE_MLA_IMPS.append(MlaFlashInferDecodeImpl)
    PREFILL_MLA_IMPS.append(MlaFlashInferPrefillImpl)
