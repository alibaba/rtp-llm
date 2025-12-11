"""Attention factory module - handles different attention implementations."""

import logging

# Import the factory after lists are defined to avoid circular imports
from rtp_llm.models_py.modules.factory.attention.attn_factory import AttnImplFactory
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import (
    FMHADecodeImplBase,
    FMHAImplBase,
    FMHAPrefillImplBase,
)
from rtp_llm.ops.compute_ops import DeviceType, get_device
import torch

__all__ = [
    "FMHAImplBase",
    "FMHAPrefillImplBase",
    "FMHADecodeImplBase",
    "AttnImplFactory",
]

# ============================================================================
# Device-specific Attention implementation registration
# ============================================================================
from rtp_llm.models_py.modules.factory.attention.attn_factory import (
    DECODE_MHA_IMPS,
    DECODE_MLA_IMPS,
    PREFILL_MHA_IMPS,
    PREFILL_MLA_IMPS,
)

device_type = get_device().get_device_type()
if device_type == DeviceType.ROCm:
    # Import to register ROCm FMHA implementations
    from rtp_llm.models_py.modules.factory.attention.rocm_impl.aiter import (
        AiterDecodeImpl,
        AiterPrefillImpl,
    )

    PREFILL_MHA_IMPS.append(AiterPrefillImpl)
    DECODE_MHA_IMPS.append(AiterDecodeImpl)
else:
    from rtp_llm.models_py.modules.factory.attention.cuda_impl.flash_infer import (
        FlashInferDecodeImpl,
        FlashInferPrefillImpl,
    )

    # currently append early means impl has higher priority
    if device_type == DeviceType.Cuda:
        from rtp_llm.models_py.modules.factory.attention.cuda_impl.trt import (
            TRTMHAImpl,
            TRTPagedMHAImpl,
        )
        from rtp_llm.models_py.modules.factory.attention.cuda_impl.xqa import XQAImpl

        PREFILL_MHA_IMPS.extend([TRTMHAImpl, TRTPagedMHAImpl])

        major, minor = map(int, torch.version.cuda.split('.')[:2])
        use_xqa_decode = False
        
        if (major, minor) >= (12, 8):
            try:
                from flashinfer.xqa import xqa
                use_xqa_decode = True
                logging.info("CUDA >= 12.8 and flashinfer.xqa available, using XQADecodeImpl")
            except (ImportError, AttributeError) as e:
                logging.info(f"CUDA >= 12.8 but flashinfer.xqa not available ({e}), using XQAImpl")
        else:
            logging.info(f"CUDA version {major}.{minor} < 12.8, using XQAImpl")
        
        if use_xqa_decode:
            from rtp_llm.models_py.modules.factory.attention.cuda_impl.xqa import XQADecodeImpl
            DECODE_MHA_IMPS.append(XQADecodeImpl)
        else:
            DECODE_MHA_IMPS.append(XQAImpl)
        
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashinfer_mla_wrapper import (
            MlaFlashInferDecodeImpl,
            MlaFlashInferPrefillImpl,
        )

        DECODE_MLA_IMPS.append(MlaFlashInferDecodeImpl)
        PREFILL_MLA_IMPS.append(MlaFlashInferPrefillImpl)

    PREFILL_MHA_IMPS.append(FlashInferPrefillImpl)
    DECODE_MHA_IMPS.append(FlashInferDecodeImpl)
