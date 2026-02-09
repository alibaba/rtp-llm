"""Attention factory module - handles different attention implementations."""

# Import the factory after lists are defined to avoid circular imports
from rtp_llm.models_py.modules.factory.attention.attn_factory import AttnImplFactory
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase
from rtp_llm.ops.compute_ops import DeviceType, get_device

__all__ = [
    "FMHAImplBase",
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
        AiterDecodeImplAsm,
        AiterDecodeImplNonAsm,
        AiterPrefillImplAsm,
        AiterPrefillImplNonAsm,
    )

    PREFILL_MHA_IMPS.append(AiterPrefillImplAsm)
    PREFILL_MHA_IMPS.append(AiterPrefillImplNonAsm)
    DECODE_MHA_IMPS.append(AiterDecodeImplAsm)
    DECODE_MHA_IMPS.append(AiterDecodeImplNonAsm)
else:
    # currently append early means impl has higher priority
    if device_type == DeviceType.Cuda:
        from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
            PyFlashinferDecodeImpl,
            PyFlashinferPagedPrefillImpl,
            PyFlashinferPrefillImpl,
        )
        from rtp_llm.models_py.modules.factory.attention.cuda_impl.trt import (
            TRTMHAImpl,
            TRTPagedMHAImpl,
        )
        from rtp_llm.models_py.modules.factory.attention.cuda_impl.trtllm_gen import (
            FlashInferTRTLLMDecodeImpl,
            FlashInferTRTLLMPrefillImpl,
            FlashInferTRTLLMSpecDecodeImpl,
        )
        from rtp_llm.models_py.modules.factory.attention.cuda_impl.xqa import (
            get_xqa_impl,
        )

        PREFILL_MHA_IMPS.extend(
            [
                TRTMHAImpl,
                PyFlashinferPrefillImpl,
                PyFlashinferPagedPrefillImpl,
                FlashInferTRTLLMSpecDecodeImpl,
                FlashInferTRTLLMPrefillImpl,
                TRTPagedMHAImpl,
            ]
        )
        DECODE_MHA_IMPS.extend([FlashInferTRTLLMDecodeImpl])
        DECODE_MHA_IMPS.append(get_xqa_impl())

        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashinfer_mla_wrapper import (
            MlaFlashInferDecodeImpl,
            MlaFlashInferPrefillImpl,
        )

        DECODE_MLA_IMPS.append(MlaFlashInferDecodeImpl)
        PREFILL_MLA_IMPS.append(MlaFlashInferPrefillImpl)
        from rtp_llm.models_py.modules.factory.attention.cuda_impl.flash_infer import (
            FlashInferDecodeImpl,
            FlashInferPrefillImpl,
        )

        PREFILL_MHA_IMPS.append(FlashInferPrefillImpl)
        DECODE_MHA_IMPS.append(FlashInferDecodeImpl)
    from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
        PyFlashinferDecodeImpl,
        PyFlashinferPrefillImpl,
    )

    PREFILL_MHA_IMPS.append(PyFlashinferPrefillImpl)
    DECODE_MHA_IMPS.append(PyFlashinferDecodeImpl)
