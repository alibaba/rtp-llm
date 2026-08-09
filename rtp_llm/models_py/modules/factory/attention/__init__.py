"""Attention factory module - handles different attention implementations."""

from rtp_llm.device.device_type import DeviceType, get_device_type

# Import the factory after lists are defined to avoid circular imports
from rtp_llm.models_py.modules.factory.attention.attn_factory import AttnImplFactory
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import (
    FMHAImplBase,
    MlaImplBase,
)

__all__ = [
    "FMHAImplBase",
    "MlaImplBase",
    "AttnImplFactory",
]

# ============================================================================
# Device-specific Attention implementation registration
# ============================================================================
from rtp_llm.models_py.modules.factory.attention import attn_factory
from rtp_llm.models_py.modules.factory.attention.attn_factory import (
    DECODE_MHA_IMPS,
    DECODE_MLA_IMPS,
    PREFILL_MHA_IMPS,
    PREFILL_MLA_IMPS,
)

from rtp_llm.utils.backend_registry import run_backend_registrations

device_type = get_device_type()
if device_type == DeviceType.ROCm:
    # Import to register ROCm FMHA implementations
    from rtp_llm.models_py.modules.factory.attention.rocm_impl.aiter import (
        AiterDecodeImplAsm,
        AiterDecodeImplNonAsm,
        AiterDecodeImplTriton,
        AiterPrefillImplAsm,
        AiterPrefillImplNonAsm,
        AiterPrefillImplPaged,
        validate_v_layout,
    )

    attn_factory.VALIDATE_FMHA_CONFIG = validate_v_layout

    PREFILL_MHA_IMPS.append(AiterPrefillImplPaged)
    PREFILL_MHA_IMPS.append(AiterPrefillImplAsm)
    PREFILL_MHA_IMPS.append(AiterPrefillImplNonAsm)
    DECODE_MHA_IMPS.append(AiterDecodeImplTriton)
    DECODE_MHA_IMPS.append(AiterDecodeImplAsm)
    DECODE_MHA_IMPS.append(AiterDecodeImplNonAsm)
elif device_type == DeviceType.Cuda:
    # currently append early means impl has higher priority
    from rtp_llm.models_py.modules.factory.attention.cuda_headwise_impl.headwise import (
        HeadWisePrefillImpl,
    )
    from rtp_llm.models_py.modules.factory.attention.cuda_headwise_impl.headwise_fp8 import (
        HeadWiseFP8PrefillImpl,
    )
    from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
        PyFlashinferDecodeImpl,
        PyFlashinferHybridPrefillImpl,
        PyFlashinferPagedPrefillImpl,
        PyFlashinferPrefillImpl,
    )
    from rtp_llm.models_py.modules.factory.attention.cuda_impl.trt import (
        FlashInferTRTLLMFMHAv2PagedPrefillImpl,
        FlashInferTRTLLMFMHAv2PrefillImpl,
    )
    from rtp_llm.models_py.modules.factory.attention.cuda_impl.trtllm_gen import (
        FlashInferTRTLLMDecodeImpl,
        FlashInferTRTLLMPrefillImpl,
        FlashInferTRTLLMSpecDecodeImpl,
    )
    from rtp_llm.models_py.modules.factory.attention.cuda_impl.xqa import (
        XQAImpl,
        get_xqa_impl,
    )

    PREFILL_MHA_IMPS.extend(
        [
            HeadWiseFP8PrefillImpl,
            HeadWisePrefillImpl,
            FlashInferTRTLLMSpecDecodeImpl,
            FlashInferTRTLLMPrefillImpl,
            FlashInferTRTLLMFMHAv2PrefillImpl,
            PyFlashinferPrefillImpl,
            PyFlashinferHybridPrefillImpl,
            PyFlashinferPagedPrefillImpl,
            FlashInferTRTLLMFMHAv2PagedPrefillImpl,
        ]
    )
    DECODE_MHA_IMPS.extend([FlashInferTRTLLMDecodeImpl])
    # XQAImpl (TRT GMMA) before XQADecodeImpl (FlashInfer HMMA): different
    # accumulation paths produce <1 ULP divergence that flips tokens in long
    # generations.  Existing golden data was generated with XQAImpl, so keep
    # it higher-priority to avoid unnecessary golden refreshes.
    DECODE_MHA_IMPS.append(XQAImpl)
    _xqa_decode_impl = get_xqa_impl()
    if _xqa_decode_impl is not XQAImpl:
        DECODE_MHA_IMPS.append(_xqa_decode_impl)
    DECODE_MHA_IMPS.append(PyFlashinferDecodeImpl)

    from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashinfer_mla_wrapper import (
        MlaFlashInferDecodeImpl,
        MlaFlashInferPrefillImpl,
    )

    DECODE_MLA_IMPS.append(MlaFlashInferDecodeImpl)
    PREFILL_MLA_IMPS.append(MlaFlashInferPrefillImpl)

    # SparseMlaImpl requires CUDA >= 12.9 for flash_mla support
    try:
        import torch

        if torch.version.cuda:
            major, minor = map(int, torch.version.cuda.split(".")[:2])
            if (major, minor) >= (12, 9):
                from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_cp_impl import (
                    SparseMlaCpImpl,
                )
                from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_impl import (
                    SparseMlaImpl,
                )

                DECODE_MLA_IMPS.append(SparseMlaImpl)
                PREFILL_MLA_IMPS.append(SparseMlaImpl)
                PREFILL_MLA_IMPS.append(SparseMlaCpImpl)
    except (ImportError, AttributeError, ValueError):
        pass  # Skip SparseMlaImpl if CUDA < 12.9 or flash_mla not available

    from rtp_llm.models_py.modules.factory.attention.cuda_cp_impl.prefill_cp_flashinfer import (
        CPFlashInferImpl,
    )

    PREFILL_MHA_IMPS.append(CPFlashInferImpl)
elif device_type == DeviceType.Ppu:
    from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
        PyFlashinferDecodeImpl,
        PyFlashinferHybridPrefillImpl,
        PyFlashinferPagedPrefillImpl,
        PyFlashinferPrefillImpl,
    )

    PREFILL_MHA_IMPS.append(PyFlashinferPrefillImpl)
    PREFILL_MHA_IMPS.append(PyFlashinferHybridPrefillImpl)
    PREFILL_MHA_IMPS.append(PyFlashinferPagedPrefillImpl)
    DECODE_MHA_IMPS.append(PyFlashinferDecodeImpl)

    from rtp_llm.models_py.modules.factory.attention.cuda_cp_impl.prefill_cp_flashinfer import (
        CPFlashInferImpl,
    )

    PREFILL_MHA_IMPS.append(CPFlashInferImpl)

# Out-of-tree backends registered a hook before this module existed. Ordering in
# these lists is priority (earlier wins), so a backend inserts rather than
# appends when it needs to outrank the device impls selected above.
run_backend_registrations(
    "attention",
    prefill_mha_imps=PREFILL_MHA_IMPS,
    decode_mha_imps=DECODE_MHA_IMPS,
    prefill_mla_imps=PREFILL_MLA_IMPS,
    decode_mla_imps=DECODE_MLA_IMPS,
)
