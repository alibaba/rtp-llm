"""Attention factory module - handles different attention implementations."""

import logging

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
from rtp_llm.models_py.modules.factory.attention.attn_factory import (
    DECODE_MHA_IMPS,
    DECODE_MLA_IMPS,
    PREFILL_MHA_IMPS,
    PREFILL_MLA_IMPS,
)

device_type = get_device_type()
if device_type == DeviceType.Xpu:
    # XPU hard-requires the accelerated vllm-xpu-kernels attention path,
    # matching the CUDA and ROCm backends (no pure-PyTorch attention fallback).
    from rtp_llm.models_py.modules.factory.attention.xpu_impl.vllm_flash_attn import (
        XpuVllmFlashAttnPrefillImpl,
        XpuVllmFlashAttnDecodeImpl,
    )
    PREFILL_MHA_IMPS.append(XpuVllmFlashAttnPrefillImpl)
    DECODE_MHA_IMPS.append(XpuVllmFlashAttnDecodeImpl)

    # FA2 preflight: XPU decode hard-requires the FA2 kernel from
    # vllm-xpu-kernels, which is installed out-of-band (see deps/requirements_xpu.txt)
    # and cannot be expressed as a pip dependency. Fail fast at startup with an
    # actionable message instead of letting the service boot and only fail at the
    # first decode step.
    try:
        from rtp_llm.models_py.modules.base.xpu.vllm_xpu_ops import check_fa2_requirements
        _fa2_error = check_fa2_requirements()
    except Exception as _fa2_exc:  # noqa: BLE001 - normalize probe failure to a hard error
        _fa2_error = f"could not probe vllm-xpu-kernels: {_fa2_exc}"
        logging.getLogger(__name__).error(
            "XPU FA2 preflight could not probe vllm-xpu-kernels: %s", _fa2_exc)
    if _fa2_error:
        raise RuntimeError(
            "XPU decode attention requires the FA2 kernel from vllm-xpu-kernels "
            ">= 0.1.10. " + _fa2_error + ". Install a compatible FA2-enabled build "
            "via the enforced installer (does not silently drift like a hand-run "
            "pip install would): "
            "`VLLM_XPU_KERNELS_INDEX_URL=<url> deps/install_xpu_fa2.sh` "
            "(see deps/requirements_xpu.txt and deps/install_xpu_fa2.sh).")
elif device_type == DeviceType.ROCm:
    # Import to register ROCm FMHA implementations
    from rtp_llm.models_py.modules.factory.attention.rocm_impl.aiter import (
        AiterDecodeImplAsm,
        AiterDecodeImplNonAsm,
        AiterDecodeImplTriton,
        AiterPrefillImplAsm,
        AiterPrefillImplNonAsm,
        AiterPrefillImplPaged,
    )

    PREFILL_MHA_IMPS.append(AiterPrefillImplPaged)
    PREFILL_MHA_IMPS.append(AiterPrefillImplAsm)
    PREFILL_MHA_IMPS.append(AiterPrefillImplNonAsm)
    DECODE_MHA_IMPS.append(AiterDecodeImplTriton)
    DECODE_MHA_IMPS.append(AiterDecodeImplAsm)
    DECODE_MHA_IMPS.append(AiterDecodeImplNonAsm)
else:
    # currently append early means impl has higher priority
    if device_type == DeviceType.Cuda:
        from rtp_llm.models_py.modules.factory.attention.cuda_headwise_impl.headwise import (
            HeadWisePrefillImpl,
        )
        from rtp_llm.models_py.modules.factory.attention.cuda_headwise_impl.headwise_fp8 import (
            HeadWiseFP8PrefillImpl,
        )
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
            XQAImpl,
            get_xqa_impl,
        )

        PREFILL_MHA_IMPS.extend(
            [
                HeadWiseFP8PrefillImpl,
                HeadWisePrefillImpl,
                FlashInferTRTLLMSpecDecodeImpl,
                FlashInferTRTLLMPrefillImpl,
                TRTMHAImpl,
                PyFlashinferPrefillImpl,
                PyFlashinferPagedPrefillImpl,
                TRTPagedMHAImpl,
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

    from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
        PyFlashinferDecodeImpl,
        PyFlashinferPagedPrefillImpl,
        PyFlashinferPrefillImpl,
    )

    PREFILL_MHA_IMPS.append(PyFlashinferPrefillImpl)
    PREFILL_MHA_IMPS.append(PyFlashinferPagedPrefillImpl)
    DECODE_MHA_IMPS.append(PyFlashinferDecodeImpl)

    from rtp_llm.models_py.modules.factory.attention.cuda_cp_impl.prefill_cp_flashinfer import (
        CPFlashInferImpl,
    )

    PREFILL_MHA_IMPS.append(CPFlashInferImpl)
