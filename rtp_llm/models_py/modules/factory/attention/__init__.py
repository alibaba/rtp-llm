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
if device_type == DeviceType.ROCm:
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
    DECODE_MHA_IMPS.append(AiterDecodeImplAsm)
    DECODE_MHA_IMPS.append(AiterDecodeImplNonAsm)
    DECODE_MHA_IMPS.append(AiterDecodeImplTriton)
else:
    # currently append early means impl has higher priority
    if device_type == DeviceType.Cuda:
        prefill_mha_impls = []
        try:
            from rtp_llm.models_py.modules.factory.attention.cuda_headwise_impl.headwise import (
                HeadWisePrefillImpl,
            )

            prefill_mha_impls.append(HeadWisePrefillImpl)
        except (ImportError, AttributeError) as e:
            logging.warning("Skip HeadWise prefill implementation: %s", e)

        try:
            from rtp_llm.models_py.modules.factory.attention.cuda_headwise_impl.headwise_fp8 import (
                HeadWiseFP8PrefillImpl,
            )

            prefill_mha_impls.insert(0, HeadWiseFP8PrefillImpl)
        except (ImportError, AttributeError) as e:
            logging.warning("Skip HeadWise FP8 prefill implementation: %s", e)

        try:
            from rtp_llm.models_py.modules.factory.attention.cuda_impl.trt import (
                TRTMHAImpl,
                TRTPagedMHAImpl,
            )

            prefill_mha_impls.extend([TRTMHAImpl, TRTPagedMHAImpl])
        except (ImportError, AttributeError) as e:
            logging.warning("Skip TRT attention implementations: %s", e)

        py_flashinfer_impls = None
        try:
            from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
                PyFlashinferDecodeImpl,
                PyFlashinferPagedPrefillImpl,
                PyFlashinferPrefillImpl,
            )

            py_flashinfer_impls = (
                PyFlashinferPrefillImpl,
                PyFlashinferPagedPrefillImpl,
                PyFlashinferDecodeImpl,
            )
        except (ImportError, AttributeError) as e:
            logging.warning("Skip Python FlashInfer MHA implementations: %s", e)

        trtllm_impls = None
        try:
            from rtp_llm.models_py.modules.factory.attention.cuda_impl.trtllm_gen import (
                FlashInferTRTLLMDecodeImpl,
                FlashInferTRTLLMPrefillImpl,
                FlashInferTRTLLMSpecDecodeImpl,
            )

            trtllm_impls = (
                FlashInferTRTLLMSpecDecodeImpl,
                FlashInferTRTLLMPrefillImpl,
                FlashInferTRTLLMDecodeImpl,
            )
        except (ImportError, AttributeError) as e:
            logging.warning("Skip FlashInfer TRTLLM implementations: %s", e)

        xqa_impl = None
        try:
            from rtp_llm.models_py.modules.factory.attention.cuda_impl.xqa import (
                get_xqa_impl,
            )

            xqa_impl = get_xqa_impl()
        except (ImportError, AttributeError) as e:
            logging.warning("Skip XQA implementation: %s", e)

        PREFILL_MHA_IMPS.extend(prefill_mha_impls)
        if trtllm_impls is not None:
            PREFILL_MHA_IMPS.extend([trtllm_impls[0], trtllm_impls[1]])
            DECODE_MHA_IMPS.append(trtllm_impls[2])
        if py_flashinfer_impls is not None:
            PREFILL_MHA_IMPS.extend([py_flashinfer_impls[0], py_flashinfer_impls[1]])
            DECODE_MHA_IMPS.append(py_flashinfer_impls[2])
        if xqa_impl is not None:
            DECODE_MHA_IMPS.append(xqa_impl)

        try:
            from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashinfer_mla_wrapper import (
                MlaFlashInferDecodeImpl,
                MlaFlashInferPrefillImpl,
            )

            DECODE_MLA_IMPS.append(MlaFlashInferDecodeImpl)
            PREFILL_MLA_IMPS.append(MlaFlashInferPrefillImpl)
        except (ImportError, AttributeError) as e:
            logging.warning("Skip FlashInfer MLA implementations: %s", e)

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

        try:
            from rtp_llm.models_py.modules.factory.attention.cuda_impl.flash_infer import (
                FlashInferDecodeImpl,
                FlashInferPrefillImpl,
            )

            PREFILL_MHA_IMPS.append(FlashInferPrefillImpl)
            DECODE_MHA_IMPS.append(FlashInferDecodeImpl)
        except (ImportError, AttributeError) as e:
            logging.warning("Skip FlashInfer C++ attention implementations: %s", e)

    try:
        from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
            PyFlashinferDecodeImpl,
            PyFlashinferPagedPrefillImpl,
            PyFlashinferPrefillImpl,
        )

        PREFILL_MHA_IMPS.append(PyFlashinferPrefillImpl)
        PREFILL_MHA_IMPS.append(PyFlashinferPagedPrefillImpl)
        DECODE_MHA_IMPS.append(PyFlashinferDecodeImpl)
    except (ImportError, AttributeError) as e:
        logging.warning("Skip Python FlashInfer MHA implementations: %s", e)

    try:
        from rtp_llm.models_py.modules.factory.attention.cuda_cp_impl.prefill_cp_flashinfer import (
            CPFlashInferImpl,
        )

        PREFILL_MHA_IMPS.append(CPFlashInferImpl)
    except (ImportError, AttributeError) as e:
        logging.warning("Skip context-parallel FlashInfer implementation: %s", e)
