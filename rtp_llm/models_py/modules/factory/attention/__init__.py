"""Attention factory module - handles different attention implementations."""

import logging

# Import the factory after lists are defined to avoid circular imports
from rtp_llm.models_py.modules.factory.attention.attn_factory import AttnImplFactory
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import (
    FMHAImplBase,
    MlaImplBase,
)
from rtp_llm.ops.compute_ops import DeviceType, get_exec_ctx

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

device_type = get_exec_ctx().get_device_type()
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
        # Toggle: set RTP_LLM_DISABLE_FA3_TARGET_VERIFY=1 to fall back to the
        # legacy FlashInfer BatchPrefill target-verify path.  Default order
        # gives priority to PyFA3TargetVerifyImpl, mirroring what SGLang's
        # FlashAttention backend uses on Hopper for MTP target verify and
        # side-stepping the FlashInfer plan() / CG buffer aliasing pitfalls
        # (see project_target_verify_cg_bug memory).
        import os as _os

        from rtp_llm.models_py.modules.factory.attention.cuda_headwise_impl.headwise import (
            HeadWisePrefillImpl,
        )
        from rtp_llm.models_py.modules.factory.attention.cuda_headwise_impl.headwise_fp8 import (
            HeadWiseFP8PrefillImpl,
        )
        from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_fa3_draft_prefill import (
            PyFA3DraftPrefillImpl,
        )
        from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_fa3_target_verify import (
            PyFA3TargetVerifyImpl,
        )
        from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
            PyFlashinferDecodeImpl,
            PyFlashinferPagedPrefillImpl,
            PyFlashinferPrefillImpl,
            PyFlashinferTargetVerifyPrefillImpl,
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

        _target_verify_chain = (
            [PyFlashinferTargetVerifyPrefillImpl, PyFA3TargetVerifyImpl]
            if _os.environ.get("RTP_LLM_DISABLE_FA3_TARGET_VERIFY") == "1"
            else [PyFA3TargetVerifyImpl, PyFlashinferTargetVerifyPrefillImpl]
        )
        # Toggle: set RTP_LLM_DISABLE_FA3_DRAFT_PREFILL=1 to fall back to the
        # legacy FlashInfer BatchPrefillWithPagedKVCacheWrapper path for the
        # draft-model prefill step in MTP.  Default order prefers
        # PyFA3DraftPrefillImpl, which sidesteps the FlashInfer plan() / CG
        # buffer aliasing that produces residual non-determinism in CG mode
        # (see project_draft_prefill_cg_root_cause memory).
        _draft_prefill_chain = (
            []
            if _os.environ.get("RTP_LLM_DISABLE_FA3_DRAFT_PREFILL") == "1"
            else [PyFA3DraftPrefillImpl]
        )
        PREFILL_MHA_IMPS.extend(
            [
                HeadWiseFP8PrefillImpl,
                HeadWisePrefillImpl,
                FlashInferTRTLLMSpecDecodeImpl,
                FlashInferTRTLLMPrefillImpl,
                *_target_verify_chain,
                *_draft_prefill_chain,
                TRTMHAImpl,
                PyFlashinferPrefillImpl,
                PyFlashinferPagedPrefillImpl,
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

    from rtp_llm.models_py.modules.factory.attention.cuda_cp_impl.prefill_cp_flashinfer import (
        CPFlashInferImpl,
    )

    PREFILL_MHA_IMPS.append(CPFlashInferImpl)
