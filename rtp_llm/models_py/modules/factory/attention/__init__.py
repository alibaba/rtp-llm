"""Attention factory module - handles different attention implementations."""

import logging

from rtp_llm.device.device_type import DeviceType, get_device_type

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

from rtp_llm.models_py.modules.factory.attention import attn_factory
from rtp_llm.models_py.modules.factory.attention.attn_factory import (
    DECODE_MHA_IMPS,
    DECODE_MLA_IMPS,
    PREFILL_MHA_IMPS,
    PREFILL_MLA_IMPS,
)
from rtp_llm.models_py.modules.factory.platform_ext_loader import load_platform_extension

from rtp_llm.utils.backend_registry import run_backend_registrations

device_type = get_device_type()
if device_type == DeviceType.ROCm:
    from rtp_llm.models_py.modules.factory.attention.rocm_impl.aiter import (
        AiterDecodeImplAsm,
        AiterDecodeImplNonAsm,
        AiterDecodeImplTriton,
        AiterPrefillImplAsm,
        AiterPrefillImplNonAsm,
        AiterPrefillImplPaged,
        AiterPrefillImplTriton,
    )

    PREFILL_MHA_IMPS.extend(
        [AiterPrefillImplTriton, AiterPrefillImplPaged, AiterPrefillImplAsm, AiterPrefillImplNonAsm]
    )
    DECODE_MHA_IMPS.extend(
        [AiterDecodeImplTriton, AiterDecodeImplAsm, AiterDecodeImplNonAsm]
    )
elif device_type == DeviceType.Cuda:
    # These modules import CUDA-only packages at module scope. Keeping every
    # import inside this branch makes non-CUDA test collection import-safe.
    from rtp_llm.models_py.modules.factory.attention.cuda_headwise_impl.headwise import (
        HeadWisePrefillImpl,
    )
    from rtp_llm.models_py.modules.factory.attention.cuda_headwise_impl.headwise_fp8 import (
        HeadWiseFP8PrefillImpl,
    )
    from rtp_llm.models_py.modules.factory.attention.cuda_impl.flash_attn_3 import (
        FlashAttn3PagedShortGraphImpl,
    )
    from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
        PyFlashinferHybridPrefillImpl,
        PyFlashinferPagedPrefillImpl,
        PyFlashinferPrefillImpl,
    )

    try:
        from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
            PyFlashinferDecodeImpl,
        )
    except ImportError as e:
        PyFlashinferDecodeImpl = None
        logging.warning("Skip Python FlashInfer decode implementation: %s", e)

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
            FlashAttn3PagedShortGraphImpl,
            FlashInferTRTLLMSpecDecodeImpl,
            FlashInferTRTLLMPrefillImpl,
            FlashInferTRTLLMFMHAv2PrefillImpl,
            PyFlashinferPrefillImpl,
            PyFlashinferHybridPrefillImpl,
            PyFlashinferPagedPrefillImpl,
            FlashInferTRTLLMFMHAv2PagedPrefillImpl,
        ]
    )
    DECODE_MHA_IMPS.append(FlashInferTRTLLMDecodeImpl)
    # Preserve the established accumulation path before trying FlashInfer's
    # XQA fallback; the two implementations can differ by less than one ULP.
    DECODE_MHA_IMPS.append(XQAImpl)
    _xqa_decode_impl = get_xqa_impl()
    if _xqa_decode_impl is not XQAImpl:
        DECODE_MHA_IMPS.append(_xqa_decode_impl)
    if PyFlashinferDecodeImpl is not None:
        DECODE_MHA_IMPS.append(PyFlashinferDecodeImpl)

    from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashinfer_mla_wrapper import (
        MlaFlashInferDecodeImpl,
        MlaFlashInferPrefillImpl,
    )

    DECODE_MLA_IMPS.append(MlaFlashInferDecodeImpl)
    PREFILL_MLA_IMPS.append(MlaFlashInferPrefillImpl)

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
                PREFILL_MLA_IMPS.extend([SparseMlaImpl, SparseMlaCpImpl])
    except (ImportError, AttributeError, ValueError):
        pass

    from rtp_llm.models_py.modules.factory.attention.cuda_cp_impl.prefill_cp_flashinfer import (
        CPFlashInferImpl,
    )

    PREFILL_MHA_IMPS.append(CPFlashInferImpl)

extension = load_platform_extension()
if extension and hasattr(extension, "register_attention"):
    extension.register_attention(
        device_type=device_type,
        prefill_mha_imps=PREFILL_MHA_IMPS,
        decode_mha_imps=DECODE_MHA_IMPS,
        prefill_mla_imps=PREFILL_MLA_IMPS,
        decode_mla_imps=DECODE_MLA_IMPS,
    )


def _validate_impl_names() -> None:
    """Assert every registered implementation has a public backend name."""
    for registry_name, registry in (
        ("PREFILL_MHA_IMPS", PREFILL_MHA_IMPS),
        ("DECODE_MHA_IMPS", DECODE_MHA_IMPS),
        ("PREFILL_MLA_IMPS", PREFILL_MLA_IMPS),
        ("DECODE_MLA_IMPS", DECODE_MLA_IMPS),
    ):
        for cls in registry:
            if not getattr(cls, "NAME", ""):
                raise RuntimeError(
                    f"Impl class {cls.__module__}.{cls.__name__} registered "
                    f"in {registry_name} has empty NAME - set a backend NAME "
                    f"matching the help text in fmha_group_args.py, or remove "
                    f"the class from the registry."
                )

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
_validate_impl_names()
