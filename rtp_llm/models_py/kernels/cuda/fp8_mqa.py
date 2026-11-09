"""FP8 indexer kernels with a private native/JIT runtime.

The CUDA 13 wheel supplies this extension independently of global DeepGEMM.
Other builds can keep using legacy FP8 APIs when their DeepGEMM provides them.
No modules or environment variables belonging to global DeepGEMM are replaced.
"""

import importlib.util
import os
from functools import lru_cache
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[3] / "libs/fp8_mqa_isolated"


def _binary():
    paths = sorted(_ROOT.glob("_rtp_llm_fp8_mqa*.so"))
    if len(paths) > 1:
        raise RuntimeError("Expected exactly one isolated FP8 MQA library")
    return paths[0] if paths else None


def available() -> bool:
    if _binary() is not None:
        return True
    try:
        import deep_gemm

        return all(
            callable(getattr(deep_gemm, name, None))
            for name in (
                "fp8_mqa_logits",
                "fp8_paged_mqa_logits",
                "get_paged_mqa_logits_metadata",
            )
        )
    except ImportError:
        return False


@lru_cache(maxsize=1)
def _backend():
    binary = _binary()
    if binary is None:
        import deep_gemm

        if not available():
            raise RuntimeError(
                "The RTP wheel is missing the isolated FP8 MQA extension"
            )
        return deep_gemm
    if torch.cuda.is_initialized() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Warm up isolated FP8 MQA before CUDA Graph capture")
    spec = importlib.util.spec_from_file_location("_rtp_llm_fp8_mqa", binary)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    from torch.utils.cpp_extension import CUDA_HOME

    cuda = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or CUDA_HOME
    if not cuda:
        raise RuntimeError("CUDA toolkit is required by the FP8 MQA JIT compiler")
    native.init(str(_ROOT), str(cuda))
    return native


def _configured_backend():
    native = _backend()
    # Retain the process's SM/PDL budget without sharing its C++ runtime.
    import deep_gemm

    if native is not deep_gemm:
        native.set_num_sms(int(deep_gemm.get_num_sms()))
        native.set_pdl(bool(deep_gemm.get_pdl()))
    return native


def get_num_sms():
    return int(_configured_backend().get_num_sms())


def fp8_mqa_logits(*args, **kwargs):
    return _configured_backend().fp8_mqa_logits(*args, **kwargs)


def get_paged_mqa_logits_metadata(*args, **kwargs):
    return _configured_backend().get_paged_mqa_logits_metadata(*args, **kwargs)


def fp8_paged_mqa_logits(*args, **kwargs):
    return _configured_backend().fp8_paged_mqa_logits(*args, **kwargs)
