"""Load the packaged FP8 MQA adapter or use a legacy DeepGEMM installation."""

import importlib.util
from functools import lru_cache
from pathlib import Path

_ADAPTER = Path(__file__).parents[3] / "libs/fp8_mqa_adapter.py"


def _legacy_backend():
    try:
        import deep_gemm
    except ImportError:
        return None
    if hasattr(deep_gemm, "get_legacy_paged_mqa_logits_metadata"):
        # This release needs its FP32-scale adapter to select the schedule.
        return None
    if all(
        callable(getattr(deep_gemm, name, None))
        for name in (
            "fp8_mqa_logits",
            "fp8_paged_mqa_logits",
            "get_paged_mqa_logits_metadata",
        )
    ):
        return deep_gemm
    return None


def available() -> bool:
    return _ADAPTER.is_file() or _legacy_backend() is not None


@lru_cache(maxsize=1)
def _backend():
    if _ADAPTER.is_file():
        spec = importlib.util.spec_from_file_location("fp8_mqa_adapter", _ADAPTER)
        adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(adapter)
        return adapter
    backend = _legacy_backend()
    if backend is None:
        raise RuntimeError("No DeepGEMM FP32-scale FP8 MQA backend is available")
    return backend


def get_num_sms():
    return int(_backend().get_num_sms())


def fp8_mqa_logits(*args, **kwargs):
    return _backend().fp8_mqa_logits(*args, **kwargs)


def get_paged_mqa_logits_metadata(*args, **kwargs):
    return _backend().get_paged_mqa_logits_metadata(*args, **kwargs)


def fp8_paged_mqa_logits(*args, **kwargs):
    return _backend().fp8_paged_mqa_logits(*args, **kwargs)
