import os

SM120_FP8_BACKEND_ENV = "RTP_LLM_SM120_FP8_BACKEND"
SM120_FP8_BACKENDS = frozenset({"auto", "cutlass", "deepgemm"})


def get_sm120_fp8_backend() -> str:
    """Return the requested SM120 FP8 backend."""
    backend = os.environ.get(SM120_FP8_BACKEND_ENV, "auto").strip().lower()
    if backend not in SM120_FP8_BACKENDS:
        choices = ", ".join(sorted(SM120_FP8_BACKENDS))
        raise ValueError(
            f"invalid {SM120_FP8_BACKEND_ENV}={backend!r}; expected one of {choices}"
        )
    return backend


def resolve_sm120_fp8_backend() -> str:
    """Resolve auto while preserving V3's existing DeepGEMM default."""
    backend = get_sm120_fp8_backend()
    return "deepgemm" if backend == "auto" else backend
