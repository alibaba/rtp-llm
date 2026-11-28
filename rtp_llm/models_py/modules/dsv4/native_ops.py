"""Resolve V4's compute operators on use, not when reading checkpoint weights."""


from functools import lru_cache


@lru_cache(maxsize=1)
def has_topk_v3() -> bool:
    from rtp_llm.ops.compute_ops import rtp_llm_ops

    return hasattr(rtp_llm_ops, "topk_v3")


@lru_cache(maxsize=1)
def has_fast_prefill_topk() -> bool:
    from rtp_llm.ops.compute_ops import rtp_llm_ops

    return hasattr(rtp_llm_ops, "fast_topk_v2_variable")


def require_serving_bindings():
    """Fail before staging weights if the serving ABI is missing or incompatible."""
    from rtp_llm.ops.compute_ops import (  # noqa: F401 - validate the actual ABI
        PyModelInputs,
        PyModelOutputs,
        rtp_llm_ops,
    )


def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(name)
    from rtp_llm.ops.compute_ops import rtp_llm_ops

    # Do not mask import/ABI failures. Successful resolutions have no per-call
    # import or indirection in the inference path.
    value = getattr(rtp_llm_ops, name)
    globals()[name] = value
    return value
