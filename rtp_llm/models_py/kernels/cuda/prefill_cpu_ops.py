"""Opt-in CPU batching of fresh inference output allocations."""

import logging
from functools import lru_cache
from pathlib import Path

from rtp_llm.models_py import prefill_optimization as _prefill_optimization
from torch.nn.modules import module as _module

NATIVE_PLAN = _prefill_optimization.ENABLED
NATIVE_ALLOC = _prefill_optimization.ENABLED
DIRECT_ROUTER = _prefill_optimization.ENABLED
NATIVE_LAUNCH = _prefill_optimization.ENABLED
NATIVE_ROUTER = _prefill_optimization.ENABLED


def no_module_hooks(module):
    return not (
        module._forward_hooks
        or module._forward_pre_hooks
        or module._backward_hooks
        or module._backward_pre_hooks
        or _module._global_forward_hooks
        or _module._global_forward_pre_hooks
        or _module._global_backward_hooks
        or _module._global_backward_pre_hooks
        or module._compiled_call_impl is not None
    )


@lru_cache(maxsize=1)
def native():
    # Compile during startup/warmup only. Existing path remains available when
    # the toolchain is absent. The extension contains CPU code, no CUDA compiler.
    try:
        from torch.utils.cpp_extension import load

        return load(
            name="rtp_prefill_cpu_ops",
            sources=[str(Path(__file__).with_suffix(".cc"))],
            extra_cflags=["-O3", "-g0"],
            extra_ldflags=["-ldl"],
            with_cuda=False,
            verbose=False,
        )
    except Exception:
        logging.exception("Prefill CPU allocator unavailable; using Torch allocations")
        return None


# When enabled, pay toolchain/import cost during worker startup rather than
# inserting it into the first prefill request.
if NATIVE_ALLOC or NATIVE_LAUNCH or NATIVE_ROUTER or NATIVE_PLAN:
    native()
