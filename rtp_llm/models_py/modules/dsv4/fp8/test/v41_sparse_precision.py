"""Scoped SM120 sparse sink normalization diagnostic; native Q is retained."""

from contextlib import ExitStack, contextmanager
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4.fp8 import (
    _v41_attention_triton,
    _v41_prefill_attention_sm120,
)


class _DirectSinkKernel:
    def __init__(self, original):
        self.original = original

    def __getitem__(self, grid):
        launch = self.original[grid]

        def direct(*args, **kwargs):
            kwargs["DIRECT_SINK_DIV_RN"] = True
            return launch(*args, **kwargs)

        return direct


@contextmanager
def sparse_sink_div_rn(enabled, *, decode=True):
    if not enabled:
        yield
        return
    if torch.cuda.get_device_capability() != (12, 0):
        raise ValueError("Sparse sink diagnostic is qualified only on SM120")
    targets = [(_v41_prefill_attention_sm120, "_sparse_prefill")]
    if decode:
        targets.append((_v41_attention_triton, "_attention"))
    with ExitStack() as stack:
        for module, name in targets:
            stack.enter_context(
                patch.object(module, name, _DirectSinkKernel(getattr(module, name)))
            )
        yield
