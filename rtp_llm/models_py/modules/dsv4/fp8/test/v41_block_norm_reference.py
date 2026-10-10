"""Select released weighted-normalization order on explicit SM120 block norms."""

import math
from contextlib import contextmanager
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.base.cuda.norm import RMSNorm
from rtp_llm.models_py.modules.dsv4.v41_cache import _shares_storage


@contextmanager
def block_norm_reference(enabled, *, norms=(), dim=5120):
    if type(enabled) is not bool:
        raise ValueError("Block normalization selection requires a bool")
    if not enabled:
        yield
        return
    if type(dim) is not int or dim not in (512, 5120):
        raise ValueError("Block normalization requires an admitted checkpoint width")
    norms = tuple(norms)
    selected = {id(norm) for norm in norms}
    if not norms or len(selected) != len(norms):
        raise ValueError("Select distinct block RMSNorm instances")
    for norm in norms:
        if (
            type(norm) is not RMSNorm
            or norm.weight.shape != (dim,)
            or norm.weight.dtype != torch.bfloat16
            or not norm.weight.is_cuda
            or not norm.weight.is_contiguous()
            or not math.isfinite(norm.variance_epsilon)
            or norm.variance_epsilon < 0
            or torch.cuda.get_device_capability(norm.weight.device) != (12, 0)
        ):
            raise ValueError("Block normalization requires SM120 BF16 checkpoint norms")
    original = RMSNorm.forward

    def forward(norm, hidden_states, output=None):
        if id(norm) not in selected:
            return original(norm, hidden_states, output=output)
        if (
            hidden_states.ndim != 2
            or hidden_states.shape[-1] != dim
            or hidden_states.dtype != torch.bfloat16
            or hidden_states.device != norm.weight.device
            or not hidden_states.is_contiguous()
        ):
            raise ValueError("Block normalization requires contiguous BF16 token rows")
        if output is not None and (
            output.shape != hidden_states.shape
            or output.dtype != hidden_states.dtype
            or output.device != hidden_states.device
            or not output.is_contiguous()
            or _shares_storage(output, norm.weight)
        ):
            raise ValueError(
                "Block normalization output must match rows and preserve weights"
            )
        # Keep the released FP32 reduction and multiplication boundaries. Prefill
        # may alias output with input; all reads finish before the final copy.
        value = hidden_states.float()
        variance = value.square().mean(-1, keepdim=True)
        value = value * torch.rsqrt(variance + norm.variance_epsilon)
        result = (norm.weight * value).to(hidden_states.dtype)
        if output is not None:
            output.copy_(result)
            return output
        return result

    with patch.object(RMSNorm, "forward", forward):
        yield
