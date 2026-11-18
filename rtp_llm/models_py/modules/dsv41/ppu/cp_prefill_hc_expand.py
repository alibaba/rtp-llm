"""Qualify the existing HC bit-copy kernel for the original CP embedding.

The public coordinator propagates CP before embedding. That propagation
binds this forward's original-row birth, so the broadcast consumes an
existing proof rather than certifying a shape at a later cache hit. Every
row is copied, including framework padding and all BF16 IEEE bit patterns.
"""

import torch

from rtp_llm.models_py.modules.dsv41.cp import CPContext
from rtp_llm.models_py.modules.dsv41.ppu import _quant
from rtp_llm.models_py.modules.dsv41.ppu.prefill_runtime import SharedForwardState
from rtp_llm.models_py.modules.dsv41.ppu.row_metadata import (
    is_plain_tensor,
    tensor_metadata_key,
)


def _device_supported(hidden):
    return (
        hidden.is_cuda
        and torch.version.hip is None
        and _quant.is_ppu()
        and torch.cuda.get_device_capability(hidden.device) == (8, 9)
        and not torch.cuda.is_current_stream_capturing()
    )


def enabled(v4, hidden, input_ids):
    """Metadata-only original CP8/YOCO140 eligibility; unknown owners decline."""
    runtime = getattr(v4, "_ppu_runtime", None)
    state = getattr(runtime, "current", None)
    if (
        type(state) is not SharedForwardState
        or state.runtime is not runtime
        or getattr(getattr(runtime, "model", None), "v4", None) is not v4
        or state.v4 is not v4
        or not state.prefill
        or state.idle
        or state.compacted
        or torch.is_grad_enabled()
        or getattr(v4, "hc_mult", None) != 4
    ):
        return False
    native_ids = state.inputs.input_ids
    # PyModelInputs getters may wrap one TensorImpl more than once. The
    # existing CP birth uses this same exact metadata key: Python identity
    # is unnecessary, while clones and derived TensorImpls remain untrusted.
    if (
        not is_plain_tensor(native_ids)
        or not is_plain_tensor(input_ids)
        or tensor_metadata_key(native_ids) != tensor_metadata_key(input_ids)
    ):
        return False
    cp = state.context
    if (
        not isinstance(cp, CPContext)
        or cp is not state.original_context
        or cp.cp_size != 8
        or not cp.kv_cache_sharded
        or cp.gather_restore_positions is not None
        or getattr(v4, "_cp_size", None) != 8
        or getattr(v4, "_cp_rank", None) != cp.cp_rank
        or getattr(v4, "_cp_info", None) is not cp.cp_info
        or not state._cp_rows_phase()
        or not state._cp_full_rows_valid()
    ):
        return False
    return (
        isinstance(hidden, torch.Tensor)
        and hidden.layout == torch.strided
        and hidden.dtype == torch.bfloat16
        and hidden.shape == (cp.chunk_length, 5120)
        and hidden.device == input_ids.device
        and hidden.is_contiguous()
        and not hidden.requires_grad
        and not hidden.is_neg()
        and not hidden.is_conj()
        and _device_supported(hidden)
    )
