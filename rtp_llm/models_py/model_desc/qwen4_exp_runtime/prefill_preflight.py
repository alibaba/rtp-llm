"""Invocation-scoped main-page checks before the first Prefill GPU producer.

The engine stages attention metadata before model forward and keeps it stable
until forward returns. Validate the live device tables together, then retain
only scoped receipts for that forward. Side-cache and selected-token checks
keep their existing mutation boundaries. No receipt survives a call.
"""

import os
from contextlib import contextmanager, nullcontext

import torch


def _signature(tensor, *, version=False):
    signature = (
        tensor.data_ptr(),
        tuple(tensor.shape),
        tensor.stride(),
        tensor.dtype,
        tensor.device,
    )
    # The native engine creates inference tensors without version counters.
    # Their metadata is read-only for the entire forward, just like the
    # graph replay metadata checked once by the model's existing callbacks.
    # Normal tensors additionally detect debug-time in-place changes.
    return signature + (
        (None if tensor.is_inference() else tensor._version,) if version else ()
    )


class MainGeometryReceipt:
    def __init__(self, impl, cache, table, bases, lengths, page_size):
        self.owners = (cache, table, bases, lengths)
        self.inputs = impl.attn_inputs
        self.padded = bool(self.inputs.is_s_padded)
        self.signatures = (
            _signature(cache),
            *(_signature(t, version=True) for t in (table, bases, lengths)),
        )
        self.page_size = page_size

    def matches(self, impl, cache, table, bases, lengths):
        if self.inputs is not impl.attn_inputs or self.padded != bool(
            impl.attn_inputs.is_s_padded
        ):
            return False
        try:
            return self.signatures == (
                _signature(cache),
                *(_signature(t, version=True) for t in (table, bases, lengths)),
            )
        except RuntimeError:
            return False


@contextmanager
def main_geometry_preflight(entries):
    """Check all entries before publishing any receipt, and clear on exceptions."""
    from rtp_llm.models_py.modules.qwen4_exp.qsa.fmha_validation import (
        _validate_main_cache_layout,
    )
    from rtp_llm.models_py.modules.qwen4_exp.qsa.paged_validation import (
        geometry_flags,
        raise_geometry_error,
    )

    pending = []
    flags = []
    for impl, cache, bases, lengths in entries:
        table = impl.attn_inputs.kv_cache_kernel_block_id_device
        if getattr(impl, "_model_prefill_geometry", None) is not None:
            raise RuntimeError("Qwen4 Prefill geometry scopes must not overlap")
        page_size, pages = _validate_main_cache_layout(
            impl, cache, table, bases.numel(), cache.device
        )
        receipt = MainGeometryReceipt(impl, cache, table, bases, lengths, page_size)
        flags.append(
            geometry_flags(
                lengths.to(device=cache.device, non_blocking=True),
                bases.to(device=cache.device, non_blocking=True),
                table,
                page_size,
                pages,
                receipt.padded,
            )
        )
        pending.append((impl, receipt))
    if flags:
        # A single acceptance boundary precedes embedding and all cache writes.
        values = torch.cat(flags).cpu().tolist()
        offset = 0
        for (_, receipt), flag in zip(pending, flags):
            verdict = 0
            for value in values[offset : offset + flag.numel()]:
                verdict |= value
            raise_geometry_error(verdict)
            offset += flag.numel()
    try:
        for impl, receipt in pending:
            receipts = getattr(impl, "_model_prefill_geometry", None)
            if receipts is None:
                impl._model_prefill_geometry = receipts = []
            receipts.append(receipt)
        yield
    finally:
        for impl, _ in pending:
            impl._model_prefill_geometry = None


def prefill_geometry_scope(model, inputs, fmha_impl):
    if (
        os.getenv("RTP_QWEN4_PREFILL_GEOMETRY_IMPL", "legacy") != "batched"
        or getattr(model.config, "is_mtp", False)
        or model.kv_cache is None
        or fmha_impl is None
        or torch.version.hip is not None
        or not inputs.input_ids.is_cuda
        or torch.cuda.get_device_capability(inputs.input_ids.device) != (12, 0)
        or any(
            module._forward_hooks or module._forward_pre_hooks
            for module in model.modules()
        )
        or torch.nn.modules.module._global_forward_hooks
        or torch.nn.modules.module._global_forward_pre_hooks
    ):
        return nullcontext()
    from rtp_llm.models_py.model_desc.qwen4_exp import Qwen4ExpModel
    from rtp_llm.models_py.modules.qwen4_exp.qsa.sparse_gqa_impl import (
        SparseGqaFmhaImpl,
    )

    if type(model) is not Qwen4ExpModel or any(
        "forward" in module.__dict__ for module in model.modules()
    ):
        return nullcontext()

    entries = []
    for index, layer in enumerate(model.layers):
        impl = model._layer_fmha_impl(layer, fmha_impl, index)
        if type(impl) is not SparseGqaFmhaImpl:
            continue
        if (
            not impl.is_prefill
            or impl.is_target_verify
            or impl.attn_inputs.is_cuda_graph
        ):
            return nullcontext()
        bases = impl.attn_inputs.prefix_lengths
        lengths = impl.attn_inputs.input_lengths
        if (
            bases.device.type != "cpu"
            or lengths.device.type != "cpu"
            or bases.ndim != 1
            or lengths.shape != bases.shape
            or bases.dtype not in (torch.int32, torch.int64)
            or lengths.dtype not in (torch.int32, torch.int64)
        ):
            return nullcontext()
        if not any(bases.tolist()):
            continue
        cache = model.kv_cache.get_layer_cache(index, model._attention_tag(index))
        entries.append((impl, cache.kv_cache_base, bases, lengths))
    return main_geometry_preflight(entries)
