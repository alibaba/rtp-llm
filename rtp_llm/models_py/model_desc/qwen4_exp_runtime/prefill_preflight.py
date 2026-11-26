"""Invocation-scoped main-page checks before the first Prefill GPU producer.

The engine stages attention metadata before model forward and keeps it stable
until forward returns. Validate the live device tables together, then retain
only scoped receipts for that forward. Side-cache and selected-token checks
keep their existing mutation boundaries. No receipt survives a call.
"""

from contextlib import contextmanager, nullcontext
from typing import NamedTuple

import torch

from rtp_llm.config.qwen4_exp_config import Qwen4ModelConfig
from rtp_llm.models_py.model_desc.block_map import get_primary_attention_inputs
from rtp_llm.utils.hot_hook_runtime import runtime as hot_hook_runtime


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


def _host_values(tensor, values):
    key = _signature(tensor, version=True)
    if key not in values:
        host = tensor.tolist()
        values[key] = (host, any(host))
    return values[key]


class PrefillHostGeometry(NamedTuple):
    lengths: list[int]
    prefixes: list[int]
    token_count: int
    max_end: int


class MainGeometryReceipt:
    def __init__(
        self,
        impl,
        cache,
        table,
        bases,
        lengths,
        page_size,
        device_lengths=None,
        host_geometry=None,
    ):
        self.owners = (cache, table, bases, lengths)
        self.inputs = impl.attn_inputs
        self.padded = bool(self.inputs.is_s_padded)
        self.signatures = (
            _signature(cache),
            *(_signature(t, version=True) for t in (table, bases, lengths)),
        )
        self.page_size = page_size
        self.head_geometry = (impl.kv_head_num, impl.head_dim)
        self.device_lengths = device_lengths
        self.host_geometry = host_geometry

    def matches(self, impl, cache, table, bases, lengths):
        if (
            self.inputs is not impl.attn_inputs
            or self.padded != bool(impl.attn_inputs.is_s_padded)
            or self.page_size != impl.attn_configs.kernel_tokens_per_block
            or self.head_geometry != (impl.kv_head_num, impl.head_dim)
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
def main_geometry_preflight(entries, host_values=None):
    """Check all entries before publishing any receipt, and clear on exceptions."""
    from rtp_llm.models_py.kernels.cuda.qsa.paged_validation import (
        geometry_flags,
        raise_geometry_error,
    )
    from rtp_llm.models_py.modules.qwen4_exp.qsa.fmha_validation import (
        _validate_main_cache_layout,
    )

    # Stage the current host values once per input and device. The FMHA layer
    # can reuse these copies while its receipt matches the current inputs.
    staged = {}
    host_values = {} if host_values is None else host_values
    request_geometry = {}

    def stage(source, device):
        key = (_signature(source, version=True), device)
        if key not in staged:
            staged[key] = source.to(device=device, non_blocking=True)
        return staged[key]

    pending = []
    flags_by_device = {}
    for impl, cache, bases, lengths in entries:
        attn_inputs = impl.attn_inputs
        table = attn_inputs.kv_cache_kernel_block_id_device
        if impl._model_prefill_geometry is not None:
            raise RuntimeError("Qwen4 Prefill geometry scopes must not overlap")
        page_size, pages = _validate_main_cache_layout(
            impl, cache, table, bases.numel(), cache.device
        )
        ld, pd = stage(lengths, cache.device), stage(bases, cache.device)
        host_geometry = None
        if lengths.device.type == bases.device.type == "cpu":
            key = (_signature(lengths, version=True), _signature(bases, version=True))
            if key not in request_geometry:
                lens, _ = _host_values(lengths, host_values)
                prefixes, _ = _host_values(bases, host_values)
                request_geometry[key] = PrefillHostGeometry(
                    lens,
                    prefixes,
                    sum(lens),
                    max((p + n for p, n in zip(prefixes, lens)), default=0),
                )
            host_geometry = request_geometry[key]
        receipt = MainGeometryReceipt(
            impl,
            cache,
            table,
            bases,
            lengths,
            page_size,
            device_lengths=(ld, pd),
            host_geometry=host_geometry,
        )
        # Physical IDs depend on pool capacity, not on the address of the pool.
        # Each cache layout has already been checked and gets its own receipt.
        flags = flags_by_device.setdefault(cache.device, {})
        key = (receipt.signatures[1:], page_size, pages, receipt.padded)
        if key not in flags:
            flags[key] = geometry_flags(ld, pd, table, page_size, pages, receipt.padded)
        pending.append((impl, receipt))
    for flags in flags_by_device.values():
        # Accept every device's verdicts before publishing any receipt.
        for verdict in torch.stack(tuple(flags.values())).cpu().tolist():
            raise_geometry_error(verdict)
    packed_lengths = {}
    try:
        for impl, receipt in pending:
            if (
                receipt.host_geometry is not None
                and receipt.host_geometry.max_end < 2**31
            ):
                ld, pd = receipt.device_lengths
                key = (id(ld), id(pd))
                if key not in packed_lengths:
                    packed_lengths[key] = (
                        ld.to(torch.int32).contiguous(),
                        pd.to(torch.int32).contiguous(),
                    )
                receipt.device_lengths = packed_lengths[key]
            receipts = impl._model_prefill_geometry
            if receipts is None:
                impl._model_prefill_geometry = receipts = {}
            cache = receipt.owners[0]
            receipts.setdefault((cache.device, cache.data_ptr()), []).append(receipt)
        yield True
    finally:
        for impl, _ in pending:
            impl._model_prefill_geometry = None


def prefill_geometry_scope(model, inputs, fmha_impl):
    if model.kv_cache is None:
        return nullcontext(True)
    attn_inputs = get_primary_attention_inputs(inputs)
    if (
        not attn_inputs.is_prefill
        or attn_inputs.is_target_verify
        or attn_inputs.is_cuda_graph
    ):
        return nullcontext(True)
    config = model.config
    if type(config) is Qwen4ModelConfig and not config.enable_qwen4_qsa:
        return nullcontext(False)
    if (
        hot_hook_runtime().enabled
        or torch.nn.modules.module._global_forward_hooks
        or torch.nn.modules.module._global_forward_pre_hooks
        or any(
            module._forward_hooks
            or module._forward_pre_hooks
            or "forward" in module.__dict__
            for module in model.modules()
        )
    ):
        # Hooks can update inference tensors without a version counter. Read
        # QSA input metadata afresh in each layer when Python can change it.
        return nullcontext(False)
    if (
        fmha_impl is None
        or torch.version.hip is not None
        or not inputs.input_ids.is_cuda
        or torch.cuda.get_device_capability(inputs.input_ids.device) != (12, 0)
    ):
        return nullcontext(True)
    from rtp_llm.models_py.model_desc.qwen4_exp import Qwen4ExpModel
    from rtp_llm.models_py.modules.qwen4_exp.qsa.sparse_gqa_impl import (
        SparseGqaFmhaImpl,
    )

    if type(model) is not Qwen4ExpModel or model.config.is_mtp:
        return nullcontext(True)

    entries = []
    host_values = {}
    for index, layer in enumerate(model.layers):
        impl = model._layer_fmha_impl(layer, fmha_impl, index)
        if type(impl) is not SparseGqaFmhaImpl:
            continue
        if (
            not impl.is_prefill
            or impl.is_target_verify
            or impl.attn_inputs.is_cuda_graph
        ):
            return nullcontext(True)
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
            return nullcontext(True)
        if not _host_values(bases, host_values)[1]:
            continue
        cache = model.kv_cache.get_layer_cache(index, model._attention_tag(index))
        entries.append((impl, cache.kv_cache_base, bases, lengths))
    return main_geometry_preflight(entries, host_values)
