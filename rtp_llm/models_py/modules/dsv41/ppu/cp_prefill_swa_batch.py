"""Batch the existing CP SWA segment decoder without changing its row math.

The original CP/compact births authenticate rows once when the workspace is
created. Refresh checks their direct bindings and available versions, rather
than rebuilding CP140's host geometry. Each layer uploads one pointer table
and writes the existing BF16 workspace directly, including disjoint offsets.
"""

from dataclasses import dataclass, field
import os
import weakref

import torch

from rtp_llm.models_py.modules.dsv41.cp import CPContext
from rtp_llm.models_py.modules.dsv41.ppu import _quant
from rtp_llm.models_py.modules.dsv41.ppu.native_cache_adapter import (
    SharedNativeCache,
    _source_tensor_key,
    _StagedSwaSegments,
)


_METHODS = tuple(
    (name, getattr(SharedNativeCache, name))
    for name in ("stage_swa", "try_decode_swa_range", "read_swa", "release_swa")
)
_LEGACY_DECODE = _quant.try_dequantize_swa_segments
_CP_FIELDS = (
    "cp_size", "cp_rank", "kv_cache_sharded", "cp_info",
    "global_positions", "req_id_per_token", "local_is_real", "unpad_restore",
    "relative_positions", "prefix_lengths", "input_lengths_global", "cu_seqlens_global",
    "gather_restore_positions", "input_lengths_global_host", "prefix_lengths_host",
    "chunk_lengths_per_req", "chunk_length", "padded_seq_len", "seq_len_full",
    "swa_replay_start", "swa_replay_starts_host",
)
_EXCHANGE_FIELDS = (
    "send_indices", "receive_positions", "send_sizes", "receive_sizes",
    "candidate_rows_host", "projection_groups_host", "_layout_host_buffer",
    "original_rows", "compact_rows", "receive_is_identity", "group",
)
_FLAGS = (
    "is_graph", "is_graph_commit", "verify", "proposal", "is_proposal",
    "is_speculative_decode", "is_cuda_graph", "is_target_verify", "is_packed_target_verify",
)


def _raw(value):
    return (type(value) is torch.Tensor and value.layout == torch.strided
            and not value.requires_grad and not value.is_neg() and not value.is_conj())


def _device_supported(value):
    return (value.is_cuda and torch.version.hip is None and _quant.is_ppu()
            and torch.cuda.get_device_capability(value.device) == (8, 9)
            and not torch.cuda.is_current_stream_capturing())


def _current_stream(device):
    stream = torch.cuda.current_stream(device)
    return (stream.device, int(stream.cuda_stream)), stream


def _provider_available():
    try:
        return hasattr(_quant._triton_kernels(), "dequant_v41_swa_batch")
    except ImportError:
        return False


def _unobserved(state, cache):
    from torch.nn.modules import module

    names = ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_backward_pre_hooks")
    return (
        not any(getattr(module, "_global" + name, {}) for name in names)
        and not any(getattr(state.runtime.model, name, {}) for name in names)
        and all(getattr(getattr(cache, name, None), "__func__", None) is original
                for name, original in _METHODS)
        and _quant.try_dequantize_swa_segments is _LEGACY_DECODE
    )


def _binding(owner, name):
    value = getattr(owner, name, None)
    if isinstance(value, torch.Tensor):
        return _source_tensor_key(value)
    # Host tuples in the native CP planner are immutable. Retain their owner
    # and identity rather than scanning the CP140 selected-row tuple again.
    return value if type(value) in (int, bool, float, str, type(None)) else (type(value), id(value))


def _immutable_host(value):
    if type(value) is tuple:
        return all(_immutable_host(item) for item in value)
    return type(value) in (int, bool, float, str, type(None))


def _host_layouts_immutable(state, cp):
    contexts = (cp,) if cp is state.original_context else (state.original_context, cp)
    values = [getattr(owner, name, None) for owner in contexts for name in _CP_FIELDS]
    values += [state.lengths, state.query_prefixes, state.cache.lengths]
    if state.compacted:
        values += [getattr(state.plan.exchange, name, None) for name in _EXCHANGE_FIELDS]
    # Only immutable host containers can use the cheap identity stamp below.
    # Device tensor contents remain covered by their original row birth and
    # available versions; opaque native owners are bound by object identity.
    return all(_immutable_host(value) for value in values
               if isinstance(value, (tuple, list, dict, set)))


def _bindings(state, cp):
    contexts = (cp,) if cp is state.original_context else (state.original_context, cp)
    result = (id(state.inputs), id(state.attn), id(state.cache), id(state.runtime),
              id(state.runtime.model), id(state.runtime.policy), id(state.original_context),
              id(state.lengths), id(state.query_prefixes), id(state.cache.lengths))
    result += tuple(_binding(owner, name) for owner in contexts
                   for name in _CP_FIELDS)
    result += tuple(_binding(owner, name) for owner, names in (
        (state.inputs, ("input_ids", "combo_position_ids")),
        (state.attn, ("cu_seqlens", "input_lengths", "prefix_lengths")),
    ) for name in names)
    if state.compacted:
        plan, exchange = state.plan, state.plan.exchange
        result += (id(plan), id(exchange), id(plan._ppu_cp8_query_birth),
                   id(plan.context), id(plan.original_context))
        result += tuple(_binding(exchange, name) for name in _EXCHANGE_FIELDS)
        result += (_binding(plan, "cu_seqlens"), _binding(state, "compact_ids"))
        result += tuple(_source_tensor_key(value) for pair in plan.indexer_groups for value in pair)
    return result


def _ranges(ranges):
    try:
        return tuple(tuple(entry[:4]) for entry in ranges
                     if len(entry) == 5)
    except (TypeError, IndexError):
        return None


def _output_key(out):
    # The workspace is intentionally mutable. Its allocation/layout is fixed;
    # its version cannot certify an immutable value across decoder writes.
    return _source_tensor_key(out)[:-1]


def _packed(value, device):
    return (_raw(value) and value.dtype == torch.uint8 and value.ndim == 2
            and value.shape[1] == 528 and value.is_contiguous()
            and value.device == device)


@dataclass
class SWABatchReceipt:
    state_ref: object
    cache: object
    cp: object
    full_birth: object
    phase_birth: object
    compacted: bool
    range_owner: object
    ranges: tuple
    output: torch.Tensor
    output_key: tuple
    stream_key: tuple
    stream: object
    binding_keys: tuple
    owners: tuple
    live: tuple = field(default=(), repr=False)

    def current(self):
        try:
            return self._current()
        except (AttributeError, TypeError):
            return False

    def _current(self):
        state = self.state_ref()
        if (state is None or state.cache is not self.cache or state.context is not self.cp
                or state._cp_full_rows_birth is not self.full_birth
                or bool(state.compacted) != self.compacted
                or getattr(state, "_cp_full_rows_invalid", True)
                or getattr(state, "_cp_metadata_invalid", True)
                or getattr(state.runtime, "current", None) is not state
                or not state._cp_rows_phase()
                or torch.is_grad_enabled()
                or any(getattr(owner, flag, False) for owner in (state, state.inputs, state.attn)
                       for flag in _FLAGS)
                or not _unobserved(state, self.cache)
                or _ranges(self.range_owner) != self.ranges
                or _output_key(self.output) != self.output_key):
            return False
        if self.compacted:
            if (state._cp_metadata_compact_birth is not self.phase_birth
                    or state.plan is not self.owners[0]
                    or state.plan.context is not self.cp):
                return False
        elif state.original_context is not self.cp:
            return False
        try:
            return (_current_stream(self.output.device)[0] == self.stream_key
                    and _bindings(state, self.cp) == self.binding_keys)
        except (AttributeError, TypeError):
            return False


def prepare(state, cp, cache, kv, swa_ranges):
    """Return a workspace-local receipt, or decline before upload/device work."""
    from rtp_llm.models_py.modules.dsv41.ppu.prefill_runtime import SharedForwardState

    if (type(state) is not SharedForwardState or type(cache) is not SharedNativeCache
            or os.environ.get("RTP_V41_PPU_CP8_REUSE_LOCAL_ROWS", "0") != "1"
            or os.environ.get("RTP_V41_PPU_CP8_BATCH_SWA_DECODE", "0") != "1"
            or state.cache is not cache or state.context is not cp
            or not isinstance(cp, CPContext) or cp.cp_size != 8 or not cp.kv_cache_sharded
            or not state._cp_rows_phase() or not _unobserved(state, cache)
            or not _raw(kv) or kv.dtype != torch.bfloat16 or kv.ndim != 2
            or kv.shape[1] != 512 or kv.stride(1) != 1 or kv.stride(0) < 512
            or kv.device != cp.global_positions.device or not _device_supported(kv)):
        return None
    if (state._cp_full_rows_birth is None
            or (state.compacted and state._cp_metadata_compact_birth is None)):
        return None
    try:
        if not _host_layouts_immutable(state, cp):
            return None
    except (AttributeError, TypeError) as error:
        raise RuntimeError("CP SWA batch lost its creation-owned host layout") from error
    epoch = state._cp_metadata_epoch(cp)
    if epoch is None:
        raise RuntimeError("CP SWA batch lost its creation-owned row epoch")
    stream_key, stream = _current_stream(kv.device)
    if state._cp_metadata_stream is None or state._cp_metadata_stream[0] != stream_key:
        raise RuntimeError("CP SWA batch changed its creation stream")
    ranges = _ranges(swa_ranges)
    if ranges is None or len(ranges) != len(swa_ranges) or not ranges:
        return None
    if any(len(entry) != 4 or any(type(value) is not int for value in entry)
           or not 0 <= entry[0] < len(cache.lengths) or entry[1] < 0 or entry[2] < 0
           or entry[3] < 0 or entry[1] + entry[3] > kv.shape[0]
           or entry[2] + entry[3] > cache.lengths[entry[0]] for entry in ranges):
        return None
    intervals = sorted((begin, begin + count) for _, begin, _, count in ranges if count)
    if not intervals or any(left[1] > right[0] for left, right in zip(intervals, intervals[1:])):
        return None
    # Resolve the already-existing provider before any per-layer upload.
    if not _provider_available():
        return None
    birth = state._cp_metadata_compact_birth if state.compacted else state._cp_full_rows_birth
    receipt = SWABatchReceipt(
        weakref.ref(state), cache, cp, state._cp_full_rows_birth, birth, bool(state.compacted),
        swa_ranges, ranges, kv, _output_key(kv), stream_key, stream,
        _bindings(state, cp), (state.plan if state.compacted else None, *epoch[1]),
    )
    stats = getattr(cache, "_cp_swa_batch_stats", None)
    if stats is None:
        stats = cache._cp_swa_batch_stats = dict(plans=0, launches=0, segments=0, rows=0, fallbacks=0, layers=[])
    stats["plans"] += 1
    return receipt


def try_refresh(receipt, cache, layer_id):
    """Validate every range before one descriptor upload and one native launch."""
    if receipt is None:
        return False
    if (type(receipt) is not SWABatchReceipt or cache is not receipt.cache
            or type(layer_id) is not int or not 0 <= layer_id < 40 or not receipt.current()):
        raise RuntimeError("stale CP SWA batch receipt")
    staged = getattr(cache, "_fresh_swa", {})
    sources, owners, descriptors = {}, [], []

    def decline():
        if receipt.live and receipt.live[0] == layer_id:
            raise RuntimeError("CP SWA batch source changed after admission")
        cache._cp_swa_batch_stats["fallbacks"] += 1
        return False

    if type(staged) is not dict:
        return decline()
    for request, begin, first, count in receipt.ranges:
        if not count:
            continue
        if request not in sources:
            saved = staged.get((layer_id, request))
            if (type(saved) is not _StagedSwaSegments or type(saved.prefix) is not int
                    or not 0 <= saved.prefix <= cache.lengths[request]
                    or not _packed(saved.tail, receipt.output.device)
                    or not _packed(saved.fresh, receipt.output.device)
                    or saved.prefix + saved.fresh.shape[0] > cache.lengths[request]
                    or any(torch._C._is_alias_of(receipt.output, value)
                           for value in (saved.tail, saved.fresh))):
                return decline()
            sources[request] = saved
            owners.extend((saved.tail, saved.fresh))
        saved = sources[request]
        start = first - saved.prefix + saved.tail.shape[0]
        if not 0 <= start <= saved.tail.shape[0] + saved.fresh.shape[0] - count:
            return decline()
        descriptors.append((saved.tail.data_ptr(), saved.fresh.data_ptr(), begin,
                            start, saved.tail.shape[0], count))
    if any(not 0 <= value < 2**63 for row in descriptors for value in row):
        return decline()
    keys = tuple(_source_tensor_key(value) for value in owners)
    if receipt.live and receipt.live[0] == layer_id:
        if (len(owners) != len(receipt.live[1])
                or any(old is not new for old, new in zip(receipt.live[1], owners))
                or keys != receipt.live[2]):
            raise RuntimeError("CP SWA batch source changed after admission")
    from rtp_llm.models_py.modules.dsv41.fp8._v41_swa_metadata import _upload

    device_descriptors = _upload(tuple(descriptors), receipt.output.device, torch.int64)
    reference = next(value for value in owners if value.shape[0]).view(torch.float8_e4m3fn)
    if not _quant.try_dequantize_swa_batch(
        device_descriptors, receipt.output, max(row[5] for row in descriptors), reference=reference,
    ):
        raise RuntimeError("CP SWA batch decoder declined after descriptor upload")
    # Retain only this layer. Previous work precedes this enqueue on the same
    # creation stream, so allocator reuse remains ordered when old refs drop.
    receipt.live = layer_id, tuple(owners), keys, device_descriptors
    stats = cache._cp_swa_batch_stats
    stats["launches"] += 1
    stats["segments"] += len(descriptors)
    stats["rows"] += sum(row[5] for row in descriptors)
    stats["layers"].append(layer_id)
    return True
