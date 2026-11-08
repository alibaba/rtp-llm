"""Default-off packed Attention over authoritative native PPU pools.

Only metadata is materialized. KV tensors are zero-copy strided views of
the engine allocations, including their original block padding. Eligibility
is resolved before projections, cache writes or collectives.
"""

import importlib
import inspect
import hashlib
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch

PACKED_ENV = "RTP_V41_PPU_PACKED_ATTENTION"
FUSION_ENV = "RTP_V41_PPU_FUSED_QK_ROPE"
STREAMS_ENV = "RTP_V41_PPU_QKV_STREAMS"
DEPENDENCY_ENV = "RTP_V41_PPU_PACKED_DEPENDENCY"
PACKED_REVISION = "4305d1bdaeb8451eb5d78c67a5122ecf38f5014d"
PACKED_INTERFACE_SHA256 = "8861f77da7d7900ffeb60cb2f652176d4342c947d516fbcc0928eba81e952d43"
OLD_NATIVE_SHA256 = "26037418e0d634cc39eff935fe40afab0e069928dc42f9d4ffaeda3c54efefa9"


def dependency_reason(receipt):
    """Pure identity/schema gate; no import, allocation or native probe."""
    modules = {"flash_mla": "flash_mla_cuda",
               "flash_mla_v41_packed": "flash_mla_v41_packed.flash_mla_cuda"}
    if (receipt.get("format") != "rtp-v41-packed-flashmla-v1"
            or receipt.get("source_revision") != PACKED_REVISION
            or receipt.get("upstream_interface_sha256") != PACKED_INTERFACE_SHA256
            or receipt.get("native_entry") != "sparse_decode_fwd"
            or modules.get(receipt.get("module")) != receipt.get("extension_module")
            or receipt.get("module") not in modules):
        return "packed_native_profile_unqualified"
    for field in ("wrapper_sha256", "extension_sha256"):
        value = receipt.get(field, "")
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            return "packed_native_hash_missing"
    if receipt["module"] == "flash_mla" and receipt["wrapper_sha256"] != PACKED_INTERFACE_SHA256:
        return "packed_wrapper_revision_mismatch"
    if receipt["extension_sha256"] == OLD_NATIVE_SHA256:
        return "packed_old_native_abi"
    return None


@lru_cache(maxsize=1)
def _load_packed_backend(receipt_path):
    receipt = json.loads(Path(receipt_path).read_text())
    reason = dependency_reason(receipt)
    if reason:
        raise ValueError(reason)
    module = importlib.import_module(receipt["module"])
    extension = importlib.import_module(receipt["extension_module"])
    native = getattr(module, "flash_mla_with_kvcache", None)
    if (not callable(native) or not callable(getattr(module, "get_mla_metadata", None))
            or not callable(getattr(extension, receipt["native_entry"], None))
            or native.__globals__.get("flash_mla_cuda") is not extension):
        raise ValueError("packed_native_capability_missing")
    wrapper = Path(inspect.getfile(native)).resolve()
    binary = Path(extension.__file__).resolve()
    for path, field in ((wrapper, "wrapper_sha256"), (binary, "extension_sha256")):
        if hashlib.sha256(path.read_bytes()).hexdigest() != receipt[field]:
            raise ValueError("packed_native_import_hash_mismatch:" + field)
    identity = {key: receipt[key] for key in (
        "source_revision", "upstream_interface_sha256", "module", "extension_module",
        "native_entry", "wrapper_sha256", "extension_sha256",
    )}
    identity.update(receipt_path=receipt_path, wrapper_path=str(wrapper), extension_path=str(binary))
    return module, identity


def load_packed_backend():
    """Require L0's immutable isolated-build receipt; never probe a kernel."""
    path = Path(os.environ.get(DEPENDENCY_ENV, ""))
    if not path.is_absolute():
        raise ValueError("packed_native_dependency_receipt_required")
    return _load_packed_backend(str(path.resolve()))


def enabled(name):
    value = os.environ.get(name, "0")
    if value not in ("0", "1"):
        raise ValueError(f"{name} must be explicitly 0 or 1")
    return value == "1"


def ordinary_eager_reason(state):
    if state is None or state.idle:
        return "idle"
    if getattr(state, "is_graph", False):
        return "graph_deferred_n10"
    if bool(getattr(state.attn, "is_target_verify", False)):
        return "verify_deferred_n10"
    if getattr(state, "restoring_aux", False) or getattr(state, "compacted", False):
        return "ced_deferred_n10"
    if state.runtime.policy.mode != "full":
        return "ced_policy_deferred_n10"
    if getattr(state.runtime.policy, "max_rollback_tokens", 0):
        return "dspark_deferred_n10"
    if getattr(state.runtime.model, "_is_speculative", False):
        return "speculative_deferred_n10"
    policy = getattr(state.cache, "policy", None)
    if any(getattr(policy, name, False) for name in ("ced", "bounded_replay", "encoder_replay")):
        return "ced_cache_deferred_n10"
    return None


def packed_pool_view(pool, width):
    """Retain the framework block stride without reshape or copy."""
    tensor = pool.tensor
    return tensor.as_strided(
        (tensor.shape[0], pool.entries, 1, width),
        (tensor.stride(0), width, width, 1),
    )


def physical_slots(pool, requests, raw, *, ratio, full, valid):
    """Use each tag's OWN raw span, entries and physical table."""
    logical = raw.clamp_min(0) // pool.span
    in_table = (logical < pool.table.shape[1]) & valid
    blocks = pool.table[
        requests.clamp(0, pool.table.shape[0] - 1)[:, None].expand_as(raw),
        logical.clamp_max(pool.table.shape[1] - 1)
    ].long()
    offset = (raw % pool.span) // ratio if full else raw % pool.entries
    live = in_table & (blocks >= 0) & (blocks < pool.tensor.shape[0])
    live = live & (offset >= 0) & (offset < pool.entries)
    return torch.where(live, blocks * pool.entries + offset, -1).int()


def compact_prefix(indices):
    """Stable live prefix; lengths never count padding or interior holes."""
    live = indices >= 0
    ranks = live.long().cumsum(-1)
    lengths = live.sum(-1, dtype=torch.int32)
    columns = torch.arange(indices.shape[-1], device=indices.device)[None]
    destinations = torch.where(live, ranks - 1, lengths[:, None] + columns - ranks)
    compact = torch.empty_like(indices).scatter_(1, destinations, indices)
    return compact[:, None], lengths


@dataclass
class PackedAttentionPlan:
    swa_cache: object
    global_cache: object
    swa_pool: object
    global_pool: object
    positions: object
    request_ids: object
    valid_mask: object
    ratio: int
    window: int
    native: object
    metadata_factory: object
    swa_indices: object = None
    swa_lengths: object = None
    global_indices: object = None
    global_lengths: object = None
    backend_identity: object = None

    def bind_selection(self, selected):
        offsets = torch.arange(self.window, device=self.positions.device)[None]
        raw = self.positions[:, None] - (self.window - 1) + offsets
        valid = self.valid_mask[:, None] & (raw >= 0)
        slots = physical_slots(
            self.swa_pool, self.request_ids, raw, ratio=1, full=False, valid=valid
        )
        self.swa_indices, self.swa_lengths = compact_prefix(slots)
        if self.global_pool is not None:
            visible = (self.positions[:, None] + 1) // self.ratio
            columns = torch.arange(selected.shape[-1], device=selected.device)[None]
            live = self.valid_mask[:, None] & (columns < visible)
            live = live & (selected >= 0) & (selected < visible)
            slots = physical_slots(
                self.global_pool, self.request_ids, selected.long() * self.ratio,
                ratio=self.ratio, full=True, valid=live,
            )
            self.global_indices, self.global_lengths = compact_prefix(slots)
        return self


def _pool_reason(pool, width, lengths, *, ratio, full):
    tensor, table = pool.tensor, pool.table
    if (tensor.ndim != 2 or tensor.dtype != torch.uint8
            or tensor.device.type != "cuda" or tensor.stride(1) != 1
            or tensor.shape[1] < pool.entries * width
            or tensor.stride(0) % 16 or tensor.data_ptr() % 16
            or pool.entries <= 0 or pool.entries & (pool.entries - 1)
            or pool.span <= 0
            or tensor.shape[0] * pool.entries >= 2 ** 31):
        return "unsupported_pool_layout"
    if (table.ndim != 2 or table.shape[0] < len(lengths) or not table.shape[1]
            or table.device != tensor.device or table.dtype not in (torch.int32, torch.int64)):
        return "unsupported_physical_table"
    if full and (pool.span % ratio or pool.span // ratio > pool.entries):
        return "unsupported_compressed_page"
    for rid, total in enumerate(lengths):
        last = (total // ratio - 1) * ratio if full else total - 1
        if last < 0:
            continue
        columns = last // pool.span + 1
        if columns > table.shape[1]:
            return "short_physical_table"
        blocks = table[rid, :columns]
        if bool(((blocks < 0) | (blocks >= tensor.shape[0])).any()):
            return "unallocated_physical_page"
    return None


def prepare_packed_attention(attn, state, context, x):
    """Return (plan, reason) before the first state write; no launch probes."""
    if not enabled(PACKED_ENV):
        return None, "disabled"
    reason = ordinary_eager_reason(state)
    if reason:
        return None, reason
    if (x.dtype != torch.bfloat16 or x.device.type != "cuda" or not x.shape[0]
            or attn.geometry.head_dim != 512 or attn.geometry.window != 128
            or not 0 < attn.geometry.n_heads <= 128):
        return None, "unsupported_query_geometry"
    if torch.cuda.is_current_stream_capturing():
        return None, "capture_deferred_n10"
    if not state.prefill and any(n != 1 for n in state.lengths):
        return None, "multi_query_decode_deferred_n10"
    if not bool(context.valid_mask.all()):
        return None, "invalid_rows_deferred_n10"
    cache = state.cache
    pool = cache.pools[attn.layer_id, "swa_kv"]
    if state.prefill and (any(state.query_prefixes) or any(n > pool.entries for n in cache.lengths)):
        return None, "prefill_swa_not_resident"
    if pool.entries < attn.geometry.window:
        return None, "short_swa_ring"
    reason = _pool_reason(pool, 528, cache.lengths, ratio=1, full=False)
    if reason:
        return None, reason
    source, ratio = attn.kv_source_layer_id, attn.compress_ratio
    # SWA-only Prefill (layers 0 and 1) uses the single-bank packed kernel.
    # The qualified comparison allows its FP32 softmax accumulation order.
    global_pool = None
    if ratio:
        if ratio not in (1, 2):
            return None, "unsupported_compression_ratio"
        tag = "global_kv_2" if ratio == 2 else "global_kv_1"
        global_pool = cache.pools[source, tag]
        reason = _pool_reason(global_pool, 288, cache.lengths, ratio=ratio, full=True)
        if reason:
            return None, reason
    try:
        module, identity = load_packed_backend()
    except (ImportError, OSError, ValueError):
        return None, "packed_native_dependency_unqualified"
    native = getattr(module, "flash_mla_with_kvcache", None)
    metadata = getattr(module, "get_mla_metadata", None)
    if not callable(native) or not callable(metadata):
        return None, "packed_abi_unavailable"
    required = {"softmax_scale", "extra_k_cache", "extra_indices_in_kvcache",
                "topk_length", "extra_topk_length", "tile_scheduler_metadata"}
    try:
        parameters = inspect.signature(native).parameters
    except (TypeError, ValueError):
        return None, "packed_abi_unqualified"
    if not required.issubset(parameters):
        return None, "packed_abi_unqualified"
    sink = attn.attn_sink
    if (sink.dtype != torch.float32 or sink.shape != (attn.geometry.n_heads,)
            or sink.device != x.device):
        return None, "unsupported_sink"
    return PackedAttentionPlan(
        packed_pool_view(pool, 528),
        packed_pool_view(global_pool, 288) if global_pool is not None else None,
        pool, global_pool, context.positions, context.request_ids.long(),
        context.valid_mask, ratio, attn.geometry.window, native, metadata,
        backend_identity=identity,
    ), "eligible"


def run_packed_attention(q, plan, *, prefill, softmax_scale, attn_sink):
    if prefill:
        from rtp_llm.models_py.modules.dsv41.ppu.shared_packed_prefill import packed_prefill

        return packed_prefill(q, plan, softmax_scale=softmax_scale, attn_sink=attn_sink)
    from rtp_llm.models_py.modules.dsv41.ppu.shared_packed_decode import packed_decode

    return packed_decode(q, plan, softmax_scale=softmax_scale, attn_sink=attn_sink)
