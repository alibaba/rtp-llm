"""Apply the existing rowwise SWA norm/RoPE/codec before the full CP gather.

Only eager original CP8/YOCO140 rows enter this packet protocol. Admission
reuses their bind-time birth; a consumer never certifies a replacement CP
context. After admission every rank sends native uint8 [local_rows,528].
An unsupported local input cannot switch a rank back to the raw BF16 packet.
Raw KV and its joined raw-QR owner remain read-only throughout this helper.
"""

from __future__ import annotations

import math
import os

import torch

from rtp_llm.models_py.modules.dsv41.cp import CPContext
from rtp_llm.models_py.modules.dsv41.ppu import _quant, norm_store
from rtp_llm.models_py.modules.dsv41.ppu.native_cache_adapter import _source_tensor_key
from rtp_llm.models_py.modules.dsv41.ppu.optimization_config import NORM_STORE, enabled

PACK_LOCAL_SWA = "RTP_V41_PPU_CP8_YOCO_PACK_LOCAL_SWA"


def host_admitted(
    state,
    cp,
    *,
    layer_id,
    diagnostic=None,
    packed_plan=None,
    deferred_kv_norm,
    yoco_small_ops,
):
    """Choose only the uniform original-CP packet domain, without allocation.

    Common role/flag exclusions keep the existing path. An expected original
    CP birth that is missing or stale fails before a collective rather than
    selecting a different raw/packed wire format on just one rank.
    """
    if (
        deferred_kv_norm is not True
        or yoco_small_ops is not True
        or not enabled(NORM_STORE)
        or os.environ.get(PACK_LOCAL_SWA, "0") != "1"
        or diagnostic is not None
        or packed_plan is not None
        or type(layer_id) is not int
        or not 0 <= layer_id <= 20
        or not getattr(state, "prefill", False)
        or getattr(state, "idle", True)
        or getattr(state, "compacted", True)
        or torch.is_grad_enabled()
        or not isinstance(cp, CPContext)
        or cp.cp_size != 8
        or not cp.kv_cache_sharded
        or cp.gather_restore_positions is not None
        or cp is not getattr(state, "context", None)
        or cp is not getattr(state, "original_context", None)
    ):
        return False
    phase = getattr(state, "_cp_rows_phase", None)
    if not callable(phase) or not phase():
        return False
    valid = getattr(state, "_cp_full_rows_valid", None)
    if (
        getattr(state, "_cp_full_rows_birth", None) is None
        or not callable(valid)
        or not valid()
    ):
        raise RuntimeError("CP local SWA pack lost its original full-row birth")
    return True


def _raw(value):
    return (
        isinstance(value, torch.Tensor)
        and value.layout == torch.strided
        and not value.is_neg()
        and not value.is_conj()
        and not value.requires_grad
    )


def _device_supported(value):
    return (
        value.is_cuda
        and torch.version.hip is None
        and _quant.is_ppu()
        and torch.cuda.get_device_capability(value.device) == (8, 9)
        and not torch.cuda.is_current_stream_capturing()
    )


def _inputs_supported(state, cp, raw_kv, weight, eps, frequency_table):
    if (
        not _raw(raw_kv)
        or raw_kv.dtype != torch.bfloat16
        or raw_kv.shape != (cp.chunk_length, 512)
        or raw_kv.stride(1) != 1
        or raw_kv.stride(0) < 512
        or not isinstance(eps, (int, float))
        or not math.isfinite(eps)
        or eps < 0
        or not _raw(frequency_table)
        or frequency_table.dtype != torch.complex64
        or frequency_table.ndim != 2
        or frequency_table.shape[1] != 32
        or frequency_table.shape[0] < max(state.cache.lengths)
        or not frequency_table.is_contiguous()
        or frequency_table.device != raw_kv.device
        or cp.global_positions.device != raw_kv.device
        or not _device_supported(raw_kv)
    ):
        return False
    return weight is None or (
        _raw(weight)
        and weight.device == raw_kv.device
        and weight.dtype in (torch.bfloat16, torch.float32)
        and weight.shape == (512,)
        and weight.is_contiguous()
    )


def pack(state, cp, raw_kv, norm_weight, eps, frequency_table, *, layer_id):
    """Build a private native packet; never gather, write a pool or compact Q.

    This is called only after host_admitted selected the packet protocol.
    The same native row kernel is used by the old restored-full-row route;
    only its input row order/stride and grid size differ. Input ownership and
    the original CP birth are checked again after the native call.
    """
    if not host_admitted(
        state, cp, layer_id=layer_id, deferred_kv_norm=True, yoco_small_ops=True
    ):
        raise RuntimeError(
            "CP local SWA pack admission changed before its native launch"
        )
    if not _inputs_supported(state, cp, raw_kv, norm_weight, eps, frequency_table):
        raise RuntimeError("CP local SWA pack rejected an admitted native input")
    birth = state._cp_full_rows_birth
    owners = raw_kv, norm_weight, frequency_table
    keys = tuple(_source_tensor_key(value) for value in owners)
    # Original CP positions include the framework's clamped padding positions.
    # They belong to the full producer, including L20 before its callback.
    frequencies = frequency_table[cp.global_positions].contiguous()
    packed = norm_store.try_norm_rope_pack_swa(raw_kv, norm_weight, eps, frequencies)
    if packed is None:
        raise RuntimeError("CP local SWA native pack declined after protocol admission")
    if (
        not _raw(packed)
        or packed.device != raw_kv.device
        or packed.dtype != torch.uint8
        or packed.shape != (cp.chunk_length, 528)
        or not packed.is_contiguous()
        or any(
            value is not None and torch._C._is_alias_of(packed, value)
            for value in (*owners, frequencies)
        )
    ):
        raise RuntimeError(
            "CP local SWA native packet differs from its private 528-byte ABI"
        )
    if (
        state._cp_full_rows_birth is not birth
        or not state._cp_full_rows_valid()
        or state.context is not cp
        or state.original_context is not cp
        or state.compacted
        or keys != tuple(_source_tensor_key(value) for value in owners)
    ):
        raise RuntimeError(
            "CP local SWA pack changed its source owners or full-row epoch"
        )
    state.shared.setdefault("cp_local_swa_pack", []).append(
        dict(
            layer=layer_id,
            rows=cp.chunk_length,
            rawBytes=raw_kv.numel() * raw_kv.element_size(),
            packedBytes=packed.numel() * packed.element_size(),
        )
    )
    return packed
