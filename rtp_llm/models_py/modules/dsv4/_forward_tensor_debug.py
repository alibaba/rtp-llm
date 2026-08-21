"""Small stdout tensor summaries for DSV4 smoke debugging.

Enabled with ``DSV4_FORWARD_TENSOR_DEBUG=1`` or
``DSV4_DEBUG_HIDDEN_COMPARE=1``.  The helper intentionally snapshots only the
rows that feed sampling (prefill request tails / decode rows), so long-context
smokes do not copy the full hidden state back to CPU.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from typing import Any, Dict, Iterable, Optional

import torch

_COUNTERS: Dict[str, int] = defaultdict(int)


def enabled() -> bool:
    return (
        os.environ.get("DSV4_FORWARD_TENSOR_DEBUG", "0") != "0"
        or os.environ.get("DSV4_DEBUG_HIDDEN_COMPARE", "0") != "0"
    )


def _max_records() -> int:
    return int(os.environ.get("DSV4_FORWARD_TENSOR_DEBUG_MAX", "128"))


def _rank_info() -> Dict[str, int]:
    rank = 0
    world = 1
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            rank = int(dist.get_rank())
            world = int(dist.get_world_size())
    except Exception:
        pass
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", rank)))
    return {
        "rank": rank,
        "world": world,
        "local_rank": local_rank,
        "pid": os.getpid(),
    }


def _tolist(tensor: Optional[torch.Tensor], limit: int = 16) -> Optional[list]:
    if tensor is None:
        return None
    flat = tensor.detach().reshape(-1)
    if flat.numel() == 0:
        return []
    return flat[:limit].cpu().tolist()


def _stats(tensor: torch.Tensor) -> Dict[str, Any]:
    cpu = tensor.detach().to(torch.float32).cpu().contiguous()
    numel = int(cpu.numel())
    if numel == 0:
        return {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "numel": 0,
        }
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "numel": numel,
        "sum": float(cpu.sum().item()),
        "mean": float(cpu.mean().item()),
        "l2": float(torch.linalg.vector_norm(cpu).item()),
        "max_abs": float(cpu.abs().max().item()),
        "nan": int(torch.isnan(cpu).sum().item()),
        "inf": int(torch.isinf(cpu).sum().item()),
        "hash": hashlib.md5(cpu.numpy().tobytes()).hexdigest(),
        "values_head": cpu.reshape(-1)[:8].tolist(),
    }


def _tensor_list(value: Any) -> Optional[list]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().reshape(-1).tolist()
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        return list(value)
    return [value]


def _topk(
    last_hidden: torch.Tensor, head_weight: Optional[torch.Tensor]
) -> Dict[str, Any]:
    if head_weight is None or last_hidden.numel() == 0:
        return {}
    logits = torch.mm(last_hidden.to(head_weight.dtype), head_weight.t()).float()
    k = min(8, int(logits.shape[-1]))
    values, indices = torch.topk(logits, k=k, dim=-1)
    return {
        "top_indices": indices.detach().cpu().tolist(),
        "top_values": values.detach().cpu().tolist(),
    }


def _record_allowed(path: str) -> Optional[int]:
    if not enabled():
        return None
    idx = _COUNTERS[path]
    if idx >= _max_records():
        return None
    _COUNTERS[path] += 1
    return idx


def print_prefill(
    *,
    hidden: torch.Tensor,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor],
    attn_inputs: Any,
    cp_ctx: Any,
    head_weight: Optional[torch.Tensor],
    step: int,
) -> None:
    record_idx = _record_allowed("prefill")
    if record_idx is None:
        return

    selected = torch.empty(
        (0, hidden.shape[-1]), dtype=hidden.dtype, device=hidden.device
    )
    owned_req_ids: list[int] = []
    last_positions: list[int] = []
    if cp_ctx is not None:
        req_ids = cp_ctx.req_id_per_token.to(device=hidden.device, dtype=torch.long)
        global_pos = cp_ctx.global_positions.to(device=hidden.device, dtype=torch.long)
        real = cp_ctx.local_is_real.to(device=hidden.device, dtype=torch.bool)
        prefix = cp_ctx.prefix_lengths.to(device=hidden.device, dtype=torch.long)
        lengths = cp_ctx.input_lengths_global.to(device=hidden.device, dtype=torch.long)
        rows = []
        for req_id in range(int(lengths.numel())):
            last_pos = int((prefix[req_id] + lengths[req_id] - 1).item())
            last_positions.append(last_pos)
            mask = (req_ids == req_id) & (global_pos == last_pos) & real
            idx = torch.nonzero(mask, as_tuple=False).reshape(-1)
            if idx.numel() > 0:
                rows.append(hidden.index_select(0, idx[:1]))
                owned_req_ids.append(req_id)
        if rows:
            selected = torch.cat(rows, dim=0)
    elif cu_seqlens is not None and cu_seqlens.numel() >= 2:
        last_idx = (
            cu_seqlens.to(device=hidden.device, dtype=torch.long)[1:] - 1
        ).clamp(min=0)
        selected = hidden.index_select(0, last_idx)
        last_positions = _tolist(positions.index_select(0, last_idx), 1024) or []
        owned_req_ids = list(range(int(last_idx.numel())))
    elif hidden.shape[0] > 0:
        selected = hidden[-1:].contiguous()
        last_positions = _tolist(positions[-1:], 1) or []
        owned_req_ids = [0]

    payload: Dict[str, Any] = {
        "tag": "DSV4_FORWARD_TENSOR",
        "path": "prefill",
        "record": record_idx,
        "step": int(step),
        **_rank_info(),
        "raw_q_merge_env": os.environ.get("DSV4_CP_CACHE_HIT_RAW_Q_MERGE", "0"),
        "hidden_shape": list(hidden.shape),
        "hidden_dtype": str(hidden.dtype),
        "input_ids_shape": list(input_ids.shape),
        "input_ids_tail": _tolist(input_ids[-8:], 8),
        "positions_head": _tolist(positions[:8], 8),
        "positions_tail": _tolist(positions[-8:], 8),
        "request_last_positions": last_positions,
        "owned_request_ids": owned_req_ids,
        "last_hidden": _stats(selected),
    }
    payload.update(_topk(selected, head_weight))
    if attn_inputs is not None:
        for name in ("input_lengths", "prefix_lengths", "sequence_lengths"):
            payload[name] = _tensor_list(getattr(attn_inputs, name, None))
    if cp_ctx is not None:
        payload["cp"] = {
            "cp_size": int(cp_ctx.cp_size),
            "cp_rank": int(cp_ctx.cp_rank),
            "kv_cache_sharded": bool(getattr(cp_ctx, "kv_cache_sharded", False)),
            "seq_len_full": int(cp_ctx.seq_len_full),
            "seq_len_total": int(cp_ctx.seq_len_total),
            "chunk_length": int(cp_ctx.chunk_length),
        }
        for out_name, attr_name in (
            ("global_input_lengths", "input_lengths_global"),
            ("global_prefix_lengths", "prefix_lengths"),
            ("global_cu_seqlens", "cu_seqlens_global"),
        ):
            payload[out_name] = _tensor_list(getattr(cp_ctx, attr_name, None))
    print(json.dumps(payload, sort_keys=True), flush=True)


def print_decode(
    *,
    hidden: torch.Tensor,
    input_ids_2d: torch.Tensor,
    attn_inputs: Any,
    meta: Any,
    head_weight: Optional[torch.Tensor],
    step: int,
) -> None:
    record_idx = _record_allowed("decode")
    if record_idx is None:
        return

    selected = hidden.reshape(-1, hidden.shape[-1]).contiguous()
    payload: Dict[str, Any] = {
        "tag": "DSV4_FORWARD_TENSOR",
        "path": "decode",
        "record": record_idx,
        "step": int(step),
        **_rank_info(),
        "raw_q_merge_env": os.environ.get("DSV4_CP_CACHE_HIT_RAW_Q_MERGE", "0"),
        "hidden_shape": list(hidden.shape),
        "hidden_dtype": str(hidden.dtype),
        "input_ids": input_ids_2d.detach().cpu().tolist(),
        "batch_size": int(getattr(meta, "batch_size", selected.shape[0])),
        "q_len": int(getattr(meta, "q_len_per_req", 1)),
        "hidden": _stats(selected),
    }
    payload.update(_topk(selected, head_weight))
    seq_lens = getattr(attn_inputs, "sequence_lengths", None)
    if seq_lens is not None:
        payload["sequence_lengths"] = _tensor_list(seq_lens)
    start_pos = getattr(meta, "start_pos", None)
    if start_pos is not None:
        payload["start_pos"] = _tensor_list(start_pos)
    print(json.dumps(payload, sort_keys=True), flush=True)
