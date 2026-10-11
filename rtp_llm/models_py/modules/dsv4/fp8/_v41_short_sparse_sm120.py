"""Explicit short SM120 attention with preserved SWA mask positions.

Uses maintained TileLang attention and typed cache readers. Callers supply
admitted initial-prefill or first-ring decode plans; this component does not
select runtime policy, write caches or perform GLOBAL selection.
"""

import torch

from rtp_llm.models_py.modules.dsv4 import tilelang_kernels
from rtp_llm.models_py.modules.dsv4.v41_cache import _shares_storage


def _require_output(query, out, inputs):
    if out is not None and (
        not isinstance(out, torch.Tensor)
        or out.shape != query.shape
        or out.dtype != query.dtype
        or out.device != query.device
        or not out.is_contiguous()
        or any(_shares_storage(out, value) for value in inputs if value is not None)
    ):
        raise ValueError("Short sparse output must match query and not alias inputs")


def prefill_plan(indices, lengths, kv_rows, global_rows):
    rows = len(lengths)
    if (
        type(global_rows) is not int
        or not 0 <= global_rows <= rows
        or kv_rows != rows + global_rows
        or not 1 <= rows <= 128
        or indices.ndim != 3
        or indices.shape[:2] != (rows, 1)
        or indices.dtype != torch.int32
        or not indices.is_cuda
        or not indices.is_contiguous()
        or lengths.shape != (rows,)
        or lengths.dtype != torch.int32
        or lengths.device != indices.device
        or not lengths.is_contiguous()
    ):
        raise ValueError("Short prefill requires one initial SWA workspace")
    # Membership preserves the supplied GLOBAL selection and causal mask. The
    # fixed SWA segment retains holes rather than compacting its live entries.
    positions = torch.cat(
        (
            torch.arange(global_rows, kv_rows, device=indices.device),
            torch.arange(global_rows, device=indices.device),
        )
    ).int()
    live = (
        torch.arange(indices.shape[-1], device=indices.device)[None] < lengths[:, None]
    )
    matches = (indices[:, 0, :, None] == positions[None, None]) & live[:, :, None]
    return torch.where(matches.any(1), positions[None], -1).unsqueeze(1).contiguous()


def prefill_attention(query, kv, indices, lengths, sinks, *, global_rows, out=None):
    if (
        query.ndim != 3
        or query.shape[-1] != 512
        or not 1 <= query.shape[1] <= 64
        or query.dtype != torch.bfloat16
        or not query.is_cuda
        or not query.is_contiguous()
        or torch.cuda.get_device_capability(query.device) != (12, 0)
        or kv.shape != (len(query) + global_rows, 1, 512)
        or kv.dtype != query.dtype
        or kv.device != query.device
        or not kv.is_contiguous()
        or indices.device != query.device
        or lengths.shape != query.shape[:1]
        or sinks.shape != (query.shape[1],)
        or sinks.dtype != torch.float32
        or sinks.device != query.device
        or not sinks.is_contiguous()
    ):
        raise ValueError("Unsupported short SM120 sparse prefill geometry")
    _require_output(query, out, (query, kv, indices, lengths, sinks))
    plan = prefill_plan(indices, lengths, len(kv), global_rows)
    output = torch.cat(
        [
            tilelang_kernels.sparse_attn(
                query[:, head : head + 8].unsqueeze(0).contiguous(),
                kv.reshape(1, -1, 512),
                sinks[head : head + 8].contiguous(),
                plan.reshape(1, len(query), -1),
                512**-0.5,
                initial_max=-1e30,
            )[0]
            for head in range(0, query.shape[1], 8)
        ],
        1,
    ).contiguous()
    if out is not None:
        out.copy_(output)
        return out
    return output


def decode_plans(plans, starts, *, swa_entries):
    slots = plans.get("swa")
    if (
        slots is None
        or set(plans) not in ({"swa"}, {"swa", "global"})
        or slots.ndim != 2
        or slots.shape[1] != 128
        or not slots.is_cuda
        or slots.dtype not in (torch.int32, torch.int64)
        or not slots.is_contiguous()
        or starts.shape != slots.shape[:1]
        or starts.dtype != torch.int32
        or starts.device != slots.device
        or not starts.is_contiguous()
        or swa_entries not in (128, 130)
    ):
        raise ValueError("Short decode requires device positions and a 128-slot ring")
    # Generation one adds MTP entries to each page. Physical page stride is
    # distinct from the 128-slot SWA ring and comes from the cache layout.
    ring = (starts[:, None] + 1 + torch.arange(128, device=slots.device)) % 128
    matches = (slots[:, :, None] >= 0) & (
        slots[:, :, None] % swa_entries == ring[:, None]
    )
    ordered = torch.where(matches, slots[:, :, None], -1).amax(1).contiguous()
    return dict(plans, swa=ordered)


def _workspace_decode(query, sinks, caches, plans, *, valid_rows=None):
    if not tilelang_kernels.tilelang_available():
        raise RuntimeError("The maintained TileLang sparse provider is unavailable")
    if (
        not query.is_cuda
        or torch.cuda.get_device_capability(query.device) != (12, 0)
        or query.ndim != 3
        or query.shape[-1] != 512
        or not 1 <= query.shape[1] <= 64
        or query.dtype != torch.bfloat16
        or not query.is_contiguous()
        or sinks.shape != (query.shape[1],)
        or sinks.dtype != torch.float32
        or sinks.device != query.device
        or not sinks.is_contiguous()
        or set(caches) != set(plans)
        or set(caches) not in ({"swa"}, {"swa", "global"})
    ):
        raise ValueError("Unsupported SM120 TileLang decode geometry")
    if valid_rows is not None:
        if (
            valid_rows.shape != query.shape[:1]
            or valid_rows.dtype != torch.bool
            or valid_rows.device != query.device
            or not valid_rows.is_contiguous()
        ):
            raise ValueError("TileLang decode requires device bool valid_rows")
        query = torch.where(valid_rows[:, None, None], query, 0)
    rows, segments, slots = query.shape[0], [], []
    for name in ("swa", "global"):
        if name not in caches:
            continue
        cache, plan = caches[name], plans[name]
        width = 128 if name == "swa" else 512
        tags = ("swa_kv",) if name == "swa" else ("csa_kv", "hca_kv")
        if (
            cache.tag not in tags
            or cache.gen_num_per_cycle not in (0, 1)
            or cache.raw.device != query.device
            or plan.device != query.device
            or plan.dtype not in (torch.int32, torch.int64)
            or plan.shape != (rows, width)
            or not plan.is_contiguous()
        ):
            raise ValueError("Unsupported SM120 TileLang cache/plan")
        selected = plan
        if valid_rows is not None:
            selected = torch.where(valid_rows[:, None], selected, -1)
        segments.append(cache.read(selected.reshape(-1)).view(rows, width, 512))
        slots.append(selected)
    workspace = torch.cat(segments, dim=1).contiguous()
    live = torch.cat(slots, dim=1) >= 0
    positions = torch.arange(live.shape[1], dtype=torch.int32, device=query.device)
    indices = torch.where(live, positions[None], -1).unsqueeze(1).contiguous()
    if not rows:
        return torch.empty_like(query), workspace, indices
    output = torch.cat(
        [
            tilelang_kernels.sparse_attn(
                query[:, start : start + 8].unsqueeze(1).contiguous(),
                workspace,
                sinks[start : start + 8].contiguous(),
                indices,
                512**-0.5,
                initial_max=-1e30,
            )
            for start in range(0, query.shape[1], 8)
        ],
        dim=2,
    ).reshape_as(query)
    if valid_rows is not None:
        output = torch.where(valid_rows[:, None, None], output, 0)
    return output.contiguous(), workspace, indices


class ShortSparseDecodeAttention:
    """Retain first-ring SWA holes; warm each layout before graph capture."""

    def __init__(self):
        self._warmed = set()

    def __call__(
        self,
        query,
        sinks,
        swa,
        swa_slots,
        *,
        start_positions,
        valid_rows,
        global_cache=None,
        global_slots=None,
        out=None
    ):
        if (global_cache is None) != (global_slots is None):
            raise ValueError(
                "Short sparse GLOBAL cache and plan must be supplied together"
            )
        caches, plans = {"swa": swa}, {"swa": swa_slots}
        inputs = [query, sinks, start_positions, valid_rows, swa.raw, swa_slots]
        if global_cache is not None:
            caches["global"], plans["global"] = global_cache, global_slots
            inputs.extend((global_cache.raw, global_slots))
        _require_output(query, out, inputs)
        key = (
            query.device,
            tuple(query.shape),
            valid_rows is not None,
            tuple(
                (
                    name,
                    cache.tag,
                    cache.gen_num_per_cycle,
                    tuple(cache.raw.shape),
                    tuple(cache.raw.stride()),
                    plans[name].dtype,
                )
                for name, cache in caches.items()
            ),
        )
        if (
            query.is_cuda
            and torch.cuda.is_current_stream_capturing()
            and key not in self._warmed
        ):
            raise RuntimeError("Warm short sparse decode layout before graph capture")
        ordered = decode_plans(plans, start_positions, swa_entries=swa.layout.entries)
        output, _, _ = _workspace_decode(
            query, sinks, caches, ordered, valid_rows=valid_rows
        )
        self._warmed.add(key)
        if out is not None:
            out.copy_(output)
            return out
        return output
