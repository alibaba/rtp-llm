"""Scoped SM120 decode diagnostic using maintained TileLang attention equations.

The default workspace helper retains V4's valid-first-tile requirement. The
scoped V4.1 provider explicitly selects the released finite maximum initializer.
"""

from contextlib import contextmanager
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4 import tilelang_kernels
from rtp_llm.models_py.modules.dsv4.fp8 import _v41_decode_attention_sm120 as decode
from rtp_llm.models_py.modules.dsv4.v41_cache import _shares_storage


def workspace_decode(
    query, sinks, caches, plans, *, valid_rows=None, initial_max=-float("inf")
):
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
    if initial_max not in (-float("inf"), -1e30):
        raise ValueError("Unsupported TileLang maximum initializer")
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
                initial_max=initial_max,
            )
            for start in range(0, query.shape[1], 8)
        ],
        dim=2,
    ).reshape_as(query)
    if valid_rows is not None:
        output = torch.where(valid_rows[:, None, None], output, 0)
    return output.contiguous(), workspace, indices


class _ReferenceDecodeAttention:
    def __init__(self):
        self._warmed = set()

    def __call__(
        self,
        query,
        sinks,
        swa,
        swa_slots,
        *,
        valid_rows,
        global_cache=None,
        global_slots=None,
        out=None
    ):
        if (global_cache is None) != (global_slots is None):
            raise ValueError("TileLang GLOBAL cache and plan must be supplied together")
        caches, plans = {"swa": swa}, {"swa": swa_slots}
        inputs = [query, sinks, valid_rows, swa.raw, swa_slots]
        if global_cache is not None:
            caches["global"], plans["global"] = global_cache, global_slots
            inputs.extend((global_cache.raw, global_slots))
        if out is not None and (
            out.shape != query.shape
            or out.dtype != query.dtype
            or out.device != query.device
            or not out.is_contiguous()
            or any(_shares_storage(out, value) for value in inputs)
        ):
            raise ValueError("TileLang output must match query and not alias inputs")
        key = (
            query.device,
            tuple(query.shape),
            tuple(
                (cache.tag, cache.gen_num_per_cycle, plans[name].dtype)
                for name, cache in caches.items()
            ),
        )
        if (
            query.is_cuda
            and torch.cuda.is_current_stream_capturing()
            and key not in self._warmed
        ):
            raise RuntimeError("Warm TileLang decode layout before graph capture")
        output, _, _ = workspace_decode(
            query, sinks, caches, plans, valid_rows=valid_rows, initial_max=-1e30
        )
        self._warmed.add(key)
        if out is not None:
            out.copy_(output)
            return out
        return output


@contextmanager
def decode_tilelang_reference(enabled):
    if not enabled:
        yield
        return
    if torch.cuda.get_device_capability() != (12, 0):
        raise ValueError("TileLang decode reference is qualified only on SM120")
    with patch.object(decode, "_attention", _ReferenceDecodeAttention()):
        yield
