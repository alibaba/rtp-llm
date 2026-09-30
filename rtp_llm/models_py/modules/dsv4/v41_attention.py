"""Bounded V4.1 joint sparse attention over typed SWA and GLOBAL caches.

Indices are already causal physical slots, with -1 holes; the caller owns
request isolation, ring generations and source selection. Triton validates
metadata without synchronizing. Value checks are diagnostic (eager or
TRITON_DEBUG). This operator does not select keys or write caches.
"""

import torch

from rtp_llm.models_py.modules.dsv4.v41_cache import V41PagedCache, _shares_storage

_SCALE = 512**-0.5
_TILE = 32


class V41SparseAttention:
    def __init__(self, *, backend: str = "triton"):
        if backend not in ("triton", "eager"):
            raise ValueError("V4.1 attention backend must be triton or eager")
        self.backend = backend
        self._warmed = set()

    def __call__(
        self,
        query: torch.Tensor,
        sinks: torch.Tensor,
        swa: V41PagedCache,
        swa_slots: torch.Tensor,
        *,
        valid_rows: torch.Tensor,
        global_cache: V41PagedCache | None = None,
        global_slots: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            query.ndim != 3
            or query.shape[-1] != 512
            or not 1 <= query.shape[1] <= 64
            or query.dtype != torch.bfloat16
            or not query.is_contiguous()
        ):
            raise ValueError(
                "V4.1 attention query must be contiguous BF16 [R, H<=64, 512]"
            )
        rows, heads, _ = query.shape
        if (
            sinks.shape != (heads,)
            or sinks.dtype != torch.float32
            or sinks.device != query.device
            or not sinks.is_contiguous()
            or valid_rows.shape != (rows,)
            or valid_rows.dtype != torch.bool
            or valid_rows.device != query.device
            or not valid_rows.is_contiguous()
        ):
            raise ValueError(
                "V4.1 attention requires device FP32 sinks and bool valid_rows"
            )
        if swa.tag != "swa_kv":
            raise ValueError("V4.1 attention requires a SWA cache")
        if (global_cache is None) != (global_slots is None):
            raise ValueError("V4.1 GLOBAL cache and slots must be supplied together")
        pools = [(swa, swa_slots, 128)]
        if global_cache is not None:
            if global_cache.tag not in ("csa_kv", "hca_kv"):
                raise ValueError("V4.1 attention requires a GLOBAL cache")
            pools.append((global_cache, global_slots, 512))
        inputs = [query, sinks, valid_rows]
        for cache, slots, limit in pools:
            if (
                slots.ndim != 2
                or slots.shape[0] != rows
                or slots.shape[1] > limit
                or slots.dtype not in (torch.int32, torch.int64)
                or not slots.is_contiguous()
                or slots.device != query.device
                or cache.raw.device != query.device
            ):
                raise ValueError(
                    "V4.1 attention slots must be bounded contiguous device integers"
                )
            inputs.extend((cache.raw, slots))
        if out is None:
            out = torch.empty_like(query)
        if (
            out.shape != query.shape
            or out.dtype != query.dtype
            or out.device != query.device
            or not out.is_contiguous()
            or any(_shares_storage(out, value) for value in inputs)
        ):
            raise ValueError(
                "V4.1 attention output must match query and not alias inputs"
            )
        if self.backend == "eager":
            if query.is_cuda and torch.cuda.is_current_stream_capturing():
                raise RuntimeError("V4.1 eager attention is diagnostic, not capturable")
            self._eager(query, sinks, pools, valid_rows, out)
        else:
            if not query.is_cuda or torch.cuda.get_device_capability(query.device) != (
                12,
                0,
            ):
                raise RuntimeError(
                    "V4.1 sparse attention requires an exact SM120 device"
                )
            key = (
                query.device,
                heads,
                tuple(
                    (cache.tag, cache.layout.entries, slots.shape[1], slots.dtype)
                    for cache, slots, _ in pools
                ),
            )
            if torch.cuda.is_current_stream_capturing() and key not in self._warmed:
                raise RuntimeError("Warm V4.1 attention layout before graph capture")
            if rows:
                from rtp_llm.models_py.modules.dsv4.fp8._v41_attention_triton import run

                run(
                    query,
                    sinks,
                    swa,
                    swa_slots,
                    valid_rows,
                    global_cache,
                    global_slots,
                    out,
                )
                self._warmed.add(key)
        return out

    @staticmethod
    def _eager(query, sinks, pools, valid_rows, out):
        if not bool(torch.isfinite(sinks).all()) or not bool(
            torch.isfinite(query[valid_rows]).all()
        ):
            raise ValueError("V4.1 live queries and sinks must be finite")
        for cache, slots, _ in pools:
            active = slots[valid_rows]
            if bool(
                (
                    (active < -1)
                    | (active >= cache.raw.shape[0] * cache.layout.entries)
                ).any()
            ):
                raise ValueError("V4.1 attention slot outside cache")
        # Match the BF16 probability contraction used by the checkpoint's
        # reference kernel. State and the shared softmax denominator stay FP32.
        for row in range(query.shape[0]):
            if not bool(valid_rows[row]):
                out[row].zero_()
                continue
            q = query[row].float()
            maximum = torch.full_like(sinks, -torch.inf)
            total = torch.zeros_like(sinks)
            value = torch.zeros_like(q)
            for cache, slots, _ in pools:
                for begin in range(0, slots.shape[1], _TILE):
                    selected = slots[row, begin : begin + _TILE].contiguous()
                    kv = cache.read(selected, backend="eager").float()
                    scores = (q @ kv.T) * _SCALE
                    scores.masked_fill_((selected < 0)[None, :], -torch.inf)
                    next_max = torch.maximum(maximum, scores.amax(-1))
                    finite_max = torch.where(torch.isneginf(next_max), 0, next_max)
                    alpha = (maximum - finite_max).exp()
                    probabilities = (scores - finite_max[:, None]).exp()
                    value = (
                        value * alpha[:, None]
                        + probabilities.to(torch.bfloat16).float() @ kv
                    )
                    total = total * alpha + probabilities.sum(-1)
                    maximum = next_max
            next_max = torch.maximum(maximum, sinks)
            alpha = (maximum - next_max).exp()
            denominator = total * alpha + (sinks - next_max).exp()
            out[row].copy_(value * (alpha / denominator)[:, None])
