"""RocmFp4DequantStrategy: BF16-dequantized routed experts for AMD gfx942.

DeepGEMM's FP8xFP4 grouped kernels are SM100-only, and ROCm has no native
FP4 grouped-GEMM path in-tree (the factory MXFP4 strategy requires gfx950).
The packed E2M1 expert weights (per-32 UE8M0 scales) stay RESIDENT in
their packed form; BF16 dequantization happens transiently per use (eager:
per routed expert; graph-safe: per 96-expert bmm chunk). An E2M1 value
times a power-of-two scale is exactly representable in BF16, so the
checkpoint's quantized semantics are preserved bit-exactly (the whole
stack is still guard-validated once at weight-bind time). Forward then
mirrors ``LocalLoopStrategy``'s structure — eager per-expert gather
outside graph capture, fixed-shape masked compute during capture — using
plain BF16 GEMMs and the portable fused SiLU+clamp+mul Triton kernel, so
router-weight and swiglu-clamp semantics are identical to the SM100 path.

Memory cost: the packed FP4 footprint (~34 GiB/GPU at TP8 for
V4.1-Flash-1017) plus per-use transient BF16 (<= a few GiB, allocator-
reused; inside captures it lives in the graph private pool). The previous
persistent-BF16-stack design (~136 GiB/GPU) left only ~8 GiB free at
KV_CACHE_MEM_MB=15300, and the first genuine 1M-token prefill died
allocating the 10.00 GiB [T,5120] hidden state (n11, 2026-11-11:
torch.OutOfMemoryError with 175.72 GiB allocated / 8.09 GiB free). A
later optimization can consume the packed weights directly (Triton
in-kernel-dequant grouped GEMM, or aiter fused_moe once a clamp-correct
activation exists) to also cut decode weight bandwidth.
"""

from __future__ import annotations

import functools
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from ...qlinear import _FP4_LUT

try:  # portable fused SiLU + clamp + mul (same kernel Expert.forward uses)
    from ..._silu_mul_split_triton import silu_mul_split
except Exception:  # pragma: no cover - keep importable without Triton
    silu_mul_split = None

from .base import MoeCfg, RoutedExpertsStrategy, register_strategy

# NB: do not import .local_loop at module scope — its import side effect
# registers LocalLoopStrategy, and strategies/__init__.py relies on import
# order for priority (this module must register before local_loop lands).

_FP4_BLOCK = 32


def _check_fp4_pair(packed: torch.Tensor, scales: torch.Tensor):
    """Validate a packed E2M1 + UE8M0 pair; return ``(e, n, k)``."""
    if packed.dtype != torch.int8:
        raise TypeError(f"expected packed E2M1 int8, got {packed.dtype}")
    if scales.dtype != torch.float8_e8m0fnu:
        raise TypeError(f"expected UE8M0 scales, got {scales.dtype}")
    e, n, half_k = packed.shape
    k = half_k * 2
    if scales.shape != (e, n, k // _FP4_BLOCK):
        raise ValueError(
            f"FP4 scale shape {tuple(scales.shape)} does not tile "
            f"{(e, n, k)} at block {_FP4_BLOCK}"
        )
    return e, n, k


def _dequant_fp4_rows(
    packed: torch.Tensor, scales: torch.Tensor, lo: int, hi: int
) -> torch.Tensor:
    """Rows ``[lo, hi)`` of ``[E, N, K/2]`` packed E2M1 + ``[E, N, K/32]``
    UE8M0 -> BF16 ``(hi-lo, N, K)``.

    Nibble order matches ``qlinear._fp4_unpack_to_fp32``: the low nibble is
    the even K index, the high nibble the odd one. Products of an E2M1 value
    (<=3 significand bits) and a power-of-two scale are exact in BF16.

    Guard-free and sync-free on purpose: this runs inside eager forwards AND
    inside HIP graph capture/replay (per bmm chunk), where ``.all()``/
    ``bool()`` device reads are illegal. The whole-stack finiteness/underflow
    guard runs once at weight-bind time via :func:`_validate_fp4_stack`.
    """
    lut = _FP4_LUT.to(device=packed.device, dtype=torch.float32)
    e, n, k = _check_fp4_pair(packed, scales)
    hi = min(hi, e)
    out = torch.empty((hi - lo, n, k), dtype=torch.bfloat16, device=packed.device)
    w_uint = packed[lo:hi].to(torch.int32) & 0xFF
    lo_v = lut[(w_uint & 0x0F).long()]
    hi_v = lut[((w_uint >> 4) & 0x0F).long()]
    sc = scales[lo:hi].float().repeat_interleave(_FP4_BLOCK, dim=-1)
    out[..., 0::2] = (lo_v * sc[..., 0::2]).to(torch.bfloat16)
    out[..., 1::2] = (hi_v * sc[..., 1::2]).to(torch.bfloat16)
    return out


def _validate_fp4_stack(
    packed: torch.Tensor, scales: torch.Tensor, chunk: int = 16
) -> None:
    """Load-time guard: chunkwise dequant + finiteness/underflow checks.

    Results are discarded — the packed stack is what stays resident; this
    preserves the historical fail-at-load contract (any violation raises
    OverflowError) without paying the 4x BF16 memory.
    """
    e, n, k = _check_fp4_pair(packed, scales)
    lut = _FP4_LUT.to(device=packed.device, dtype=torch.float32)
    for start in range(0, e, chunk):
        stop = min(start + chunk, e)
        w_uint = packed[start:stop].to(torch.int32) & 0xFF
        lo_v = lut[(w_uint & 0x0F).long()]
        hi_v = lut[((w_uint >> 4) & 0x0F).long()]
        sc = scales[start:stop].float().repeat_interleave(_FP4_BLOCK, dim=-1)
        vals = torch.empty(
            (stop - start, n, k), dtype=torch.float32, device=packed.device
        )
        vals[..., 0::2] = lo_v * sc[..., 0::2]
        vals[..., 1::2] = hi_v * sc[..., 1::2]
        if not torch.isfinite(vals).all():
            raise OverflowError("FP4 expert dequant left the BF16 range")
        nz_lo = (vals[..., 0::2] == 0) & (lo_v != 0)
        nz_hi = (vals[..., 1::2] == 0) & (hi_v != 0)
        if bool(nz_lo.any()) or bool(nz_hi.any()):
            raise OverflowError("FP4 expert dequant underflowed nonzero weights")
        del vals


@functools.cache
def _has_record_expert_load() -> bool:
    """EPLB load telemetry op; registered on CUDA builds only."""
    from rtp_llm.ops.compute_ops import rtp_llm_ops

    return hasattr(rtp_llm_ops, "record_expert_load")


@register_strategy
class RocmFp4DequantStrategy(RoutedExpertsStrategy):
    name = "rocm_fp4_dequant"

    # True when the loader TP-sharded the expert inter dim (pure-TP MoE);
    # per-rank outputs are then partial sums requiring a TP all-reduce.
    _tp_sharded = False

    @classmethod
    def can_handle(cls, cfg: MoeCfg) -> bool:
        return torch.version.hip is not None and cfg.ep_size <= 1

    def setup_weights(self, layer_weights: Dict) -> None:
        """Pop ``W.v4_routed_w{1,2,3}_{w,s}``; keep the packed FP4 resident.

        The packed int8/E2M1 + UE8M0 scale stacks ARE the resident device
        tensors (``self._P{1,2,3}`` / ``self._S{1,2,3}``; checkpoint
        provenance: w1=gate ``[E,inter,dim]``, w3=up ``[E,inter,dim]``,
        w2=down ``[E,dim,inter]``). BF16 materialization is transient and
        per-use (:func:`_dequant_fp4_rows`), mirroring how Engram rows are
        staged for computation. The full-stack finiteness/underflow guard
        still runs once here, chunkwise, so violations fail at load time.
        """
        from rtp_llm.utils.model_weight import W

        sources = {
            1: (layer_weights.pop(W.v4_routed_w1_w), layer_weights.pop(W.v4_routed_w1_s)),
            2: (layer_weights.pop(W.v4_routed_w2_w), layer_weights.pop(W.v4_routed_w2_s)),
            3: (layer_weights.pop(W.v4_routed_w3_w), layer_weights.pop(W.v4_routed_w3_s)),
        }
        self._P1, self._S1 = sources[1]
        self._P2, self._S2 = sources[2]
        self._P3, self._S3 = sources[3]
        _validate_fp4_stack(self._P1, self._S1)
        _validate_fp4_stack(self._P2, self._S2)
        _validate_fp4_stack(self._P3, self._S3)
        try:
            from rtp_llm.model_loader.awi_recorder import register_transform

            for index, (packed, scales) in (
                (1, sources[1]),
                (2, sources[2]),
                (3, sources[3]),
            ):
                register_transform(
                    [packed, scales],
                    [packed, scales],
                    f"packed_e2m1_ue8m0_group32_resident_bf16_dequant_on_use_w{index}",
                )
        except Exception:  # noqa: BLE001
            pass
        # sp_v4_routed_w1/w2 shard the inter dim across TP ranks under
        # pure-TP MoE; detect it from the resident packed shape (N axis) vs
        # the global config value and reduce partial sums in forward.
        self._tp_sharded = self._P1.shape[1] < self.cfg.moe_inter_dim
        self._local_y_buf: Optional[torch.Tensor] = None

    def get_eplb_weights(self) -> tuple[torch.Tensor, ...]:
        raise NotImplementedError(
            "RocmFp4DequantStrategy does not support EPLB (the packed FP4 "
            "layout is retained but no EPLB remapping is wired for it)"
        )

    def forward(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        expert_load=None,
        eplb_layer=None,
    ) -> torch.Tensor:
        self.finalize_routing(indices, weights, eplb_layer)
        if (
            expert_load is not None
            and self.cfg.ep_size == 1
            and _has_record_expert_load()
        ):
            expert_load.record_ids(indices)
        from .local_loop import _get_or_create_local_y

        t = x.size(0)
        self._local_y_buf = _get_or_create_local_y(
            max(t, self.cfg.max_tokens_per_rank),
            self.cfg.dim,
            torch.float32,
            x.device,
        )
        y = self._local_y_buf[:t]
        y.zero_()
        x_2d = x.reshape(t, -1)
        if x_2d.dtype != torch.bfloat16:
            x_2d = x_2d.to(torch.bfloat16)
        from ..warmup_sync import cuda_graph_warmup_forward_enabled

        if (
            torch.cuda.is_current_stream_capturing()
            or cuda_graph_warmup_forward_enabled()
        ):
            # The C++ capture harness sets RTP_LLM_CUDA_GRAPH_WARMUP_FORWARD=1
            # around the two eager warmup forwards that precede every capture
            # (cuda_graph_runner.cc). Those warmups MUST run the exact branch
            # the capture will record: keying only on
            # is_current_stream_capturing() left every graph-safe-only kernel
            # (fixed-shape masks, full-batch Triton launches, hipBLASLt
            # solutions for padded shapes) cold during capture, which records
            # a sticky hipErrorInvalidConfiguration in EVERY layer's MoE
            # (load32 capture probes: ffn0..ffn60, all ranks, capture-only).
            self._forward_graph_safe(x_2d, weights, indices, y)
        else:
            self._forward_eager(x_2d, weights, indices, y)
        if self._tp_sharded:
            from rtp_llm.models_py.distributed.collective_torch import (
                Group,
                all_reduce,
            )

            all_reduce(y, Group.TP, inplace=True)
        return y

    def _expert_out(
        self, row: int, x_bf16: torch.Tensor, router_w: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """One expert's SwiGLU MLP, mirroring ``Expert.forward`` semantics.

        The expert's BF16 rows are dequantized transiently (3 small tensors,
        freed on return) instead of read from a persistent stack.
        """
        w1 = _dequant_fp4_rows(self._P1, self._S1, row, row + 1)[0]
        w3 = _dequant_fp4_rows(self._P3, self._S3, row, row + 1)[0]
        gate = torch.mm(x_bf16, w1.t()).float()
        up = torch.mm(x_bf16, w3.t()).float()
        del w1, w3
        if silu_mul_split is not None:
            sm = silu_mul_split(
                gate.contiguous(),
                up.contiguous(),
                clamp_limit=self.cfg.swiglu_limit,
            )
        else:
            limit = self.cfg.swiglu_limit
            if limit > 0:
                up = torch.clamp(up, min=-limit, max=limit)
                gate = torch.clamp(gate, max=limit)
            sm = F.silu(gate) * up
        if router_w is not None:
            sm = sm * router_w
        w2 = _dequant_fp4_rows(self._P2, self._S2, row, row + 1)[0]
        out = torch.mm(sm.to(x_bf16.dtype), w2.t())
        del w2
        return out

    def _forward_eager(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        start, end = self.cfg.local_expert_start, self.cfg.local_expert_end
        for i in range(start, end):
            idx, top = torch.where(indices == i)
            if idx.numel() == 0:
                continue
            out = self._expert_out(i - start, x[idx], weights[idx, top, None])
            y[idx] = y[idx] + out.float()
        return y

    # Experts per batched-GEMM chunk inside captured forwards: bounds the
    # (chunk, t, dim) fp32 temporary (~94MB at chunk=96, t=48) while keeping
    # the recorded node count tiny.
    _GRAPH_SAFE_EXPERT_CHUNK = 96

    def _forward_graph_safe(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Fixed-shape dense-batched compute (no data-dependent shapes).

        Batched over experts in chunks: the previous per-expert python loop
        recorded ~384 experts x ~7 launches per layer (~110k graph nodes over
        40 layers), and the backend process SEGVed during recording/
        instantiation on ROCm (loads 34-36: probe traces reached the end of
        the forward on every rank, then the process died before capture_end
        returned). This keeps the identical math - dense per-(token,expert)
        routing weights, full dense compute, weighted reduction - as chunked
        batched GEMMs (~40 nodes per layer), which also replays far faster.
        """
        start, end = self.cfg.local_expert_start, self.cfg.local_expert_end
        E = end - start
        t = x.size(0)
        limit = self.cfg.swiglu_limit

        # Dense routing weights (t, E): scatter the top-k weights into their
        # expert columns; mirrors (weights * (indices == i)).sum(-1) per i.
        loc = indices.to(torch.int64) - start
        valid = (loc >= 0) & (loc < E)
        w_dense = torch.zeros((t, E), dtype=torch.float32, device=x.device)
        w_dense.scatter_add_(
            1,
            loc.clamp(min=0, max=E - 1),
            (weights * valid).to(torch.float32),
        )

        x_bf16 = x if x.dtype == torch.bfloat16 else x.to(torch.bfloat16)
        xb = x_bf16.unsqueeze(0)  # (1, t, dim), broadcast across expert batch
        chunk = self._GRAPH_SAFE_EXPERT_CHUNK
        for c0 in range(0, E, chunk):
            c1 = min(c0 + chunk, E)
            # Per-chunk transient BF16 dequant of the resident packed FP4
            # (capture-legal: guard-free, sync-free; freed blocks recycle
            # through the graph private pool on the next chunk/replay).
            W1 = _dequant_fp4_rows(self._P1, self._S1, start + c0, start + c1)
            W3 = _dequant_fp4_rows(self._P3, self._S3, start + c0, start + c1)
            xc = xb.expand(c1 - c0, t, x_bf16.shape[-1])
            gate = torch.bmm(xc, W1.transpose(1, 2)).float()
            up = torch.bmm(xc, W3.transpose(1, 2)).float()
            del W1, W3
            if silu_mul_split is not None:
                sm = silu_mul_split(
                    gate.reshape(-1, gate.shape[-1]).contiguous(),
                    up.reshape(-1, up.shape[-1]).contiguous(),
                    clamp_limit=limit,
                ).view(gate.shape)
            else:
                if limit > 0:
                    up = torch.clamp(up, min=-limit, max=limit)
                    gate = torch.clamp(gate, max=limit)
                sm = F.silu(gate) * up
            sm = sm * w_dense[:, c0:c1].t().unsqueeze(-1)  # (e, t, 1)
            W2 = _dequant_fp4_rows(self._P2, self._S2, start + c0, start + c1)
            out = torch.bmm(sm.to(x_bf16.dtype), W2.transpose(1, 2))
            del W2
            y.add_(out.float().sum(dim=0))
        return y
