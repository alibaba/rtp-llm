"""RocmFp4DequantStrategy: BF16-dequantized routed experts for AMD gfx942.

DeepGEMM's FP8xFP4 grouped kernels are SM100-only, and ROCm has no native
FP4 grouped-GEMM path in-tree (the factory MXFP4 strategy requires gfx950).
This strategy dequantizes the packed E2M1 expert weights (per-32 UE8M0
scales) once at weight-bind time into BF16 stacks: an E2M1 value times a
power-of-two scale is exactly representable in BF16, so the checkpoint's
quantized semantics are preserved bit-exactly (guarded at load). Forward
then mirrors ``LocalLoopStrategy``'s structure — eager per-expert gather
outside graph capture, fixed-shape masked compute during capture — using
plain BF16 GEMMs and the portable fused SiLU+clamp+mul Triton kernel, so
router-weight and swiglu-clamp semantics are identical to the SM100 path.

Memory cost: 4x the packed FP4 footprint (~136 GiB/GPU at TP8 for
V4.1-Flash-1017), which fits MI308X (192 GiB) with room for full-context
KV. A later optimization can consume the packed weights directly (Triton
in-kernel-dequant grouped GEMM, or aiter fused_moe once a clamp-correct
activation exists) to cut both memory and decode weight bandwidth.
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


def _dequant_fp4_stack(
    packed: torch.Tensor, scales: torch.Tensor, chunk: int = 16
) -> torch.Tensor:
    """``[E, N, K/2]`` packed E2M1 + ``[E, N, K/32]`` UE8M0 -> BF16 [E, N, K].

    Nibble order matches ``qlinear._fp4_unpack_to_fp32``: the low nibble is
    the even K index, the high nibble the odd one. Products of an E2M1 value
    (<=3 significand bits) and a power-of-two scale are exact in BF16; the
    finiteness/underflow guard makes any violation fail at load time.
    """
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
    lut = _FP4_LUT.to(device=packed.device, dtype=torch.float32)
    out = torch.empty((e, n, k), dtype=torch.bfloat16, device=packed.device)
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
        out[start:stop] = vals.to(torch.bfloat16)
    return out


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
        """Pop ``W.v4_routed_w{1,2,3}_{w,s}`` and dequantize to BF16 stacks.

        The packed FP4/int8 + UE8M0 stacks are consumed here and released;
        the resident tensors are ``self._W{1,2,3}`` in BF16 (checkpoint
        provenance: w1=gate ``[E,inter,dim]``, w3=up ``[E,inter,dim]``,
        w2=down ``[E,dim,inter]``, each dequantized from the matching
        ``_{w,s}`` pair).
        """
        from rtp_llm.utils.model_weight import W

        sources = {
            1: (layer_weights.pop(W.v4_routed_w1_w), layer_weights.pop(W.v4_routed_w1_s)),
            2: (layer_weights.pop(W.v4_routed_w2_w), layer_weights.pop(W.v4_routed_w2_s)),
            3: (layer_weights.pop(W.v4_routed_w3_w), layer_weights.pop(W.v4_routed_w3_s)),
        }
        self._W1 = _dequant_fp4_stack(*sources[1])
        self._W2 = _dequant_fp4_stack(*sources[2])
        self._W3 = _dequant_fp4_stack(*sources[3])
        try:
            from rtp_llm.model_loader.awi_recorder import register_transform

            for index, resident in ((1, self._W1), (2, self._W2), (3, self._W3)):
                register_transform(
                    list(sources[index]),
                    [resident],
                    f"packed_e2m1_ue8m0_group32_to_bf16_stack_w{index}",
                )
        except Exception:  # noqa: BLE001
            pass
        # sp_v4_routed_w1/w2 shard the inter dim across TP ranks under
        # pure-TP MoE; detect it from the resident shape vs the global
        # config value and reduce partial sums in forward.
        self._tp_sharded = self._W1.shape[1] < self.cfg.moe_inter_dim
        self._local_y_buf: Optional[torch.Tensor] = None

    def get_eplb_weights(self) -> tuple[torch.Tensor, ...]:
        raise NotImplementedError(
            "RocmFp4DequantStrategy does not support EPLB (weights are "
            "dequantized into BF16 stacks; the packed layout is not retained)"
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
        if torch.cuda.is_current_stream_capturing():
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
        """One expert's SwiGLU MLP, mirroring ``Expert.forward`` semantics."""
        gate = torch.mm(x_bf16, self._W1[row].t()).float()
        up = torch.mm(x_bf16, self._W3[row].t()).float()
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
        return torch.mm(sm.to(x_bf16.dtype), self._W2[row].t())

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

    def _forward_graph_safe(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Fixed-shape masked compute (no data-dependent ``torch.where``)."""
        start, end = self.cfg.local_expert_start, self.cfg.local_expert_end
        for i in range(start, end):
            mask = (indices == i).to(weights.dtype)
            per_token_w = (weights * mask).sum(dim=-1, keepdim=True)
            y.add_(self._expert_out(i - start, x, per_token_w).float())
        return y
