"""Model/stream-owned scratch for the three in-repository Engram CUDA kernels."""

from __future__ import annotations

import torch
from rtp_llm.ops.compute_ops import rtp_llm_ops


def available():
    return all(
        hasattr(rtp_llm_ops, name)
        for name in ("engram_wkv_post", "engram_gate_update", "engram_pre")
    )


class State:
    def __init__(
        self,
        weight,
        weight_sf,
        fn,
        scales,
        bases,
        rmsweight,
        *,
        rows,
        gate_eps,
        hc_norm_eps,
        hc_eps,
        sinkhorn_iters,
        rmsnorm_eps,
    ):
        self.weight, self.weight_sf = weight, weight_sf
        self.fn, self.scales, self.bases, self.rmsweight = fn, scales, bases, rmsweight
        self.m, self.device = rows, weight.device
        self.stream = torch.cuda.current_stream(self.device).cuda_stream
        self.eps, self.hc_norm_eps, self.hc_eps = gate_eps, hc_norm_eps, hc_eps
        self.iters, self.rmsnorm_eps = sinkhorn_iters, rmsnorm_eps
        self.kv = torch.empty((rows, 25600), device=self.device, dtype=torch.bfloat16)
        self.hp = torch.empty((rows, 4, 5120), device=self.device, dtype=torch.bfloat16)
        self.mask = torch.ones(rows, device=self.device, dtype=torch.bool)
        splits = min(148 // ((rows + 63) // 64), 80)
        self.partial = torch.empty(
            (splits, rows, 24), device=self.device, dtype=torch.float32
        )
        self.sqr = torch.empty((splits, rows), device=self.device, dtype=torch.float32)
        # Per-stream epochs survive graph replay. Each CTA joins all stores
        # before publishing its split completion to the row owner.
        self.protocol = torch.zeros(156, device=self.device, dtype=torch.int64)

    def forward(self, a, sf, x, residual, post, comb, shifted_mix, qw, kw, mask):
        if torch.cuda.current_stream(self.device).cuda_stream != self.stream:
            raise RuntimeError("Engram transition state belongs to another CUDA stream")
        rtp_llm_ops.engram_wkv_post(
            a,
            sf,
            self.weight,
            self.weight_sf,
            x,
            residual,
            post,
            comb,
            self.kv,
            self.hp,
        )
        # Outputs are forward-owned; scratch must not escape into delayed HC
        # state, eager callers, or another captured graph's output allocation.
        h = torch.empty_like(self.hp)
        rtp_llm_ops.engram_gate_update(
            self.hp, self.kv, qw, kw, self.mask if mask is None else mask, h, self.eps
        )
        m = self.m
        y = torch.empty((m, 5120), device=self.device, dtype=torch.bfloat16)
        new_post = torch.empty((m, 4, 1), device=self.device, dtype=torch.float32)
        new_comb = torch.empty((m, 4, 4), device=self.device, dtype=torch.float32)
        pre = torch.empty_like(new_post)
        q = torch.empty_like(y, dtype=torch.float8_e4m3fn)
        out_sf = torch.empty(
            (40, (m + 3) // 4 * 4), device=self.device, dtype=torch.int32
        ).T[:m]
        rtp_llm_ops.engram_pre(
            h,
            shifted_mix,
            self.fn,
            self.scales,
            self.bases,
            self.rmsweight,
            self.partial,
            self.sqr,
            self.protocol,
            pre,
            new_post,
            new_comb,
            y,
            q,
            out_sf,
            self.hc_norm_eps,
            self.hc_eps,
            self.iters,
            self.rmsnorm_eps,
        )
        return h, y, new_post, new_comb, pre, q, out_sf
