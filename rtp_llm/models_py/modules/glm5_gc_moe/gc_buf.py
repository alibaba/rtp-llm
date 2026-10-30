"""GC-owned symmetric workspaces, separate from every Mega MoE cache."""

from __future__ import annotations

import logging

import torch

_GC_BUFFERS: dict[tuple, object] = {}
_GC_CLONE_BUFFERS: dict[tuple, object] = {}
_GC_OUTPUTS: dict[tuple, torch.Tensor] = {}


def _create_gc_buffer(group, cfg, num_shared_experts: int):
    import gc_moe

    return gc_moe.get_symm_buffer_for_gc_moe(
        group=group,
        num_experts=cfg.n_routed_experts,
        num_max_tokens_per_rank=max(cfg.max_tokens_per_rank, 1),
        num_topk=cfg.n_activated_experts,
        hidden=cfg.dim,
        intermediate_hidden=cfg.moe_inter_dim,
        num_shared_experts=num_shared_experts,
        mma_type="fp8xfp4",
        activation="swiglu",
    )


def get_or_create_gc_buffer(group, cfg, num_shared_experts: int = 0):
    key = (
        id(group),
        cfg.n_routed_experts,
        cfg.max_tokens_per_rank,
        cfg.n_activated_experts,
        cfg.dim,
        cfg.moe_inter_dim,
        num_shared_experts,
    )
    buf = _GC_BUFFERS.get(key)
    if buf is None:
        buf = _create_gc_buffer(group, cfg, num_shared_experts)
        _GC_BUFFERS[key] = buf
        logging.info(
            "[GLM5 GC MoE] allocated symmetric buffer: layer=%d "
            "requested_capacity=%d actual_capacity=%d shared=%d bytes=%d",
            cfg.layer_id,
            cfg.max_tokens_per_rank,
            buf.num_max_tokens_per_rank,
            num_shared_experts,
            buf.buffer.nbytes,
        )
    return buf


def get_or_create_gc_clone_buffer(src_buf, group, cfg, num_shared_experts: int = 0):
    if src_buf is None or group is None:
        return src_buf
    key = (id(src_buf), id(group), num_shared_experts)
    buf = _GC_CLONE_BUFFERS.get(key)
    if buf is None:
        buf = _create_gc_buffer(group, cfg, num_shared_experts)
        _GC_CLONE_BUFFERS[key] = buf
        logging.info(
            "[GLM5 GC MoE] allocated CUDA graph clone buffer: layer=%d "
            "capacity=%d shared=%d",
            cfg.layer_id,
            buf.num_max_tokens_per_rank,
            num_shared_experts,
        )
    return buf


def get_or_create_gc_output(capacity: int, hidden: int, device: torch.device):
    key = (device, hidden)
    cached = _GC_OUTPUTS.get(key)
    if cached is None or cached.size(0) < capacity:
        cached = torch.empty(
            (max(capacity, 1), hidden), dtype=torch.bfloat16, device=device
        )
        _GC_OUTPUTS[key] = cached
    return cached
