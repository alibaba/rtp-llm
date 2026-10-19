"""Kimi K3 routed MegaMoE buffer and output caches."""

from __future__ import annotations

import logging

import torch

_MEGA_K3_BUF_CACHE: dict[tuple, object] = {}
_MEGA_K3_OUTPUT_CACHE: dict[tuple, torch.Tensor] = {}


def get_or_create_kimi_k3_mega_moe_k3_buf(
    *,
    group,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    activation: str,
):
    key = (
        id(group),
        num_experts,
        num_max_tokens_per_rank,
        num_topk,
        hidden,
        intermediate_hidden,
        activation,
    )
    buf = _MEGA_K3_BUF_CACHE.get(key)
    if buf is None:
        import deep_gemm

        buf = deep_gemm.get_symm_buffer_for_mega_moe_k3(
            group=group,
            num_experts=num_experts,
            num_max_tokens_per_rank=num_max_tokens_per_rank,
            num_topk=num_topk,
            hidden=hidden,
            intermediate_hidden=intermediate_hidden,
            mma_type="fp8xfp4",
            activation=activation,
        )
        _MEGA_K3_BUF_CACHE[key] = buf
        logging.info(
            "[KimiK3 DeepGEMM MegaMoE K3] allocated buffer: experts=%d "
            "max_tokens_per_rank=%d aligned_capacity=%d topk=%d hidden=%d "
            "intermediate=%d size=%.3f MiB",
            num_experts,
            num_max_tokens_per_rank,
            int(buf.num_max_tokens_per_rank),
            num_topk,
            hidden,
            intermediate_hidden,
            buf.buffer.numel() * buf.buffer.element_size() / (1024**2),
        )
    return buf


def get_or_create_kimi_k3_mega_moe_k3_output(
    capacity: int, hidden: int, device: torch.device
) -> torch.Tensor:
    key = (device, hidden)
    output = _MEGA_K3_OUTPUT_CACHE.get(key)
    if output is None or output.size(0) < capacity:
        output = torch.empty(
            (max(capacity, 1), hidden), dtype=torch.bfloat16, device=device
        )
        _MEGA_K3_OUTPUT_CACHE[key] = output
    return output
