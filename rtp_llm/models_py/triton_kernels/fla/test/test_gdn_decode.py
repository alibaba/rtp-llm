# -*- coding: utf-8 -*-

import logging
import math
import os
import random
from typing import List

import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from rtp_llm.models_py.triton_kernels.fla import fused_recurrent_gated_delta_rule
from rtp_llm.models_py.triton_kernels.fla.utils import assert_close

logging.basicConfig(
    level="INFO",
    format="[process-%(process)d][%(name)s][%(asctime)s.%(msecs)03d][%(filename)s:%(funcName)s():%(lineno)s][%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)


def recurrent_gated_delta_rule_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
):
    q, k, v, beta, g = map(
        lambda x: x.transpose(1, 2).contiguous().to(torch.float32), [q, k, v, beta, g]
    )
    B, H, T, K, V = *k.shape, v.shape[-1]
    o = torch.zeros(B, H, T, V).to(v)
    h = torch.zeros(B, H, K, V).to(v)
    if initial_state is not None:
        h = initial_state
    if scale is None:
        scale = 1 / (q.shape[-1] ** 0.5)
    q = q * scale
    hs = []
    for i in range(T):
        b_q = q[:, :, i]
        b_k = k[:, :, i]
        b_v = v[:, :, i].clone()
        h = h.clone() * g[:, :, i].exp()[..., None, None]
        b_beta = beta[:, :, i]
        b_v = b_v - (h.clone() * b_k[..., None]).sum(-2)
        b_v = b_v * b_beta[..., None]
        h = h.clone() + b_k.unsqueeze(-1) * b_v.unsqueeze(-2)
        hs.append(h.detach().clone())
        o[:, :, i] = torch.einsum("bhd,bhdm->bhm", b_q, h)
    if not output_final_state:
        hs = None
    else:
        hs = torch.stack(hs, dim=1)
    o = o.transpose(1, 2).contiguous()
    return o, hs


def test_fused_recurrent_continuous_batching(
    B: int,
    S: int,
    H: int,
    HV: int,
    D: int,
    scale: float,
    gate_logit_normalizer: float,
    dtype: torch.dtype,
):
    device = "cuda"
    torch.manual_seed(42)
    q = torch.randn(B, S, H, D, dtype=torch.float32, device=device)
    k = torch.randn(B, S, H, D, dtype=torch.float32, device=device)
    v = torch.randn(B, S, HV, D, dtype=dtype, device=device)
    beta = torch.rand(B, S, HV, dtype=dtype, device=device).sigmoid()
    g = F.logsigmoid(torch.rand(B, S, HV, dtype=torch.float32, device=device))
    g = g / gate_logit_normalizer
    h0 = torch.randn(B, HV, D, D, dtype=torch.float32, device=device)
    q, k, v, beta, g, h0 = map(
        lambda x: x.to(device).requires_grad_(), (q, k, v, beta, g, h0)
    )
    ref, ref_ht = recurrent_gated_delta_rule_ref(
        q=F.normalize(
            repeat(q.clone(), "b t h d -> b t (h g) d", g=HV // H), p=2, dim=-1
        ).to(dtype),
        k=F.normalize(
            repeat(k.clone(), "b t h d -> b t (h g) d", g=HV // H), p=2, dim=-1
        ).to(dtype),
        v=v.clone(),
        beta=beta.clone(),
        g=g.clone(),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
    )

    seq_size_per_block = 128
    sequence_lengths = [random.randint(10, 1024) for _ in range(B)]
    block_num = [
        math.ceil(sequence_lengths[i] / seq_size_per_block) + (S - 1) for i in range(B)
    ]
    total_block_num = sum(block_num) + 1
    block_map = torch.zeros([B, total_block_num], dtype=torch.int32)
    offset = 1
    for i in range(B):
        # start from 1 to avoid treated as padding batch in kernel
        block_map[i, : block_num[i]] = torch.arange(
            offset, offset + block_num[i], dtype=torch.int32
        )
        offset += block_num[i]
    block_map = block_map.to(device)
    load_block_offset = [(x - 2) // seq_size_per_block for x in sequence_lengths]
    ssm_cache = torch.zeros(
        total_block_num, HV, D, D, dtype=torch.float32, device=device
    )
    for bs in range(B):
        ssm_cache[int(block_map[bs, load_block_offset[bs]])] = h0[bs]

    sequence_lengths_t = torch.tensor(
        sequence_lengths, dtype=torch.int32, device=device
    )
    tri, _ = fused_recurrent_gated_delta_rule(
        q=q.clone().reshape(B, S, H, D),
        k=k.clone().reshape(B, S, H, D),
        v=v.clone().reshape(B, S, HV, D),
        beta=beta.clone().reshape(B, S, HV),
        g=g.clone().reshape(B, S, HV),
        scale=scale,
        initial_state=ssm_cache,
        block_map=block_map,
        sequence_lengths=sequence_lengths_t,
        seq_size_per_block=seq_size_per_block,
        use_qk_l2norm_in_kernel=True,
        inplace_final_state=True,
    )
    assert_close("o", ref.reshape(B, S, HV, D), tri.reshape(B, S, HV, D), 0.005)
    write_block_offset = [(x - 1) // seq_size_per_block for x in sequence_lengths]
    tri_ht = torch.zeros(B, S, HV, D, D, dtype=torch.float32, device=device)
    for bs in range(B):
        for seq in range(S):
            tri_ht[bs, seq] = ssm_cache[
                int(block_map[bs, write_block_offset[bs] + seq])
            ]
    assert_close("ht", ref_ht, tri_ht, 0.005)


if __name__ == "__main__":
    H = 16
    HV = 32
    D = 128
    scale = 1
    gate_logit_normalizer = 0.1
    for bs in [1, 2, 4, 8, 16, 32, 64]:
        for seq in [1, 2, 4]:
            logging.info(f"Testing with batch size: {bs}, sequence length: {seq}")
            test_fused_recurrent_continuous_batching(
                bs, seq, H, HV, D, scale, gate_logit_normalizer, torch.bfloat16
            )
