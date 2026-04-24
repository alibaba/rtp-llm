# -*- coding: utf-8 -*-
# AMD backend CI test for chunk-gdn operator.
# Verifies end-to-end correctness with fwd_h dispatching to the Gluon kernel
# on AMD CDNA4 (MI355X/gfx950).  H<=32 and T<=65536 ensures Gluon dispatch.

import logging
import os
from typing import List

import torch
import torch.nn.functional as F

from rtp_llm.models_py.triton_kernels.fla.chunk import chunk_gated_delta_rule
from rtp_llm.models_py.triton_kernels.fla.utils import assert_close, device

logging.basicConfig(
    level="INFO",
    format="[%(name)s][%(asctime)s][%(filename)s:%(lineno)s][%(levelname)s] %(message)s",
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
    """Element-wise recurrent reference (float32)."""
    q, k, v, beta, g = map(
        lambda x: x.transpose(1, 2).contiguous().to(torch.float32), [q, k, v, beta, g]
    )
    B, H, T, K, V = *k.shape, v.shape[-1]
    o = torch.zeros(B, H, T, V).to(v)
    h = torch.zeros(B, H, K, V).to(v)
    if initial_state is not None:
        h = initial_state.to(torch.float32)
    if scale is None:
        scale = 1 / (q.shape[-1] ** 0.5)
    q = q * scale
    for i in range(T):
        b_q = q[:, :, i]
        b_k = k[:, :, i]
        b_v = v[:, :, i].clone()
        h = h.clone() * g[:, :, i].exp()[..., None, None]
        b_beta = beta[:, :, i]
        b_v = b_v - (h.clone() * b_k[..., None]).sum(-2)
        b_v = b_v * b_beta[..., None]
        h = h.clone() + b_k.unsqueeze(-1) * b_v.unsqueeze(-2)
        o[:, :, i] = torch.einsum("bhd,bhdm->bhm", b_q, h)
    if not output_final_state:
        h = None
    o = o.transpose(1, 2).contiguous()
    return o, h


def test_chunk_gdn_gluon(
    B: int,
    T: int,
    H: int,
    D: int,
    dtype: torch.dtype,
):
    """Test chunk-gdn with params that trigger Gluon fwd_h dispatch on AMD.

    Gluon dispatch requires: H<=32, T<=65536, gk=None, GLUON_AVAILABLE, hip backend.
    """
    torch.manual_seed(42)
    os.environ["TRITON_F32_DEFAULT"] = "ieee"

    q = torch.randn(B, T, H, D, dtype=dtype)
    k = F.normalize(torch.randn(B, T, H, D, dtype=torch.float32), p=2, dim=-1).to(dtype)
    v = torch.randn(B, T, H, D, dtype=dtype)
    g = F.logsigmoid(torch.rand(B, T, H, dtype=dtype))
    beta = torch.rand(B, T, H, dtype=dtype).sigmoid()
    h0 = torch.randn(B, H, D, D, dtype=dtype)

    q, k, v, beta, g, h0 = map(
        lambda x: x.to(device).requires_grad_(), (q, k, v, beta, g, h0)
    )

    tri_o, _, tri_ht = chunk_gated_delta_rule(
        q=q.clone(), k=k.clone(), v=v.clone(),
        beta=beta.clone(), g=g.clone(),
        initial_state=h0.clone(),
        output_final_state=True,
    )

    ref_o, ref_ht = recurrent_gated_delta_rule_ref(
        q=q.clone(), k=k.clone(), v=v.clone(),
        beta=beta.clone(), g=g.clone(),
        initial_state=h0.clone(),
        output_final_state=True,
    )

    assert_close("o", ref_o, tri_o, 0.005)
    assert_close("ht", ref_ht, tri_ht, 0.005)


def test_chunk_gdn_gluon_varlen(
    H: int,
    D: int,
    cu_seqlens: List[int],
    dtype: torch.dtype,
):
    """Test chunk-gdn varlen with Gluon fwd_h on AMD."""
    torch.manual_seed(42)
    os.environ["TRITON_F32_DEFAULT"] = "ieee"

    cu_seqlens_t = torch.LongTensor(cu_seqlens).to(device)
    T = cu_seqlens_t[-1]
    N = len(cu_seqlens_t) - 1

    q = torch.randn((1, T, H, D), dtype=dtype)
    k = F.normalize(torch.randn(1, T, H, D, dtype=torch.float32), p=2, dim=-1).to(dtype)
    v = torch.randn((1, T, H, D), dtype=dtype)
    g = F.logsigmoid(torch.rand(1, T, H, dtype=dtype))
    beta = torch.rand(1, T, H, dtype=dtype).sigmoid()
    h0 = torch.randn((N, H, D, D), dtype=dtype)

    q, k, v, beta, g, h0 = map(
        lambda x: x.to(device).requires_grad_(), (q, k, v, beta, g, h0)
    )

    tri, _, tri_ht = chunk_gated_delta_rule(
        q=q.clone(), k=k.clone(), v=v.clone(),
        beta=beta.clone(), g=g.clone(),
        initial_state=h0.clone(),
        output_final_state=True,
        cu_seqlens=cu_seqlens_t,
    )

    ref_parts, ref_ht_parts = [], []
    for i in range(N):
        ref_i, ref_ht_i = recurrent_gated_delta_rule_ref(
            q=q[:, cu_seqlens_t[i]:cu_seqlens_t[i + 1]],
            k=k[:, cu_seqlens_t[i]:cu_seqlens_t[i + 1]],
            v=v[:, cu_seqlens_t[i]:cu_seqlens_t[i + 1]],
            beta=beta[:, cu_seqlens_t[i]:cu_seqlens_t[i + 1]],
            g=g[:, cu_seqlens_t[i]:cu_seqlens_t[i + 1]],
            initial_state=h0[i],
            output_final_state=True,
        )
        ref_parts.append(ref_i)
        ref_ht_parts.append(ref_ht_i)
    ref = torch.cat(ref_parts, 1)
    ref_ht = torch.cat(ref_ht_parts, 0)

    assert_close("o", ref, tri, 0.005)
    assert_close("ht", ref_ht, tri_ht, 0.005)


if __name__ == "__main__":
    # H<=32 ensures Gluon dispatch on AMD CDNA4
    # Include T%64!=0 cases to verify last_idx clamp fix
    equal_length_params = [
        # (B, T, H, D, dtype)
        (2, 128, 4, 64, torch.bfloat16),
        (1, 100, 4, 64, torch.bfloat16),    # T%BT!=0
        (1, 128, 4, 128, torch.bfloat16),    # K=128 (two 64-wide tiles)
    ]
    for params in equal_length_params:
        B, T, H, D, dtype = params
        logging.info(f"Testing Gluon equal-length: B={B}, T={T}, H={H}, D={D}")
        test_chunk_gdn_gluon(B, T, H, D, dtype)

    varlen_params = [
        # (H, D, cu_seqlens, dtype)
        (4, 64, [0, 100, 200], torch.bfloat16),     # T%BT!=0
        (4, 64, [0, 256, 500, 1000], torch.bfloat16),
    ]
    for params in varlen_params:
        H, D, cu_seqlens, dtype = params
        logging.info(f"Testing Gluon varlen: H={H}, D={D}, cu_seqlens={cu_seqlens}")
        test_chunk_gdn_gluon_varlen(H, D, cu_seqlens, dtype)

    logging.info("All AMD chunk-gdn Gluon tests passed!")
