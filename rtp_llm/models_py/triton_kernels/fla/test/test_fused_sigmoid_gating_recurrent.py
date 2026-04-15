# -*- coding: utf-8 -*-

"""
Test script for fused_sigmoid_gating_delta_rule_update kernel.

Tests correctness by comparing against a naive Python reference implementation.
Covers various batch sizes including large batches that exercise the HIP
grid-size fix (N*HV > 65535).

Usage:
    python -m rtp_llm.models_py.triton_kernels.fla.test_fused_sigmoid_gating_recurrent
"""

import logging
import math

import torch

from rtp_llm.models_py.triton_kernels.fla.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
)
from rtp_llm.models_py.triton_kernels.fla.utils import assert_close

logging.basicConfig(
    level="INFO",
    format="[process-%(process)d][%(name)s][%(asctime)s.%(msecs)03d]"
    "[%(filename)s:%(funcName)s():%(lineno)s][%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)


def fused_sigmoid_gating_delta_rule_ref(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
    use_qk_l2norm: bool = False,
):
    """
    Naive Python reference implementation for correctness verification.
    Simple per-sequence recurrence without block_map / continuous batching.
    """
    B, T, H, K = q.shape
    HV = v.shape[2]
    V = v.shape[3]

    h = initial_state.clone().float()
    outputs = torch.zeros(B, T, HV, V, dtype=torch.float32, device=v.device)
    heads_per_group = HV // H

    for n in range(B):
        for t in range(T):
            for hv in range(HV):
                ih = hv // heads_per_group

                b_q = q[n, t, ih].float()
                b_k = k[n, t, ih].float()
                b_v = v[n, t, hv].float()
                b_b = b[n, t, hv].float()
                b_a = a[n, t, hv].float()

                # Gating: g = -exp(A_log) * softplus(a + dt_bias)
                x = b_a + dt_bias[hv].float()
                beta_x = softplus_beta * x
                if beta_x <= softplus_threshold:
                    softplus_x = (1.0 / softplus_beta) * math.log(
                        1.0 + math.exp(beta_x.item())
                    )
                else:
                    softplus_x = x.item()
                g = -math.exp(A_log[hv].item()) * softplus_x

                # beta = sigmoid(b)
                beta_val = 1.0 / (1.0 + math.exp(-b_b.item()))

                if use_qk_l2norm:
                    b_q = b_q / (torch.sqrt(torch.sum(b_q * b_q) + 1e-6))
                    b_k = b_k / (torch.sqrt(torch.sum(b_k * b_k) + 1e-6))

                b_q = b_q * scale

                # Recurrent delta rule
                b_h = h[n, hv].float()
                b_h *= math.exp(g)
                b_v -= torch.sum(b_h * b_k[:, None], dim=0)
                b_v *= beta_val
                b_h += b_k[:, None] * b_v[None, :]
                h[n, hv] = b_h

                outputs[n, t, hv] = torch.sum(b_h * b_q[:, None], dim=0)

    return outputs, h


def _make_inputs(B, T, H, HV, K, V, device="cuda"):
    """Create random input tensors for testing."""
    A_log = torch.randn(HV, dtype=torch.float32, device=device)
    dt_bias = torch.randn(HV, dtype=torch.bfloat16, device=device)
    q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device=device)
    k = torch.randn(B, T, H, K, dtype=torch.bfloat16, device=device)
    v = torch.randn(B, T, HV, V, dtype=torch.bfloat16, device=device)
    a = torch.randn(B, T, HV, dtype=torch.bfloat16, device=device)
    b = torch.randn(B, T, HV, dtype=torch.bfloat16, device=device)
    h0 = torch.randn(B, HV, K, V, dtype=torch.float32, device=device)
    return A_log, dt_bias, q, k, v, a, b, h0


def test_accuracy(
    B: int,
    T: int,
    H: int,
    HV: int,
    K: int,
    V: int,
    use_qk_l2norm: bool = False,
):
    """Test fused kernel accuracy against naive reference."""
    device = "cuda"
    torch.manual_seed(42)
    scale = K**-0.5

    A_log, dt_bias, q, k, v, a, b, h0 = _make_inputs(B, T, H, HV, K, V, device)

    out_tri, ht_tri = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        a=a.clone(),
        dt_bias=dt_bias,
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        b=b.clone(),
        scale=scale,
        initial_state=h0.clone(),
        inplace_final_state=True,
        cu_seqlens=None,
        block_map=None,
        sequence_lengths=None,
        use_qk_l2norm_in_kernel=use_qk_l2norm,
        softplus_beta=1.0,
        softplus_threshold=20.0,
    )

    out_ref, ht_ref = fused_sigmoid_gating_delta_rule_ref(
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        b=b,
        scale=scale,
        initial_state=h0.clone(),
        use_qk_l2norm=use_qk_l2norm,
    )

    assert_close("o", out_ref.reshape(B, T, HV, V), out_tri.reshape(B, T, HV, V), 0.005)
    assert_close("ht", ht_ref, ht_tri, 0.005)


if __name__ == "__main__":
    H = 8
    HV = 24
    K = 128
    V = 128

    # Accuracy tests with various batch sizes (T=1, inplace mode)
    for bs in [1, 2, 4, 8, 16, 32, 64]:
        logging.info(f"Testing accuracy: B={bs}, T=1, H={H}, HV={HV}, K={K}, V={V}")
        test_accuracy(bs, 1, H, HV, K, V)

    # With L2 norm
    logging.info("Testing accuracy with L2 norm: B=2, T=1")
    test_accuracy(2, 1, H, HV, K, V, use_qk_l2norm=True)

    # Various shapes
    for h, hv, k, v_dim in [(4, 8, 64, 64), (16, 32, 128, 128)]:
        logging.info(f"Testing accuracy: B=4, T=1, H={h}, HV={hv}, K={k}, V={v_dim}")
        test_accuracy(4, 1, h, hv, k, v_dim)

    # Large batch (exercises grid-size fix: N*HV > 65535).
    # Use smaller K/V to fit in GPU memory.
    for bs in [4096, 8192]:
        logging.info(f"Testing accuracy: B={bs}, T=1, H={H}, HV={HV}, K=16, V=16")
        test_accuracy(bs, 1, H, HV, 16, 16)
