"""Unit tests for KDA (Kimi Delta Attention) prefill and decode ops.

Compares Triton kernel outputs against naive PyTorch reference implementations.
Tests: fused_kda_gate, chunk_kda prefill, fused_recurrent_kda decode,
       and prefill-decode consistency.
"""

import logging
import math
import os
import unittest

import torch
import torch.nn.functional as F

os.environ.setdefault("TRITON_F32_DEFAULT", "ieee")

from rtp_llm.models_py.triton_kernels.kimi_kda import (
    chunk_kda,
    fused_kda_gate,
    fused_recurrent_kda,
)

logging.basicConfig(
    level="INFO",
    format="[%(asctime)s][%(filename)s:%(lineno)s][%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)

# Kimi Linear per-TP config
H = 16  # local_num_heads (32 / TP=2)
K = 128  # head_k_dim
V = 128  # head_v_dim
DEVICE = "cuda"
DTYPE = torch.bfloat16
COS_THRESH = 0.99


def make_kda_inputs(batch, seq_len, seed=42, heads=H):
    """Create random KDA inputs matching Kimi Linear's actual shapes."""
    torch.manual_seed(seed)
    q = torch.randn(batch, seq_len, heads, K, device=DEVICE, dtype=DTYPE)
    k = torch.randn(batch, seq_len, heads, K, device=DEVICE, dtype=DTYPE)
    v = torch.randn(batch, seq_len, heads, V, device=DEVICE, dtype=DTYPE)
    g_raw = torch.randn(batch, seq_len, heads, K, device=DEVICE, dtype=DTYPE)
    beta = torch.randn(batch, seq_len, heads, device=DEVICE, dtype=DTYPE).sigmoid()
    A_log = torch.randn(heads, device=DEVICE, dtype=torch.float32) * 0.1
    dt_bias = torch.randn(heads * K, device=DEVICE, dtype=torch.float32) * 0.01
    initial_state = torch.zeros(batch, heads, K, V, device=DEVICE, dtype=torch.float32)
    return q, k, v, g_raw, beta, A_log, dt_bias, initial_state


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(
        a.flatten().float().unsqueeze(0),
        b.flatten().float().unsqueeze(0),
    ).item()


# ============================================================================
# Naive PyTorch reference implementations
# ============================================================================


def naive_kda_gate(g_raw, A_log, dt_bias=None):
    """Naive gate: g = -exp(A_log) * softplus(g_raw + dt_bias).
    g_raw: [tokens, H, K], A_log: [H], dt_bias: [H, K] or None.
    """
    g = g_raw.float()
    if dt_bias is not None:
        g = g + dt_bias.float()
    # A_log [H] -> [1, H, 1] for broadcast with [tokens, H, K]
    return -torch.exp(A_log.float()).unsqueeze(0).unsqueeze(-1) * F.softplus(g)


def naive_recurrent_kda(
    q, k, v, g, beta, scale, initial_state, A_log, dt_bias, lower_bound=None
):
    """Naive token-by-token KDA recurrent (reference for both prefill and decode).

    Args:
        q, k: [B, T, H, K]
        v: [B, T, H, V]
        g: [B, T, H, K] raw gate input (before activation)
        beta: [B, T, H]
        scale: float
        initial_state: [B, H, K, V]
        A_log: [H]
        dt_bias: [H*K] or None
    Returns:
        output: [B, T, H, V], final_state: [B, H, K, V]
    """
    B, T, Hd, Kd = q.shape
    Vd = v.shape[-1]
    q = F.normalize(q.float(), p=2, dim=-1) * scale
    k = F.normalize(k.float(), p=2, dim=-1)
    v = v.float()
    beta_f = beta.float()

    # Compute per-dim gate
    g_flat = g.float().reshape(B * T, Hd, Kd)
    dt_bias_reshaped = dt_bias.float().reshape(Hd, Kd) if dt_bias is not None else None
    gate_list = []
    for t in range(B * T):
        gi = g_flat[t]  # [H, K]
        if dt_bias_reshaped is not None:
            gi = gi + dt_bias_reshaped
        a = torch.exp(A_log.float()).unsqueeze(-1)
        gate_list.append(
            -a * F.softplus(gi)
            if lower_bound is None
            else lower_bound * torch.sigmoid(a * gi)
        )
    gk = torch.stack(gate_list).reshape(B, T, Hd, Kd)  # [B, T, H, K]

    state = initial_state.clone().float()
    outputs = []
    for t in range(T):
        # Decay
        state = state * torch.exp(gk[:, t]).unsqueeze(-1)  # [B, H, K, V]
        # Delta rule
        kt = k[:, t].unsqueeze(-1)  # [B, H, K, 1]
        vt = v[:, t].unsqueeze(-2)  # [B, H, 1, V]
        bt = beta_f[:, t].unsqueeze(-1).unsqueeze(-1)  # [B, H, 1, 1]
        correction = (state * kt).sum(dim=-2, keepdim=True)  # [B, H, 1, V]
        vt_corrected = (vt - correction) * bt
        state = state + kt * vt_corrected
        # Output
        qt = q[:, t].unsqueeze(-2)  # [B, H, 1, K]
        ot = (qt @ state).squeeze(-2)  # [B, H, V]
        outputs.append(ot)
    output = torch.stack(outputs, dim=1)
    return output, state


# ============================================================================
# Tests
# ============================================================================


class TestKdaOps(unittest.TestCase):

    def test_k3_paged_verify_candidate_states(self):
        # Verify candidate storage, independently of the engine's later commit.
        # Four target queries mean the current token plus three draft tokens.
        batch, width, heads, block_size = 5, 4, 12, 4096
        q, k, v, g, beta, alog, bias, initial = make_kda_inputs(
            batch, width, heads=heads
        )
        initial.normal_(mean=0, std=0.02)
        prefixes = (4095, 4096, 8191, 8192, 4096)
        table = torch.arange(1, 1 + batch * 7, device=DEVICE, dtype=torch.int32)
        table = table.reshape(batch, 7)
        table[-1].zero_()  # SP dummy request must not touch the null block.
        pool = torch.randn(
            1 + batch * 7, heads, K, V, device=DEVICE, dtype=torch.float32
        )
        for row, prefix in enumerate(prefixes[:-1]):
            pool[table[row, (prefix - 1) // block_size]] = initial[row]
        before = pool.clone()
        output, _ = fused_recurrent_kda(
            q,
            k,
            v,
            g,
            beta,
            initial_state=pool,
            A_log=alog,
            dt_bias=bias,
            inplace_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            lower_bound=-5.0,
            block_map=table,
            seq_size_per_block=block_size,
            sequence_lengths=torch.tensor(
                [prefix + 1 for prefix in prefixes], device=DEVICE, dtype=torch.int32
            ),
        )
        written = set()
        for accepted in range(4):
            count = accepted + 1
            expected_output, expected_state = naive_recurrent_kda(
                q[:-1, :count],
                k[:-1, :count],
                v[:-1, :count],
                g[:-1, :count],
                beta[:-1, :count],
                K**-0.5,
                initial[:-1].clone(),
                alog,
                bias,
                lower_bound=-5.0,
            )
            for row, prefix in enumerate(prefixes[:-1]):
                slot = int(table[row, prefix // block_size + accepted])
                written.add(slot)
                torch.testing.assert_close(
                    pool[slot], expected_state[row], rtol=0.03, atol=0.003
                )
            torch.testing.assert_close(
                output[:-1, accepted].float(),
                expected_output[:, -1].float(),
                rtol=0.03,
                atol=0.003,
            )
        untouched = [index for index in range(pool.shape[0]) if index not in written]
        torch.testing.assert_close(pool[untouched], before[untouched], rtol=0, atol=0)
        self.assertEqual(pool.dtype, torch.float32)

    def test_k3_bounded_gate_extremes(self):
        g = torch.linspace(-80, 80, 12 * K, device=DEVICE).reshape(1, 12, K)
        alog = torch.linspace(-1, 1, 12, device=DEVICE)
        bias = torch.linspace(-2, 2, 12 * K, device=DEVICE).reshape(12, K)
        expected = -5.0 * torch.sigmoid(alog.exp()[None, :, None] * (g + bias))
        actual = fused_kda_gate(g, alog, dt_bias=bias, lower_bound=-5.0)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        self.assertTrue(torch.all((actual >= -5) & (actual <= 0)))

    def test_k3_bounded_prefill_continuation_and_cached_decode(self):
        # K3 TP8 has 96 / 8 heads. Cross two 64-token kernel chunks plus a tail.
        for batch in (1, 3):
            with self.subTest(batch=batch):
                q, k, v, g, beta, alog, bias, initial = make_kda_inputs(
                    batch, 132, heads=12
                )
                initial.normal_(mean=0, std=0.02)
                reference, reference_state = naive_recurrent_kda(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    K**-0.5,
                    initial.clone(),
                    alog,
                    bias,
                    lower_bound=-5.0,
                )

                def prefill(begin, end, state):
                    return chunk_kda(
                        q[:, begin:end].contiguous(),
                        k[:, begin:end].contiguous(),
                        v[:, begin:end].contiguous(),
                        g[:, begin:end].contiguous(),
                        beta[:, begin:end].contiguous(),
                        initial_state=state,
                        output_final_state=True,
                        use_qk_l2norm_in_kernel=True,
                        use_gate_in_kernel=True,
                        return_intermediate_states=True,
                        A_log=alog,
                        dt_bias=bias,
                        lower_bound=-5.0,
                    )

                whole, whole_state, _ = prefill(0, 131, initial.clone())
                first, first_state, _ = prefill(0, 64, initial.clone())
                tail, tail_state, _ = prefill(64, 131, first_state.clone())
                decoded, decoded_state = fused_recurrent_kda(
                    q[:, 131:].contiguous(),
                    k[:, 131:].contiguous(),
                    v[:, 131:].contiguous(),
                    g[:, 131:].contiguous(),
                    beta[:, 131:].contiguous(),
                    initial_state=tail_state.clone(),
                    A_log=alog,
                    dt_bias=bias,
                    inplace_final_state=True,
                    use_qk_l2norm_in_kernel=True,
                    use_gate_in_kernel=True,
                    lower_bound=-5.0,
                )
                comparisons = {
                    "prefill_reference": (whole, reference[:, :131]),
                    "split_output": (torch.cat((first, tail), dim=1), whole),
                    "split_state": (tail_state, whole_state),
                    "decode_reference": (decoded, reference[:, 131:]),
                    "decode_state": (decoded_state, reference_state),
                }
                for name, (actual, expected) in comparisons.items():
                    actual, expected = actual.float(), expected.float()
                    delta = actual - expected
                    logging.info(
                        "K3 bounded KDA %s batch=%d max_abs=%g mean_abs=%g rmse=%g relative_l2=%g cosine=%g",
                        name,
                        batch,
                        delta.abs().max().item(),
                        delta.abs().mean().item(),
                        delta.square().mean().sqrt().item(),
                        (delta.norm() / expected.norm().clamp_min(1e-12)).item(),
                        cos_sim(actual, expected),
                    )
                    # Operator-level BF16 regression bounds; not a full-model
                    # accuracy threshold or a substitute for checkpoint alignment.
                    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.003)
                self.assertEqual(decoded_state.dtype, torch.float32)

    def test_gate_vs_naive(self):
        _, _, _, g_raw, _, A_log, dt_bias, _ = make_kda_inputs(1, 8)
        g_flat = g_raw.reshape(8, H, K)
        gate_kernel = fused_kda_gate(g_flat, A_log, dt_bias=dt_bias.reshape(H, K))
        gate_naive = naive_kda_gate(g_flat, A_log, dt_bias=dt_bias.reshape(H, K))
        sim = cos_sim(gate_kernel, gate_naive)
        self.assertGreater(sim, COS_THRESH, f"Gate vs naive cos_sim={sim:.6f}")
        self.assertTrue(
            (gate_kernel <= 1e-6).all(), "Gate values should be non-positive"
        )
        logging.info(f"test_gate_vs_naive PASSED (cos_sim={sim:.6f})")

    def test_gate_no_bias_vs_naive(self):
        _, _, _, g_raw, _, A_log, _, _ = make_kda_inputs(1, 4)
        g_flat = g_raw.reshape(4, H, K)
        gate_kernel = fused_kda_gate(g_flat, A_log, dt_bias=None)
        gate_naive = naive_kda_gate(g_flat, A_log, dt_bias=None)
        sim = cos_sim(gate_kernel, gate_naive)
        self.assertGreater(sim, COS_THRESH, f"Gate no-bias vs naive cos_sim={sim:.6f}")
        logging.info(f"test_gate_no_bias_vs_naive PASSED (cos_sim={sim:.6f})")

    def test_decode_vs_naive(self):
        for B in [1, 4, 8]:
            with self.subTest(B=B):
                q, k, v, g_raw, beta, A_log, dt_bias, state = make_kda_inputs(B, 1)
                scale = K**-0.5

                state_for_kernel = state.clone()
                o_kernel, fs_kernel = fused_recurrent_kda(
                    q,
                    k,
                    v,
                    g_raw,
                    beta,
                    initial_state=state_for_kernel,
                    A_log=A_log,
                    dt_bias=dt_bias,
                    inplace_final_state=True,
                    use_qk_l2norm_in_kernel=True,
                    use_gate_in_kernel=True,
                )

                o_naive, fs_naive = naive_recurrent_kda(
                    q,
                    k,
                    v,
                    g_raw,
                    beta,
                    scale,
                    state.clone(),
                    A_log,
                    dt_bias,
                )

                sim_o = cos_sim(o_kernel, o_naive)
                sim_s = cos_sim(fs_kernel, fs_naive)
                self.assertGreater(
                    sim_o, COS_THRESH, f"Decode output cos_sim={sim_o:.6f} (B={B})"
                )
                self.assertGreater(
                    sim_s, COS_THRESH, f"Decode state cos_sim={sim_s:.6f} (B={B})"
                )
                logging.info(
                    f"test_decode_vs_naive PASSED B={B} (o={sim_o:.6f}, s={sim_s:.6f})"
                )

    def test_prefill_vs_naive(self):
        for B, seq_len in [(1, 11), (1, 64), (1, 128), (2, 11), (2, 64), (2, 128)]:
            with self.subTest(B=B, seq_len=seq_len):
                q, k, v, g_raw, beta, A_log, dt_bias, state = make_kda_inputs(
                    B, seq_len
                )
                scale = K**-0.5

                o_kernel, fs_kernel, _ = chunk_kda(
                    q,
                    k,
                    v,
                    g_raw,
                    beta,
                    initial_state=state.clone(),
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=True,
                    use_gate_in_kernel=True,
                    return_intermediate_states=True,
                    A_log=A_log,
                    dt_bias=dt_bias,
                )

                o_naive, fs_naive = naive_recurrent_kda(
                    q,
                    k,
                    v,
                    g_raw,
                    beta,
                    scale,
                    state.clone(),
                    A_log,
                    dt_bias,
                )

                sim_o = cos_sim(o_kernel, o_naive)
                sim_s = cos_sim(fs_kernel, fs_naive)
                self.assertGreater(
                    sim_o,
                    COS_THRESH,
                    f"Prefill output cos_sim={sim_o:.6f} (B={B}, T={seq_len})",
                )
                self.assertGreater(
                    sim_s,
                    COS_THRESH,
                    f"Prefill state cos_sim={sim_s:.6f} (B={B}, T={seq_len})",
                )
                logging.info(
                    f"test_prefill_vs_naive PASSED B={B} T={seq_len} (o={sim_o:.6f}, s={sim_s:.6f})"
                )

    def test_prefill_then_decode(self):
        q_all, k_all, v_all, g_all, beta_all, A_log, dt_bias, state = make_kda_inputs(
            1, 12
        )

        o_pf, final_state_pf, _ = chunk_kda(
            q_all[:, :11],
            k_all[:, :11],
            v_all[:, :11],
            g_all[:, :11],
            beta_all[:, :11],
            initial_state=state.clone(),
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            return_intermediate_states=True,
            A_log=A_log,
            dt_bias=dt_bias,
        )

        o_dec, _ = fused_recurrent_kda(
            q_all[:, 11:12],
            k_all[:, 11:12],
            v_all[:, 11:12],
            g_all[:, 11:12],
            beta_all[:, 11:12],
            initial_state=final_state_pf.clone(),
            A_log=A_log,
            dt_bias=dt_bias,
            inplace_final_state=False,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
        )

        o_full, _, _ = chunk_kda(
            q_all,
            k_all,
            v_all,
            g_all,
            beta_all,
            initial_state=state.clone(),
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            return_intermediate_states=True,
            A_log=A_log,
            dt_bias=dt_bias,
        )

        sim = cos_sim(o_dec, o_full[:, 11:12])
        self.assertGreater(
            sim, COS_THRESH, f"Prefill-decode consistency failed: cos_sim={sim:.6f}"
        )
        logging.info(f"test_prefill_then_decode PASSED (cos_sim={sim:.6f})")


if __name__ == "__main__":
    unittest.main()
