"""Unit tests for KDA (Kimi Delta Attention) prefill and decode ops.

Compares cuLA prefill and Triton recurrent outputs against naive PyTorch reference implementations.
Tests: cuLA chunk prefill, fused_recurrent_kda decode,
       and prefill-decode consistency.
"""

import logging
import math
import os
import unittest
from itertools import product

import torch
import torch.nn.functional as F

os.environ.setdefault("TRITON_F32_DEFAULT", "ieee")

from rtp_llm.models_py.triton_kernels.kda import (
    fused_recurrent_kda,
)

logging.basicConfig(
    level="INFO",
    format="[%(asctime)s][%(filename)s:%(lineno)s][%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)

# Explicit KDA operator geometry; this is not a full-model configuration.
H = 16
K = 128  # head_k_dim
V = 128  # head_v_dim
DEVICE = "cuda"
DTYPE = torch.bfloat16
COS_THRESH = 0.99
# Pinned cuLA prefill supports the bounded-gate formulation in [-5, 0).
SAFE_GATE_LOWER_BOUND = -5.0


def make_kda_inputs(batch, seq_len, seed=42):
    """Create deterministic BF16 inputs for the H=16, K=V=128 operator case."""
    torch.manual_seed(seed)
    q = torch.randn(batch, seq_len, H, K, device=DEVICE, dtype=DTYPE)
    k = torch.randn(batch, seq_len, H, K, device=DEVICE, dtype=DTYPE)
    v = torch.randn(batch, seq_len, H, V, device=DEVICE, dtype=DTYPE)
    g_raw = torch.randn(batch, seq_len, H, K, device=DEVICE, dtype=DTYPE)
    beta = torch.randn(batch, seq_len, H, device=DEVICE, dtype=DTYPE).sigmoid()
    A_log = torch.randn(H, device=DEVICE, dtype=torch.float32) * 0.1
    dt_bias = torch.randn(H * K, device=DEVICE, dtype=torch.float32) * 0.01
    initial_state = torch.zeros(batch, H, K, V, device=DEVICE, dtype=torch.float32)
    return q, k, v, g_raw, beta, A_log, dt_bias, initial_state


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(
        a.flatten().float().unsqueeze(0),
        b.flatten().float().unsqueeze(0),
    ).item()


# ============================================================================
# Naive PyTorch reference implementations
# ============================================================================


def naive_recurrent_kda(
    q, k, v, g, beta, scale, initial_state, A_log, dt_bias, *, lower_bound=None
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
        lower_bound: None for softplus gates, or the negative sigmoid gate bound
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
        rate = torch.exp(A_log.float()).unsqueeze(-1)
        if lower_bound is None:
            gate_list.append(-rate * F.softplus(gi))
        else:
            # Independent FP32 reference for the bounded gate, not a clamp of
            # the softplus gate and not a call into either tested kernel.
            gate_list.append(lower_bound * torch.sigmoid(rate * gi))
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

    def test_decode_vs_naive(self):
        for B, lower_bound, nonzero_state, transpose_state_layout in product(
            (1, 4, 8), (None, SAFE_GATE_LOWER_BOUND), (False, True), (False, True)
        ):
            with self.subTest(
                B=B,
                lower_bound=lower_bound,
                nonzero_state=nonzero_state,
                transpose_state_layout=transpose_state_layout,
            ):
                q, k, v, g_raw, beta, A_log, dt_bias, state = make_kda_inputs(B, 1)
                if nonzero_state:
                    # A one-token update from zero cannot observe the gate.
                    state = torch.randn_like(state) * 0.125
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
                    lower_bound=lower_bound,
                    inplace_final_state=True,
                    use_qk_l2norm_in_kernel=True,
                    use_gate_in_kernel=True,
                    transpose_state_layout=transpose_state_layout,
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
                    lower_bound=lower_bound,
                )

                sim_o = cos_sim(o_kernel, o_naive)
                sim_s = cos_sim(fs_kernel, fs_naive)
                self.assertGreater(
                    sim_o,
                    COS_THRESH,
                    f"Decode output cos_sim={sim_o:.6f} (B={B}, lower_bound={lower_bound})",
                )
                self.assertGreater(
                    sim_s,
                    COS_THRESH,
                    f"Decode state cos_sim={sim_s:.6f} (B={B}, lower_bound={lower_bound})",
                )
                logging.info(
                    f"test_decode_vs_naive PASSED B={B} lower_bound={lower_bound} "
                    f"(o={sim_o:.6f}, s={sim_s:.6f})"
                )

    @torch.inference_mode()
    def test_prefill_vs_naive(self):
        from rtp_llm.models_py.utils.cutlass import setup_cutlass_import_path

        setup_cutlass_import_path()
        from cula.kda import chunk_kda

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
                    use_beta_sigmoid_in_kernel=False,
                    safe_gate=True,
                    lower_bound=SAFE_GATE_LOWER_BOUND,
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
                    lower_bound=SAFE_GATE_LOWER_BOUND,
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

    @torch.inference_mode()
    def test_chunked_prefill_then_decode_preserves_state(self):
        from rtp_llm.models_py.utils.cutlass import setup_cutlass_import_path

        setup_cutlass_import_path()
        from cula.kda import chunk_kda

        q, k, v, g, beta, A_log, dt_bias, initial = make_kda_inputs(2, 274)
        initial.normal_(std=0.1)
        reference, ref_state = naive_recurrent_kda(
            q,
            k,
            v,
            g,
            beta,
            K**-0.5,
            initial.clone(),
            A_log,
            dt_bias,
            lower_bound=SAFE_GATE_LOWER_BOUND,
        )
        for chunk_tokens in (128, 256):
            with self.subTest(chunk_tokens=chunk_tokens):
                state = initial.clone()
                for start in range(0, 273, chunk_tokens):
                    end = min(start + chunk_tokens, 273)
                    chunks = [
                        tensor[:, start:end].contiguous()
                        for tensor in (q, k, v, g, beta)
                    ]
                    output, state, _ = chunk_kda(
                        *chunks,
                        initial_state=state,
                        output_final_state=True,
                        use_qk_l2norm_in_kernel=True,
                        use_gate_in_kernel=True,
                        return_intermediate_states=True,
                        use_beta_sigmoid_in_kernel=False,
                        safe_gate=True,
                        lower_bound=SAFE_GATE_LOWER_BOUND,
                        A_log=A_log,
                        dt_bias=dt_bias,
                    )
                    self.assertGreater(
                        cos_sim(output, reference[:, start:end]), COS_THRESH
                    )
                    # The first token after a boundary depends most strongly
                    # on the preceding chunk; a whole-chunk cosine can hide it.
                    self.assertGreater(
                        cos_sim(output[:, :1], reference[:, start : start + 1]),
                        COS_THRESH,
                    )
                decoded, final_state = fused_recurrent_kda(
                    q[:, 273:274].contiguous(),
                    k[:, 273:274].contiguous(),
                    v[:, 273:274].contiguous(),
                    g[:, 273:274].contiguous(),
                    beta[:, 273:274].contiguous(),
                    initial_state=state,
                    A_log=A_log,
                    dt_bias=dt_bias,
                    lower_bound=SAFE_GATE_LOWER_BOUND,
                    inplace_final_state=False,
                    use_qk_l2norm_in_kernel=True,
                    use_gate_in_kernel=True,
                )
                self.assertGreater(cos_sim(decoded, reference[:, 273:274]), COS_THRESH)
                self.assertGreater(cos_sim(final_state, ref_state), COS_THRESH)

    @torch.inference_mode()
    def test_prefill_then_decode(self):
        from rtp_llm.models_py.utils.cutlass import setup_cutlass_import_path

        setup_cutlass_import_path()
        from cula.kda import chunk_kda

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
            use_beta_sigmoid_in_kernel=False,
            safe_gate=True,
            lower_bound=SAFE_GATE_LOWER_BOUND,
            A_log=A_log,
            dt_bias=dt_bias,
        )

        o_dec, final_state_dec = fused_recurrent_kda(
            q_all[:, 11:12],
            k_all[:, 11:12],
            v_all[:, 11:12],
            g_all[:, 11:12],
            beta_all[:, 11:12],
            initial_state=final_state_pf.clone(),
            A_log=A_log,
            dt_bias=dt_bias,
            lower_bound=SAFE_GATE_LOWER_BOUND,
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
            use_beta_sigmoid_in_kernel=False,
            safe_gate=True,
            lower_bound=SAFE_GATE_LOWER_BOUND,
            A_log=A_log,
            dt_bias=dt_bias,
        )

        o_naive, final_state_naive = naive_recurrent_kda(
            q_all,
            k_all,
            v_all,
            g_all,
            beta_all,
            K**-0.5,
            state.clone(),
            A_log,
            dt_bias,
            lower_bound=SAFE_GATE_LOWER_BOUND,
        )
        self.assertGreater(
            cos_sim(o_dec, o_naive[:, 11:12]),
            COS_THRESH,
            "Prefill-decode output differs from the bounded FP32 recurrence",
        )
        self.assertGreater(
            cos_sim(final_state_dec, final_state_naive),
            COS_THRESH,
            "Prefill-decode state differs from the bounded FP32 recurrence",
        )
        sim = cos_sim(o_dec, o_full[:, 11:12])
        self.assertGreater(
            sim, COS_THRESH, f"Prefill-decode consistency failed: cos_sim={sim:.6f}"
        )
        logging.info(f"test_prefill_then_decode PASSED (cos_sim={sim:.6f})")


if __name__ == "__main__":
    unittest.main()
