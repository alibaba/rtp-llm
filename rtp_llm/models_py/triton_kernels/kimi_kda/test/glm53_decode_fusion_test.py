"""GLM Decode fusion preserves paged cache updates and bounded FP32 recurrence."""

import unittest
from itertools import product

import torch
from rtp_llm.models_py.triton_kernels.causal_conv1d import causal_conv1d_update
from rtp_llm.models_py.triton_kernels.common.layernorm_gated import RmsNormGated
from rtp_llm.models_py.triton_kernels.kimi_kda import fused_recurrent_kda
from rtp_llm.models_py.triton_kernels.kimi_kda.glm53_short_conv import (
    glm53_kda_short_conv_decode,
)
from rtp_llm.models_py.triton_kernels.kimi_kda.rms_norm_gate import (
    kimi_kda_rms_norm_sigmoid_gate,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class Glm53DecodeFusionTest(unittest.TestCase):
    def test_convolution_pages_strides_and_dtypes(self):
        torch.manual_seed(9301)
        for batch in (1, 7, 16, 48, 64):
            for dtype, weight_dtype in product(
                (torch.bfloat16, torch.float32), repeat=2
            ):
                channels = 3 * 64 * 128
                pages = batch * 2 + 1
                x = (
                    torch.randn(batch, channels, device="cuda", dtype=torch.bfloat16)
                    * 0.25
                )
                weight = (
                    torch.randn(channels, 4, device="cuda", dtype=torch.bfloat16) * 0.25
                ).to(weight_dtype)
                # A padded physical-page stride catches accidental dense-cache addressing.
                storage = (
                    torch.randn(pages, 3 * channels + 64, device="cuda", dtype=dtype)
                    * 0.25
                )
                baseline = storage.clone()
                candidate = storage.clone()
                c0 = baseline[:, : 3 * channels].view(pages, 3, channels)
                c1 = candidate[:, : 3 * channels].view(pages, 3, channels)
                table = torch.zeros(batch, 1026, device="cuda", dtype=torch.int32)
                lengths = torch.tensor(
                    [128, 129, 130, 131073] * ((batch + 3) // 4),
                    device="cuda",
                    dtype=torch.int32,
                )[:batch]
                rows = torch.arange(batch, device="cuda")
                table[rows, (lengths - 2) // 128] = (rows + 1).to(torch.int32)
                # Same-page updates deliberately alias read/write; boundaries use a new page.
                dst = torch.where(
                    (lengths - 2) // 128 == (lengths - 1) // 128,
                    rows + 1,
                    rows + batch + 1,
                )
                table[rows, (lengths - 1) // 128] = dst.to(torch.int32)
                y = causal_conv1d_update(
                    x.unsqueeze(-1),
                    c0.transpose(1, 2),
                    weight,
                    activation="silu",
                    block_map=table,
                    seq_size_per_block=128,
                    sequence_lengths=lengths,
                ).squeeze(-1)
                qkv = glm53_kda_short_conv_decode(x, weight, c1, table, lengths, 128)
                torch.testing.assert_close(torch.cat(qkv, dim=-1), y, rtol=0, atol=0)
                torch.testing.assert_close(candidate, baseline, rtol=0, atol=0)
                self.assertTrue(all(t.is_contiguous() for t in qkv))

    def test_graph_remapping_and_padding(self):
        torch.manual_seed(9302)
        batch, channels = 7, 3 * 128
        x = torch.randn(batch, channels, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(channels, 4, device="cuda", dtype=torch.bfloat16).float()
        state = torch.randn(20, 3, channels, device="cuda", dtype=torch.bfloat16)
        initial = state.clone()
        table = torch.arange(1, batch + 1, device="cuda", dtype=torch.int32).view(
            batch, 1
        )
        lengths = torch.full((batch,), 10, device="cuda", dtype=torch.int32)
        for _ in range(3):
            glm53_kda_short_conv_decode(x, weight, state, table, lengths, 128)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = glm53_kda_short_conv_decode(x, weight, state, table, lengths, 128)
        # Exit row 2, reorder all other requests, and replay with live device metadata.
        table.copy_(
            torch.tensor(
                [[9], [7], [0], [4], [1], [8], [3]], device="cuda", dtype=torch.int32
            )
        )
        state.copy_(initial)
        graph.replay()
        torch.cuda.synchronize()
        expected = initial.clone()
        reference = glm53_kda_short_conv_decode(
            x, weight, expected, table, lengths, 128
        )
        for a, b in zip(out, reference):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(state, expected, rtol=0, atol=0)
        torch.testing.assert_close(state[0], initial[0], rtol=0, atol=0)

    def test_recurrent_low_warps_multistep(self):
        torch.manual_seed(9303)
        batch, heads, dim = 48, 64, 128
        q, k, v, g = [
            torch.randn(batch, 1, heads, dim, device="cuda", dtype=torch.bfloat16) * 0.2
            for _ in range(4)
        ]
        beta = torch.randn(batch, 1, heads, device="cuda", dtype=torch.bfloat16)
        pages = batch + 1
        storage = torch.randn(pages, heads * dim * dim + 128, device="cuda") * 0.03
        a = storage.clone()
        b = storage.clone()
        states = [
            t[:, : heads * dim * dim].view(pages, heads, dim, dim) for t in (a, b)
        ]
        table = (
            torch.randperm(batch, device="cuda", dtype=torch.int32)
            .add_(1)
            .view(batch, 1)
        )
        table[3] = 0  # graph padding must remain zero and must never mutate page zero
        lengths = torch.full((batch,), 64, device="cuda", dtype=torch.int32)
        args = dict(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A_log=torch.zeros(heads, device="cuda"),
            dt_bias=torch.zeros(heads * dim, device="cuda"),
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            lower_bound=-5.0,
            block_map=table,
            seq_size_per_block=128,
            sequence_lengths=lengths,
        )
        for _ in range(32):
            old, _ = fused_recurrent_kda(**args, initial_state=states[0])
            new, _ = fused_recurrent_kda(
                **args, initial_state=states[1], decode_low_warps=True
            )
            torch.testing.assert_close(new, old, rtol=1 / 128, atol=1e-5)
            torch.testing.assert_close(b, a, rtol=3e-5, atol=3e-6)
            self.assertEqual(int(torch.count_nonzero(new[3])), 0)
        torch.testing.assert_close(b[0], storage[0], rtol=0, atol=0)

    def test_k3_norm_gate_matches_glm(self):
        torch.manual_seed(9304)
        for rows in (1, 7, 48 * 64, 64 * 64):
            x = torch.randn(rows, 128, device="cuda", dtype=torch.bfloat16)
            gate = torch.randn_like(x)
            weight = torch.randn(128, device="cuda", dtype=torch.bfloat16)
            old = RmsNormGated(weight, eps=1e-6, activation="sigmoid")(x, gate)
            new = kimi_kda_rms_norm_sigmoid_gate(x, gate, weight, 1e-6)
            torch.testing.assert_close(new, old, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
