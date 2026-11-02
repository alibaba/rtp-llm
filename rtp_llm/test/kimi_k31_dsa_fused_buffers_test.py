"""GPU contracts for K3.1 buffer fusion, independent of model golden outputs."""
import unittest

import torch
import torch.nn.functional as F


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class DsaFusedBufferTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3101)

    def test_final_absorbed_layout_and_graph(self):
        from rtp_llm.models_py.kernels.cuda.dsa_buffer_ops import absorbed_query

        q = torch.randn(17, 96, 192, device="cuda", dtype=torch.bfloat16) * .1
        weight = torch.randn(96, 128, 512, device="cuda", dtype=q.dtype) * .02

        def reference():
            nope = torch.bmm(q[..., :128].transpose(0, 1), weight).transpose(0, 1)
            return F.pad(torch.cat((nope, q[..., 128:]), -1), (0, 0, 0, 32))

        actual = absorbed_query(q, weight, 512, 128)
        torch.testing.assert_close(actual, reference(), rtol=0, atol=0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = absorbed_query(q, weight, 512, 128)
        q.add_(.01)
        graph.replay()
        torch.testing.assert_close(actual, reference(), rtol=0, atol=0)

    def test_packed_latent_norm_leaves_suffix_unchanged(self):
        from rtp_llm.models_py.triton_kernels.common.layernorm_gated import grouped_q_kv_rmsnorm

        for rows in (0, 1, 17, 128):
            source = torch.randn(rows, 2112, device="cuda", dtype=torch.bfloat16)
            q, latent, suffix = source.split((1536, 512, 64), -1)
            qw = torch.randn(1536, device="cuda", dtype=source.dtype)
            kw = torch.randn(512, device="cuda", dtype=source.dtype)
            old_q, old_latent = grouped_q_kv_rmsnorm(q, latent, qw, kw, 1e-5)
            new_q, new_latent = grouped_q_kv_rmsnorm(
                q, latent, qw, kw, 1e-5, kv_suffix=suffix
            )
            torch.testing.assert_close(new_q, old_q, rtol=0, atol=0)
            torch.testing.assert_close(new_latent, old_latent, rtol=0, atol=0)
            packed = new_latent.as_strided((rows, 576), (576, 1))
            torch.testing.assert_close(packed[:, 512:], suffix, rtol=0, atol=0)

    def test_raw_lse_strides_empty_and_nan(self):
        from rtp_llm.models_py.kernels.cuda.dsa_kv_chunk import zero_sparse_empty_rows
        from rtp_llm.models_py.triton_kernels.mla.lse_merge import merge_attention_states_in_place

        partial = torch.randn(3, 128, 512, device="cuda", dtype=torch.bfloat16)
        lse = torch.zeros(128, 3, device="cuda").t()
        lse[0, :96] = float("inf")
        partial[0] = float("nan")
        lse[2, 0] = float("nan")
        running = torch.zeros(3, 96, 512, device="cuda")
        canonical_lse = torch.full((3, 96), -float("inf"), device="cuda")
        merge_attention_states_in_place(
            running, canonical_lse, partial[:, :96], lse[:, :96], raw_partial_lse=True
        )
        self.assertTrue((running[0] == 0).all())
        self.assertTrue(torch.isneginf(canonical_lse[0]).all())
        torch.testing.assert_close(running[1], partial[1, :96].float(), rtol=0, atol=0)
        self.assertTrue(torch.isnan(running[2, 0]).all())
        zero_sparse_empty_rows(partial, lse, 96)
        self.assertTrue((partial[0, :96] == 0).all())
        self.assertTrue(torch.isnan(partial[2, 0]).all())

    def test_gate_masks_invalid_nan_without_hiding_valid_nan(self):
        from rtp_llm.models_py.triton_kernels.common.activation import gate_sigmoid_mul

        x = torch.ones(3, 96 * 128, device="cuda", dtype=torch.bfloat16)
        gate = torch.zeros_like(x)
        gate[0] = float("nan")
        gate[2] = float("nan")
        mask = torch.tensor([False, True, True], device="cuda")
        result = gate_sigmoid_mul(x, gate, mask)
        self.assertTrue((result[0] == 0).all())
        self.assertTrue((result[1] == .5).all())
        self.assertTrue(torch.isnan(result[2]).all())


if __name__ == "__main__":
    unittest.main()
