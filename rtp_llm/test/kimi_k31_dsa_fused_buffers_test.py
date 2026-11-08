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

    def test_prefill_absorb_fuses_suffix_and_padded_heads(self):
        from rtp_llm.models_py.kernels.cuda.dsa_buffer_ops import absorbed_query

        q = torch.randn(130, 96, 192, device="cuda", dtype=torch.bfloat16)[::2]
        weight = torch.randn(96, 128, 512, device="cuda", dtype=q.dtype) * .02
        actual = absorbed_query(q, weight, 512, 128, fuse_suffix=True)
        reference = torch.bmm(q[..., :128].transpose(0, 1), weight).transpose(0, 1)

        torch.testing.assert_close(actual[:, :96, :512], reference, rtol=0.008, atol=2e-5)
        torch.testing.assert_close(actual[:, :96, 512:], q[..., 128:], rtol=0, atol=0)
        self.assertEqual(actual[:, 96:].count_nonzero().item(), 0)

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

    def test_grouped_indexer_k_and_sfa_kv_match_separate_norms(self):
        from rtp_llm.models_py.triton_kernels.common.layernorm_gated import (
            grouped_indexer_k_sfa_kv_norm,
            independent_q_kv_rmsnorm,
            layer_norm_fwd,
        )

        bf16 = dict(device="cuda", dtype=torch.bfloat16)
        for rows, with_bias in ((129, False), (16384, True)):
            with self.subTest(rows=rows, with_bias=with_bias):
                packed = torch.randn(rows, 768, **bf16)
                latent, suffix, _, key = packed.split((512, 64, 64, 128), dim=-1)
                key_weight = torch.randn(128, device="cuda",
                                         dtype=torch.bfloat16 if with_bias else torch.float32)
                bias = torch.randn(128, **bf16) * .1 if with_bias else None
                kv_weight = torch.randn(512, **bf16)
                key_eps, kv_eps = 2e-5, 1e-6
                expected_key, _, _ = layer_norm_fwd(key, key_weight, bias, key_eps)
                expected_latent = independent_q_kv_rmsnorm(
                    latent, kv_weight, kv_eps,
                    q_width=1536, kv_width=512, full_rows=rows, suffix=suffix,
                )
                key_out, latent_out, suffix_out = grouped_indexer_k_sfa_kv_norm(
                    key, latent, suffix, key_weight, bias, kv_weight, key_eps=key_eps, kv_eps=kv_eps,
                )
                torch.testing.assert_close(key_out, expected_key, rtol=0, atol=0)
                torch.testing.assert_close(latent_out, expected_latent, rtol=0, atol=0)
                torch.testing.assert_close(suffix_out, suffix, rtol=0, atol=0)
                self.assertTrue(key_out.is_contiguous())
                self.assertEqual(latent_out.stride(), (576, 1))
                self.assertEqual(suffix_out.stride(), (576, 1))
                self.assertEqual(latent_out.untyped_storage().data_ptr(), suffix_out.untyped_storage().data_ptr())
                self.assertEqual(suffix_out.storage_offset(), latent_out.storage_offset() + 512)

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

    def test_flashmla_writes_zero_for_an_empty_query(self):
        from rtp_llm.models_py.utils.flash_mla_legacy import flash_mla_sparse_fwd

        query = torch.zeros(2, 128, 576, device="cuda", dtype=torch.bfloat16)
        kv = torch.ones(1, 1, 576, device="cuda", dtype=torch.bfloat16)
        indices = torch.full((2, 1, 2048), -1, device="cuda", dtype=torch.int32)
        indices[1, 0, 0] = 0

        output, _, lse = flash_mla_sparse_fwd(query, kv, indices, 192**-0.5)
        torch.testing.assert_close(output[0], torch.zeros_like(output[0]), rtol=0, atol=0)
        torch.testing.assert_close(output[1], torch.ones_like(output[1]), rtol=0, atol=0)
        self.assertTrue(torch.isposinf(lse[0]).all())

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
