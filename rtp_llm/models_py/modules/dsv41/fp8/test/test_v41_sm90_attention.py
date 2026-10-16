"""Real Hopper provider and V4.1 selected-row adapter contracts."""

import unittest

import torch


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class SM90AttentionTest(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() != (9, 0):
            self.skipTest("requires an SM90 GPU")

    def test_flashmla_hopper_bf16_head512_provider(self):
        """Do not mock the provider: missing Hopper kernels must fail here."""
        from flash_mla import flash_mla_sparse_fwd

        torch.manual_seed(4190)
        q = torch.randn(2, 64, 512, device="cuda", dtype=torch.bfloat16) * 0.1
        kv = torch.randn(128, 1, 512, device="cuda", dtype=torch.bfloat16) * 0.1
        indices = torch.arange(128, device="cuda", dtype=torch.int32).view(1, 1, -1)
        indices = indices.expand(2, 1, 128).contiguous()
        indices[1, :, 64:] = -1
        sink = torch.linspace(-1, 1, 64, device="cuda")
        out, _, lse = flash_mla_sparse_fwd(
            q, kv, indices, 512**-0.5, d_v=512, attn_sink=sink
        )
        for row, count in ((0, 128), (1, 64)):
            keys = kv[:count, 0].float()
            logits = q[row].float() @ keys.T * 512**-0.5
            logz = torch.logsumexp(logits, -1)
            denom = torch.logaddexp(logz, sink)
            expected = torch.exp(logits - denom[:, None]) @ keys
            torch.testing.assert_close(
                out[row].float(), expected, rtol=0.02, atol=0.003
            )
            torch.testing.assert_close(lse[row], logz, rtol=0.003, atol=0.003)

    def test_selected_dual_pool_and_graph_replay(self):
        self._check_selected_dual_pool_and_graph_replay(heads=32)

    def test_tp8_heads8_decode_rejects_native_v41_dual_pool(self):
        """SM90 decode must not silently fall back to sparse-prefill attention."""
        self._check_selected_dual_pool_and_graph_replay(heads=8, dispatch=True)

    def _check_selected_dual_pool_and_graph_replay(self, *, heads, dispatch=False):
        from rtp_llm.models_py.modules.dsv41.fp8._v41_fp4_triton import (
            quantize_and_insert_k_cache_fp4,
        )
        from rtp_llm.models_py.modules.dsv41.fp8._v41_sm90_attention import (
            selected_rows_attention,
        )
        from rtp_llm.models_py.modules.dsv41.fp8._v41_swa_triton import (
            quantize_and_insert_swa_k_cache,
        )
        from rtp_llm.models_py.modules.dsv41.fp8.test.test_v41_flash_mla_cache_format import (
            _read_rows,
        )
        from rtp_llm.models_py.modules.dsv41.fp8.decode.fp8_sparse_attn_decode_op import (
            SparseAttnV4DecodeFp8Op,
        )

        torch.manual_seed(4191)

        # Deliberately padded pages: the logical entry divisor is not stride/width.
        def pool(width):
            stride = 128 * width + 512
            storage = torch.zeros((3, stride), dtype=torch.uint8, device="cuda")
            return storage.as_strided((3, 128, width), (stride, width, 1))

        swa, glob = pool(528), pool(288)
        slots = torch.arange(128, 256, device="cuda")
        for target, quant in (
            (swa, quantize_and_insert_swa_k_cache),
            (glob, quantize_and_insert_k_cache_fp4),
        ):
            quant(
                torch.randn(128, 512, device="cuda", dtype=torch.bfloat16) * 0.2,
                target,
                slots,
            )
        q = torch.randn(2, 2, heads, 512, device="cuda", dtype=torch.bfloat16) * 0.1
        sink = torch.linspace(-1, 1, heads, device="cuda")
        op = SparseAttnV4DecodeFp8Op(heads, 512, 512**-0.5)
        primary = slots[:64].int().reshape(1, 1, 64).repeat(2, 2, 1)
        secondary = slots[64:].int().reshape(1, 1, 64).repeat(2, 2, 1)
        primary[0, 0, :3] = -1
        swa_lengths = torch.tensor([61, 57], device="cuda", dtype=torch.int32)
        global_lengths = torch.tensor([55, 63], device="cuda", dtype=torch.int32)

        def run():
            if dispatch:
                return op.forward(
                    q,
                    swa,
                    sink,
                    primary,
                    None,
                    topk_length=swa_lengths,
                    extra_k_cache=glob,
                    extra_topk_idxs=secondary,
                    extra_topk_length=global_lengths,
                    return_lse=True,
                )
            return selected_rows_attention(
                q,
                swa,
                primary,
                sink,
                512**-0.5,
                global_pool=glob,
                global_slots=secondary,
                swa_lengths=swa_lengths,
                global_lengths=global_lengths,
                return_lse=True,
            )

        def check(actual):
            out, actual_lse = actual
            self.assertEqual(out.shape, (2, 2, heads, 512))
            self.assertEqual(actual_lse.shape, (2, heads, 2))
            for b in range(2):
                for s in range(2):
                    ps = primary[b, s, : int(swa_lengths[b])]
                    gs = secondary[b, s, : int(global_lengths[b])]
                    kv = torch.cat(
                        (_read_rows(swa, ps[ps >= 0]), _read_rows(glob, gs[gs >= 0]))
                    )
                    logits = q[b, s].float() @ kv.T * 512**-0.5
                    logz = torch.logsumexp(logits, -1)
                    denominator = torch.logaddexp(logz, sink)
                    expected = torch.exp(logits - denominator[:, None]) @ kv
                    torch.testing.assert_close(
                        out[b, s].float(), expected, rtol=0.02, atol=0.003
                    )
                    torch.testing.assert_close(
                        actual_lse[b, :, s], logz, rtol=0.003, atol=0.003
                    )

        if dispatch:
            with self.assertRaisesRegex(ValueError, "SM90.*584B"):
                run()
            return

        check(run())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run()
        primary.copy_(primary.flip(-1))
        global_lengths.copy_(torch.tensor([19, 33], device="cuda", dtype=torch.int32))
        graph.replay()
        check(captured)
        # Head padding and slicing must remain live under replay, not freeze
        # the original query/sink tensors captured by the Python wrapper.
        q.copy_(q.flip(-2))
        sink.copy_(sink.flip(0) + 0.25)
        graph.replay()
        check(captured)

    def test_decode_rejects_native_v41_single_pool(self):
        from rtp_llm.models_py.modules.dsv41.fp8._v41_swa_triton import (
            quantize_and_insert_swa_k_cache,
        )
        from rtp_llm.models_py.modules.dsv41.fp8.decode.fp8_sparse_attn_decode_op import (
            SparseAttnV4DecodeFp8Op,
        )

        pool = torch.zeros((2, 128, 528), dtype=torch.uint8, device="cuda")
        slots = torch.arange(128, 256, device="cuda")
        quantize_and_insert_swa_k_cache(
            torch.ones(128, 512, dtype=torch.bfloat16, device="cuda"), pool, slots
        )
        q = torch.zeros((1, 1, 64, 512), dtype=torch.bfloat16, device="cuda")
        sink = torch.zeros(64, device="cuda")
        with self.assertRaisesRegex(ValueError, "SM90.*584B"):
            SparseAttnV4DecodeFp8Op(64, 512, 512**-0.5).forward(
                q, pool, sink, slots.int().view(1, 1, 128), None
            )

    def test_per_query_lengths_and_empty_rows(self):
        from rtp_llm.models_py.modules.dsv41.fp8._v41_sm90_attention import (
            selected_rows_attention,
        )
        from rtp_llm.models_py.modules.dsv41.fp8._v41_swa_triton import (
            quantize_and_insert_swa_k_cache,
        )

        pool = torch.zeros((2, 128, 528), dtype=torch.uint8, device="cuda")
        slots = torch.arange(128, 256, device="cuda")
        quantize_and_insert_swa_k_cache(
            torch.ones(128, 512, dtype=torch.bfloat16, device="cuda"), pool, slots
        )
        # 32 model heads exercise padding to the provider's 64-head tile.
        q = torch.zeros((2, 2, 32, 512), dtype=torch.bfloat16, device="cuda")
        selected = slots.int().view(1, 1, 128).expand(2, 2, 128).contiguous()
        lengths = torch.tensor([[0, 17], [13, 19]], dtype=torch.int32, device="cuda")
        out, lse = selected_rows_attention(
            q,
            pool,
            selected,
            torch.zeros(32, device="cuda"),
            512**-0.5,
            swa_lengths=lengths,
            return_lse=True,
        )
        counts = lengths.float()
        expected = (counts / (counts + 1))[:, :, None, None].expand_as(out)
        torch.testing.assert_close(out.float(), expected, rtol=0.004, atol=0.004)
        expected_lse = torch.where(counts > 0, counts.log(), float("inf"))
        torch.testing.assert_close(
            lse, expected_lse[:, None, :].expand_as(lse), rtol=0.003, atol=0.003
        )


if __name__ == "__main__":
    unittest.main()
