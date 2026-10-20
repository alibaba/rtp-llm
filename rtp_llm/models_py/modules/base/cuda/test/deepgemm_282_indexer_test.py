"""GLM5 MXFP8 integration checks for the default CUDA13 DeepGEMM dependency."""

from types import SimpleNamespace
from unittest import SkipTest, TestCase, main

import torch

from rtp_llm.models_py.modules.base.cuda.indexer_op import IndexerOp
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_impl import (
    SparseMlaImpl,
)
from rtp_llm.ops.compute_ops import LayerKVCache


class DeepGemm282IndexerTest(TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is required")
        if torch.cuda.get_device_capability()[0] != 10:
            raise SkipTest("Blackwell is required for MXFP8 MQA")
        import deep_gemm

        self.assertTrue(
            not hasattr(deep_gemm, "fp8_mqa_logits")
            and hasattr(deep_gemm, "fp8_fp4_mqa_logits"),
            "DeepGEMM 2.8.2 with the new MQA APIs is required",
        )
        self.op = IndexerOp(32, 128, 512, 64)
        self.assertTrue(self.op.use_mxfp8)
        self.device = torch.device("cuda", torch.cuda.current_device())

    def test_q_quantization_matches_upstream_bytes(self) -> None:
        from deep_gemm.utils import per_token_cast_to_fp8

        torch.manual_seed(282)
        for rows in (1, 3, 129):
            q = torch.randn(rows, 32, 128, dtype=torch.bfloat16, device=self.device)
            q[0, 0] = torch.tensor(
                [223, 224, 225, -448], dtype=q.dtype, device=self.device
            ).repeat_interleave(32)
            actual_q, actual_sf = self.op.quant_q_only(q)
            expected_q, expected_sf = per_token_cast_to_fp8(
                q.view(-1, 128),
                use_ue8m0=True,
                gran_k=32,
                use_packed_ue8m0=True,
            )
            torch.testing.assert_close(
                actual_q.view(torch.uint8),
                expected_q.view(rows, 32, 128).view(torch.uint8),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                actual_sf, expected_sf.view(rows, 32, 1), rtol=0, atol=0
            )

    def _inputs(self):
        # Two independent requests with noncontiguous physical pages.
        table = torch.tensor([[2, 0], [1, 3]], dtype=torch.int32, device=self.device)
        logical_slots = torch.arange(256, device=self.device, dtype=torch.int64)
        slots = table.flatten().long()[logical_slots // 64] * 64 + logical_slots % 64
        k_values = (torch.arange(256, device=self.device) % 8 + 1).float() / 8
        k = k_values[:, None].expand(256, 128).to(torch.bfloat16).contiguous()
        cache = LayerKVCache()
        cache.kv_scale_base = torch.zeros(
            4, 64, 132, dtype=torch.uint8, device=self.device
        )
        self.op.quant_k_only(k, cache, slots)
        q = torch.ones(4, 32, 128, dtype=torch.bfloat16, device=self.device)
        q_fp8, q_sf = self.op.quant_q_only(q)
        weights = torch.ones(4, 32, 1, dtype=torch.bfloat16, device=self.device)
        fmha = SimpleNamespace(
            prefill_total_kv_tokens=256,
            prefill_max_kv_len=128,
            ks=torch.tensor([0, 0, 128, 128], dtype=torch.int32, device=self.device),
            ke=torch.tensor([7, 100, 129, 205], dtype=torch.int32, device=self.device),
            expanded_seq_lens=torch.tensor(
                [7, 100, 1, 77], dtype=torch.int32, device=self.device
            ),
            topk_indices_offset=torch.zeros(4, dtype=torch.int32, device=self.device),
            schedule_metadata=None,
        )
        inputs = SimpleNamespace(
            is_prefill=True,
            is_target_verify=False,
            is_draft_extend=False,
            kv_cache_kernel_block_id_device=table,
            cu_kv_seqlens=torch.tensor(
                [0, 128, 256], dtype=torch.int32, device=self.device
            ),
            decode_cu_seqlens_d=torch.tensor(
                [0, 2, 4], dtype=torch.int32, device=self.device
            ),
        )
        return q_fp8, q_sf, weights, cache, fmha, inputs, k_values

    def _assert_all_valid_indices(self, result, lengths) -> None:
        for row, length in enumerate(lengths):
            values = result[row]
            actual = values[values >= 0].sort().values
            expected = torch.arange(length, device=self.device, dtype=torch.int32)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(int((values == -1).sum()), self.op.index_topk - length)

    def test_ragged_and_cp_keep_request_local_coordinates(self) -> None:
        q, sf, weights, cache, fmha, inputs, _ = self._inputs()
        for backend in ("dsv4_per_row", "topk_v3_tie_break"):
            with self.subTest(backend=backend):
                self.op._prefill_topk_backend = backend
                result = self.op._get_topk_ragged(
                    q, weights, cache, fmha, inputs, q_scale=sf
                )
                self._assert_all_valid_indices(result, [7, 100, 1, 77])
                selected = torch.tensor(
                    [3, 1], dtype=torch.int64, device=self.device
                )
                cp_result = self.op._get_topk_ragged_cp(
                    q,
                    weights,
                    cache,
                    fmha,
                    inputs,
                    selected,
                    inputs.cu_kv_seqlens,
                    256,
                    fmha.ks[selected],
                    fmha.ke[selected],
                    fmha.expanded_seq_lens[selected],
                    fmha.topk_indices_offset[selected],
                    q_scale=sf,
                )
                self._assert_all_valid_indices(cp_result, [77, 100])

    def test_compressed_logits_match_exact_representable_scores(self) -> None:
        from deep_gemm.utils import per_token_cast_to_fp8

        q, sf, weights, _, fmha, _, k_values = self._inputs()
        k = k_values[:, None].expand(256, 128).to(torch.bfloat16).contiguous()
        k_fp8, k_sf = per_token_cast_to_fp8(
            k, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True
        )
        logits, starts, ends, base = self.op._prefill_logits(
            q, sf, (k_fp8, k_sf.flatten()), weights.squeeze(-1), fmha.ks, fmha.ke, 128
        )
        torch.testing.assert_close(starts, torch.zeros_like(starts), rtol=0, atol=0)
        torch.testing.assert_close(ends, fmha.ke - fmha.ks, rtol=0, atol=0)
        self.assertIsNone(base)
        # Q=1, K in exact eighths and gate=1: sum over 32 heads x 128 elements.
        for row, (ks, ke) in enumerate(zip(fmha.ks.tolist(), fmha.ke.tolist())):
            torch.testing.assert_close(
                logits[row, : ke - ks], 4096 * k_values[ks:ke], rtol=0, atol=0
            )

    def test_selected_ranking_depends_on_reordered_q_scales(self) -> None:
        # Every FP8 Q group normalizes to the same value, so its packed scales
        # alone determine which half of the keys wins. Dropping/reordering SF
        # independently of Q reverses the expected set of 512 selected tokens.
        table = (
            torch.arange(32, dtype=torch.int32, device=self.device).roll(7).view(2, 16)
        )
        ids = torch.arange(2048, dtype=torch.int64, device=self.device)
        slots = table.flatten().long()[ids // 64] * 64 + ids % 64
        sign = torch.where(ids % 1024 < 512, 1.0, -1.0)
        k = torch.zeros(2048, 128, dtype=torch.bfloat16, device=self.device)
        k[:, :32] = sign[:, None]
        k[:, 32:64] = -sign[:, None]
        cache = LayerKVCache()
        cache.kv_scale_base = torch.zeros(
            32, 64, 132, dtype=torch.uint8, device=self.device
        )
        self.op.quant_k_only(k, cache, slots)
        q = torch.zeros(4, 32, 128, dtype=torch.bfloat16, device=self.device)
        for row, (left, right) in enumerate(
            ((1, 0.25), (0.25, 1), (4, 16), (0.0625, 0.03125))
        ):
            q[row, :, :32] = left
            q[row, :, 32:64] = right
        q, sf = self.op.quant_q_only(q)
        weights = torch.ones(4, 32, 1, dtype=torch.bfloat16, device=self.device)
        fmha = SimpleNamespace(
            prefill_total_kv_tokens=2048,
            prefill_max_kv_len=1024,
            ks=torch.tensor([0, 0, 1024, 1024], dtype=torch.int32, device=self.device),
            ke=torch.tensor(
                [1024, 1024, 2048, 2048], dtype=torch.int32, device=self.device
            ),
            expanded_seq_lens=torch.full(
                (4,), 1024, dtype=torch.int32, device=self.device
            ),
            topk_indices_offset=torch.zeros(4, dtype=torch.int32, device=self.device),
            schedule_metadata=None,
        )
        inputs = SimpleNamespace(
            is_prefill=True,
            is_target_verify=False,
            is_draft_extend=False,
            kv_cache_kernel_block_id_device=table,
            cu_kv_seqlens=torch.tensor(
                [0, 1024, 2048], dtype=torch.int32, device=self.device
            ),
            decode_cu_seqlens_d=torch.tensor(
                [0, 2, 4], dtype=torch.int32, device=self.device
            ),
        )
        lower = torch.arange(512, dtype=torch.int32, device=self.device)
        upper = lower + 512
        expected = torch.stack((lower, upper, upper, lower))
        ragged = self.op._get_topk_ragged(q, weights, cache, fmha, inputs, q_scale=sf)
        torch.testing.assert_close(ragged.sort(dim=1).values, expected, rtol=0, atol=0)
        selected = torch.tensor([1, 0, 3, 2], dtype=torch.int64, device=self.device)
        cp = self.op._get_topk_ragged_cp(
            q,
            weights,
            cache,
            fmha,
            inputs,
            selected,
            inputs.cu_kv_seqlens,
            2048,
            fmha.ks[selected],
            fmha.ke[selected],
            fmha.expanded_seq_lens[selected],
            fmha.topk_indices_offset[selected],
            q_scale=sf,
        )
        torch.testing.assert_close(
            cp.sort(dim=1).values, expected[selected], rtol=0, atol=0
        )
        inputs.is_target_verify = True
        plan = SimpleNamespace(fmha_params=fmha, seq_size_per_block=64)
        SparseMlaImpl._refresh_paged_mqa_schedule_metadata(plan, inputs, False)
        paged = self.op._get_topk_paged(q, weights, cache, fmha, inputs, q_scale=sf)
        torch.testing.assert_close(paged.sort(dim=1).values, expected, rtol=0, atol=0)

    def test_paged_mtp_schedule_and_graph_replay(self) -> None:
        q, sf, weights, cache, fmha, inputs, _ = self._inputs()
        inputs.is_target_verify = True
        plan = SimpleNamespace(fmha_params=fmha, seq_size_per_block=64)

        def refresh():
            SparseMlaImpl._refresh_paged_mqa_schedule_metadata(plan, inputs, False)

        def run():
            return self.op._get_topk_paged(q, weights, cache, fmha, inputs, q_scale=sf)

        refresh()
        self._assert_all_valid_indices(run(), [7, 100, 1, 77])
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                refresh()
                run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            refresh()
            output = run()
        metadata_ptr = fmha.schedule_metadata.data_ptr()
        for lengths in ([7, 100, 1, 77], [3, 65, 2, 64]):
            fmha.expanded_seq_lens.copy_(
                torch.tensor(lengths, dtype=torch.int32, device=self.device)
            )
            graph.replay()
            self._assert_all_valid_indices(output, lengths)
            self.assertEqual(fmha.schedule_metadata.data_ptr(), metadata_ptr)


if __name__ == "__main__":
    main()
