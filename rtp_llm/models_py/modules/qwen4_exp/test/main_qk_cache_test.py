import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.modules.qwen4_exp.qsa.main_qk_cache_triton import (
    prepare_main_qk_cache,
)
from rtp_llm.models_py.modules.qwen4_exp.test import (
    main_qk_norm_rope_test as native_tests,
)
from rtp_llm.ops.fused_rope_kvcache_op import (
    FusedRopeAttnParams,
    _get_fused_rope_kvcache,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class MainQkCacheTest(unittest.TestCase):
    def setUp(self):
        self.tester = native_tests.MainQkNormRopeTest()
        self.tester.setUp()

    def case(self, heads, batch, query, prefixes):
        cfg, norm, source, positions, params, original, plain, _ = (
            self.tester.setup_case(heads, batch * query)
        )
        columns = (max(prefixes) + query + 127) // 128
        pages = 1 + batch * columns
        table = torch.arange(1, pages, device="cuda", dtype=torch.int32).view(
            batch, columns
        )
        cu = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * query
        length = torch.full((batch,), query, dtype=torch.int32).pin_memory()
        prefix = torch.tensor(prefixes, dtype=torch.int32).pin_memory()
        params = FusedRopeAttnParams(
            _get_fused_rope_kvcache().convert_offset_to_block_array(table),
            None,
            None,
            positions.flatten(),
            cu,
            cu,
            length,
            prefix,
            length,
            query,
            max(prefixes),
            sum(prefixes) + batch * query,
            False,
            torch.bfloat16,
        )
        lens = (
            torch.tensor(prefixes, device="cuda", dtype=torch.int32)[:, None]
            + torch.arange(1, query + 1, device="cuda", dtype=torch.int32)
        ).contiguous()
        raw = (torch.randn(pages, 2, 1, 128, 256, device="cuda") * 0.2).to(
            torch.float8_e4m3fn
        )
        scales = torch.rand(pages, 256, device="cuda") + 0.5
        cache = SimpleNamespace(kv_cache_base=raw, kv_scale_base=scales)
        return cfg, norm, source, positions, params, original, table, lens, cache

    def prepared(self, c, cache):
        cfg, norm, source, positions, params, _, table, lens, _ = c
        plan = SimpleNamespace(kv_lens=lens, block_table=table)
        return prepare_main_qk_cache(
            source,
            positions,
            norm,
            cfg.rope_config,
            cache.kv_cache_base,
            cache.kv_scale_base,
            plan,
            params,
        )

    def compare(self, c):
        cfg, norm, source, positions, params, original, table, lens, initial = c
        copies = [
            SimpleNamespace(
                kv_cache_base=initial.kv_cache_base.clone(),
                kv_scale_base=initial.kv_scale_base.clone(),
            )
            for _ in range(2)
        ]
        source_saved = source.clone()
        operation = self.prepared(c, copies[1])
        self.assertIsNotNone(operation)
        torch.testing.assert_close(
            copies[1].kv_cache_base.view(torch.uint8),
            initial.kv_cache_base.view(torch.uint8),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            copies[1].kv_scale_base, initial.kv_scale_base, rtol=0, atol=0
        )
        actual = operation.run()
        expected = original.forward(norm(source.clone()), copies[0], params)
        torch.testing.assert_close(
            actual, expected[:, : cfg.head_num * 256], rtol=0, atol=0
        )
        torch.testing.assert_close(source, source_saved, rtol=0, atol=0)
        torch.testing.assert_close(
            copies[0].kv_cache_base.view(torch.uint8),
            copies[1].kv_cache_base.view(torch.uint8),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            copies[0].kv_scale_base, copies[1].kv_scale_base, rtol=0, atol=0
        )

    def test_native_query_cache_scale_bytes_across_pages_and_heads(self):
        for heads in (3, 6, 12):
            for batch, query in (
                (4, 1),
                (1, 4),
                (2, 4),
                (3, 4),
                (4, 4),
                (5, 4),
                (6, 4),
                (7, 4),
                (8, 4),
                (3, 3),
            ):
                for prefix in (0, 127, 4093):
                    c = self.case(
                        heads, batch, query, [prefix + i for i in range(batch)]
                    )
                    c[3][:, ::2] = -1
                    c[3][0] = 0
                    if prefix == 127:
                        c[1].q_weight = c[1].q_weight.bfloat16()
                        c[1].k_weight = c[1].k_weight.bfloat16()
                    self.compare(c)

    def test_native_dense_packed_cache_and_four_dimensional_scales(self):
        c = list(self.case(12, 4, 4, [127, 255, 4093, 19]))
        cache = c[-1]
        cache.kv_cache_base = cache.kv_cache_base.view(cache.kv_cache_base.shape[0], -1)
        cache.kv_scale_base = cache.kv_scale_base.view(-1, 2, 1, 128)
        c[4].padding_offset = torch.zeros(16, device="cuda", dtype=torch.int32)
        self.compare(c)

    def test_dummy_page_zero_matches_native_with_disjoint_positions(self):
        c = list(self.case(12, 4, 4, [1, 11, 21, 31]))
        c[6].zero_()
        c[4].kv_cache_offset.copy_(
            _get_fused_rope_kvcache().convert_offset_to_block_array(c[6])
        )
        self.compare(c)

    def test_single_row_retains_original_complete_consumer(self):
        for query in (1, 2, 3):
            c = self.case(12, 1, query, [127])
            self.assertIsNone(self.prepared(c, c[-1]))

    def test_unsupported_layout_returns_before_any_write(self):
        c = self.case(12, 1, 4, [127])
        initial = c[-1]
        for mode in ("cache_stride", "scale_stride", "bf16_cache"):
            cache = SimpleNamespace(
                kv_cache_base=initial.kv_cache_base, kv_scale_base=initial.kv_scale_base
            )
            if mode == "cache_stride":
                base = torch.zeros(
                    initial.kv_cache_base.shape[0],
                    65664,
                    device="cuda",
                    dtype=torch.float8_e4m3fn,
                )
                cache.kv_cache_base = torch.as_strided(
                    base, initial.kv_cache_base.shape, (65664, 32768, 32768, 256, 1)
                )
            elif mode == "scale_stride":
                base = torch.zeros(initial.kv_scale_base.shape[0], 272, device="cuda")
                cache.kv_scale_base = base[:, :256]
            else:
                cache.kv_cache_base = initial.kv_cache_base.bfloat16()
            before_cache = cache.kv_cache_base.view(torch.uint8).clone()
            before_scales = cache.kv_scale_base.clone()
            self.assertIsNone(self.prepared(c, cache))
            torch.testing.assert_close(
                before_cache, cache.kv_cache_base.view(torch.uint8), rtol=0, atol=0
            )
            torch.testing.assert_close(
                before_scales, cache.kv_scale_base, rtol=0, atol=0
            )

    def test_two_graphs_refresh_positions_pages_gamma_lengths_and_scales(self):
        c = self.case(12, 4, 4, [127, 255, 4093, 19])
        cfg, norm, source, positions, params, original, table, lens, cache = c
        graphs = []
        outputs = []
        for _ in range(3):
            self.prepared(c, cache).run()
        for _ in range(2):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outputs.append(self.prepared(c, cache).run())
            graphs.append(graph)
        for generation in range(8):
            source.normal_()
            positions.random_(0, 32768)
            norm.q_weight.normal_(0, 0.1)
            norm.k_weight.normal_(0, 0.1)
            lens.add_(1)
            params.prefix_lengths.add_(1)
            table.copy_(table.flip(1))
            params.kv_cache_offset.copy_(
                _get_fused_rope_kvcache().convert_offset_to_block_array(table)
            )
            cache.kv_cache_base.copy_(
                (torch.randn_like(cache.kv_cache_base, dtype=torch.float32) * 0.2).to(
                    torch.float8_e4m3fn
                )
            )
            cache.kv_scale_base.uniform_(0.1, 2.0)
            oracle = SimpleNamespace(
                kv_cache_base=cache.kv_cache_base.clone(),
                kv_scale_base=cache.kv_scale_base.clone(),
            )
            graphs[generation % 2].replay()
            expected = original.forward(norm(source.clone()), oracle, params)
            torch.testing.assert_close(
                outputs[generation % 2], expected[:, :3072], rtol=0, atol=0
            )
            torch.testing.assert_close(
                cache.kv_cache_base.view(torch.uint8),
                oracle.kv_cache_base.view(torch.uint8),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                cache.kv_scale_base, oracle.kv_scale_base, rtol=0, atol=0
            )
        for graph in graphs:
            graph.reset()

    def test_target_owner_routes_raw_norm_and_gate_only_with_explicit_flag(self):
        from rtp_llm.models_py.modules.qwen4_exp.qsa.sparse_gqa_impl import (
            SparseGqaFmhaImpl,
        )

        c = self.case(12, 1, 4, [127])
        source = c[2]
        gate = torch.zeros(4, 3072, device="cuda", dtype=torch.bfloat16)
        result = torch.empty_like(gate)
        owner = SimpleNamespace(
            is_target_verify=True,
            _is_mtp_draft=False,
            attn_configs=SimpleNamespace(use_logn_attn=False),
            forward=Mock(return_value=result),
        )
        with patch.dict(os.environ, {"QWEN4_QSA_FUSED_KV_WRITER": "1"}):
            actual = SparseGqaFmhaImpl.forward_with_qk_norm_and_output_gate(
                owner, source, c[-1], 3, gate, c[1]
            )
        self.assertIs(actual, result)
        owner.forward.assert_called_once_with(
            source, c[-1], 3, qk_norm=c[1], output_gate=gate
        )

    def mutation_owner(self, c):
        cfg, norm, source, positions, params, original, table, lens, cache = c
        return SimpleNamespace(
            _selected_indices=None,
            _paged_kv_scale_args=Mock(return_value={}),
            _is_draft_graph=lambda: False,
            _is_prefix_reuse_prefill=lambda: False,
            is_target_verify=True,
            is_prefill=True,
            _prepare_paged_inputs=Mock(
                return_value=SimpleNamespace(kv_lens=lens, block_table=table)
            ),
            rope_kvcache_impl=Mock(),
            rope_params=params,
            attn_configs=cfg,
            _qsa_main_cache_mutation_started=False,
        )

    def test_fused_launch_failure_marks_mutation_before_run(self):
        from rtp_llm.models_py.modules.qwen4_exp.qsa.sparse_gqa_impl import (
            SparseGqaFmhaImpl,
        )

        c = self.case(12, 1, 4, [127])
        owner = self.mutation_owner(c)

        def fail():
            self.assertTrue(owner._qsa_main_cache_mutation_started)
            raise RuntimeError("injected fused writer launch failure")

        operation = SimpleNamespace(run=Mock(side_effect=fail))
        with patch(
            "rtp_llm.models_py.modules.qwen4_exp.qsa.main_qk_cache_triton.prepare_main_qk_cache",
            return_value=operation,
        ):
            with self.assertRaisesRegex(
                RuntimeError, "injected fused writer launch failure"
            ):
                SparseGqaFmhaImpl.forward(
                    owner,
                    c[2],
                    c[-1],
                    3,
                    selected_indices=torch.empty(
                        4, 2051, device="cuda", dtype=torch.int32
                    ),
                    qk_norm=c[1],
                )
        owner.rope_kvcache_impl.forward.assert_not_called()

    def test_tiny_fallback_normalizes_once_then_uses_original_writer(self):
        from rtp_llm.models_py.modules.qwen4_exp.qsa.sparse_gqa_impl import (
            SparseGqaFmhaImpl,
        )

        c = self.case(12, 1, 1, [127])
        owner = self.mutation_owner(c)
        normalized = c[1](c[2].clone())
        norm = Mock(return_value=normalized)

        def stop(qkv, cache, params):
            self.assertIs(qkv, normalized)
            self.assertTrue(owner._qsa_main_cache_mutation_started)
            raise RuntimeError("original writer reached")

        owner.rope_kvcache_impl.forward.side_effect = stop
        with patch(
            "rtp_llm.models_py.modules.qwen4_exp.qsa.main_qk_cache_triton.prepare_main_qk_cache",
            return_value=None,
        ):
            with self.assertRaisesRegex(RuntimeError, "original writer reached"):
                SparseGqaFmhaImpl.forward(
                    owner,
                    c[2],
                    c[-1],
                    3,
                    selected_indices=torch.empty(
                        1, 2051, device="cuda", dtype=torch.int32
                    ),
                    qk_norm=norm,
                )
        norm.assert_called_once_with(c[2])


if __name__ == "__main__":
    unittest.main()
