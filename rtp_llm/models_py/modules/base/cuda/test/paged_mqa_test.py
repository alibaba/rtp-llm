"""Exercise the production MTP logits wrapper and graph schedule refresh."""

import os
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, main, skipUnless
from unittest.mock import patch

os.environ.setdefault(
    "DG_JIT_CACHE_DIR",
    str(Path(os.environ.get("TEST_TMPDIR", "/tmp")) / "paged_mqa_jit"),
)

import torch

from rtp_llm.models_py.modules.base.cuda.indexer_op import IndexerOp
from rtp_llm.models_py.modules.base.cuda.paged_mqa import (
    adaptive_target_layout,
    expand_adaptive_paged_mqa_table,
    paged_mqa_context_lens,
    refresh_paged_mqa_request_ids,
    reuse_adaptive_paged_mqa_tables,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_impl import (
    SparseMlaImpl,
    SparseMlaOp,
)
from rtp_llm.ops.compute_ops import PyAttentionInputs, rtp_llm_ops

PAGE_TABLE_REUSE_ENV = {
    "RTP_LLM_ADAPTIVE_VERIFICATION": "1",
    "RTP_LLM_ADAPTIVE_INDEXER_REUSE_PAGE_TABLE": "1",
}


class PagedMqaLayoutTest(TestCase):
    @patch.dict(os.environ, {"RTP_LLM_ADAPTIVE_VERIFICATION": "1"})
    def test_adaptive_rescan_uses_mla_implementation_graph_context(self):
        stock = object()
        op = SimpleNamespace(
            _paged_topk_op=stock, _adaptive_target_cluster_rescan=True, index_topk=2048
        )
        inputs = PyAttentionInputs()
        inputs.is_target_verify = True
        # MLA does not populate the Python attention-input graph attribute.
        inputs.is_cuda_graph = False

        def select(*, graph=True, rows=64, width=264704):
            return IndexerOp._select_paged_topk(
                op, inputs, rows, width, is_cuda_graph=graph
            )

        self.assertIs(select(rows=56), rtp_llm_ops.topk_v3_adaptive_cluster_rescan)
        self.assertIs(select(), rtp_llm_ops.topk_v3_adaptive_cluster_rescan)
        self.assertIs(select(graph=False), stock)
        self.assertIs(select(rows=84), stock)
        self.assertIs(select(width=65536), stock)
        self.assertIs(select(width=524352), stock)
        for field in ("uniform_target_verify_layout", "is_draft_extend"):
            setattr(inputs, field, True)
            self.assertIs(select(), stock)
            setattr(inputs, field, False)
        op._adaptive_target_cluster_rescan = False
        self.assertIs(select(), stock)

    def test_full_width_binding_keeps_uniform_layout_with_adaptive_enabled(self):
        inputs = PyAttentionInputs()
        inputs.is_target_verify = True
        inputs.kv_cache_kernel_block_id_device = torch.zeros((14, 1), dtype=torch.int32)
        params = SimpleNamespace(expanded_seq_lens=torch.arange(84, dtype=torch.int32))
        with (
            patch.dict(os.environ, {"RTP_LLM_ADAPTIVE_VERIFICATION": "1"}),
            patch(
                "rtp_llm.models_py.modules.base.cuda.paged_mqa._native_paged_mqa_enabled",
                return_value=True,
            ),
        ):
            self.assertFalse(inputs.uniform_target_verify_layout)
            self.assertTrue(adaptive_target_layout(inputs))
            self.assertEqual(paged_mqa_context_lens(inputs, params).shape, (84, 1))
            inputs.uniform_target_verify_layout = True
            self.assertFalse(adaptive_target_layout(inputs))
            self.assertEqual(paged_mqa_context_lens(inputs, params).shape, (14, 6))
            self.assertIsNone(
                refresh_paged_mqa_request_ids(
                    inputs, params, params.expanded_seq_lens, None, forbid_realloc=True
                )
            )

    @patch.dict(os.environ, {"RTP_LLM_ADAPTIVE_VERIFICATION": "1"}, clear=True)
    def test_page_table_reuse_respects_scoped_role_and_switch(self):
        inputs = SimpleNamespace(is_target_verify=True)
        table = torch.arange(12, dtype=torch.int32).view(3, 4)
        ids = torch.tensor([0, 1, 1], dtype=torch.int32)
        with reuse_adaptive_paged_mqa_tables(inputs):
            cached = expand_adaptive_paged_mqa_table(table, ids)
            self.assertIs(
                expand_adaptive_paged_mqa_table(table.view_as(table), ids), cached
            )
            for disabled in (
                SimpleNamespace(is_target_verify=False),
                SimpleNamespace(
                    is_target_verify=True, uniform_target_verify_layout=True
                ),
                SimpleNamespace(is_target_verify=True, is_draft_extend=True),
            ):
                with (
                    self.subTest(inputs=disabled),
                    reuse_adaptive_paged_mqa_tables(disabled),
                ):
                    self.assertIsNot(
                        expand_adaptive_paged_mqa_table(table, ids), cached
                    )
            self.assertIs(expand_adaptive_paged_mqa_table(table, ids), cached)
        for flags in (
            {"RTP_LLM_ADAPTIVE_INDEXER_REUSE_PAGE_TABLE": "0"},
            {**PAGE_TABLE_REUSE_ENV, "RTP_LLM_ADAPTIVE_VERIFICATION": "0"},
        ):
            with (
                self.subTest(flags=flags),
                patch.dict(os.environ, flags),
                reuse_adaptive_paged_mqa_tables(inputs),
            ):
                first = expand_adaptive_paged_mqa_table(table, ids)
                self.assertIsNot(expand_adaptive_paged_mqa_table(table, ids), first)

    @patch.dict(os.environ, PAGE_TABLE_REUSE_ENV)
    def test_page_table_reuse_distinguishes_groups_views_and_row_maps(self):
        inputs = SimpleNamespace(is_target_verify=True)
        table_a = torch.arange(16, dtype=torch.int32).view(4, 4)
        table_b = table_a + 100
        ids_a = torch.tensor([0, 1, 1], dtype=torch.int32)
        ids_b = torch.tensor([2, 1, 0], dtype=torch.int32)
        with reuse_adaptive_paged_mqa_tables(inputs):
            for table, ids in (
                (table_a, ids_a),
                (table_b, ids_a),
                (table_a.T, ids_a),
                (table_a, ids_b),
            ):
                with self.subTest(table=table, ids=ids):
                    expanded = expand_adaptive_paged_mqa_table(table, ids)
                    torch.testing.assert_close(
                        expanded, table.index_select(0, ids.long())
                    )
                    self.assertIs(expand_adaptive_paged_mqa_table(table, ids), expanded)

    @patch.dict(os.environ, PAGE_TABLE_REUSE_ENV)
    def test_page_table_reuse_is_discarded_on_forward_exit_and_exception(self):
        inputs = SimpleNamespace(is_target_verify=True)
        table = torch.arange(12, dtype=torch.int32).view(3, 4)
        ids = torch.tensor([0, 1, 1], dtype=torch.int32)
        with self.assertRaisesRegex(RuntimeError, "test forward failure"):
            with reuse_adaptive_paged_mqa_tables(inputs):
                old = expand_adaptive_paged_mqa_table(table, ids)
                raise RuntimeError("test forward failure")
        table.add_(100)
        ids.copy_(torch.tensor([2, 0, 1], dtype=torch.int32))
        self.assertIsNot(expand_adaptive_paged_mqa_table(table, ids), old)
        with reuse_adaptive_paged_mqa_tables(inputs):
            current = expand_adaptive_paged_mqa_table(table, ids)
            self.assertIsNot(current, old)
            torch.testing.assert_close(current, table.index_select(0, ids.long()))
        with reuse_adaptive_paged_mqa_tables(inputs):
            self.assertIsNot(expand_adaptive_paged_mqa_table(table, ids), current)


@skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10,
    "SM10x CUDA required",
)
class PagedMqaTest(TestCase):
    def setUp(self):
        torch.manual_seed(921)
        self.env = patch.dict(
            os.environ,
            {"RTP_LLM_INDEXER_NATIVE_MTP": "1", "RTP_LLM_ADAPTIVE_VERIFICATION": "0"},
        )
        self.env.start()
        self.addCleanup(self.env.stop)

    def fixture(
        self,
        batch,
        next_n,
        draft=False,
        heads=32,
        *,
        blocks=20,
        topk=512,
        use_configured_topk=False,
    ):
        page, dim = 64, 128
        count = batch * blocks + 1
        cache = torch.empty((count, page, dim + 4), device="cuda", dtype=torch.uint8)
        raw = cache.view(count, -1)
        keys = torch.randn((count, page, dim), device="cuda").to(torch.float8_e4m3fn)
        raw[:, : page * dim].copy_(keys.view(torch.uint8).reshape(count, -1))
        scales = torch.rand((count, page), device="cuda") * 0.05 + 0.01
        raw[:, page * dim :].copy_(scales.view(torch.uint8))
        table = (torch.randperm(count - 1, device="cuda", dtype=torch.int32) + 1).view(
            batch, blocks
        )
        # Include a zero-length graph-padding row and a page-boundary crossing.
        prefix = (
            torch.arange(batch, device="cuda", dtype=torch.int32) * 59 + 63
        ) % 1000
        lengths = prefix[:, None] + torch.arange(
            1, next_n + 1, device="cuda", dtype=torch.int32
        )
        if batch > 1:
            lengths[-1].zero_()
        inputs = SimpleNamespace(
            is_prefill=True,
            is_target_verify=not draft,
            is_draft_extend=draft,
            kv_cache_kernel_block_id_device=table,
            kv_cache_block_id_device=torch.zeros_like(table),
        )
        params = SimpleNamespace(
            expanded_seq_lens=lengths.flatten(),
            kvlen_d=lengths[:, -1].contiguous(),
            schedule_metadata=None,
        )
        owner = SimpleNamespace(fmha_params=params, seq_size_per_block=page)
        q = torch.randn((batch * next_n, heads, dim), device="cuda").to(
            torch.float8_e4m3fn
        )
        weights = torch.rand((batch * next_n, heads), device="cuda") / heads
        op = IndexerOp(heads, dim, topk, 64)
        paged_topk = op._paged_topk_op

        def record_logits(scores, lens, out, workspace, k, width):
            op.last_logits = scores
            if use_configured_topk:
                return paged_topk(scores, lens, out, workspace, k, width)
            return rtp_llm_ops.topk_v3_tie_break(
                scores, torch.zeros_like(lens), lens, out, k, width
            )

        op._paged_topk_op = record_logits
        return op, owner, inputs, SimpleNamespace(kv_scale_base=cache), q, weights

    def call(self, op, owner, inputs, cache, q, weights):
        return op._get_topk_paged(q, weights, cache, owner.fmha_params, inputs)

    def refresh(self, owner, inputs, forbid=False):
        SparseMlaImpl._refresh_paged_mqa_schedule_metadata(owner, inputs, forbid)

    @patch.dict(os.environ, PAGE_TABLE_REUSE_ENV)
    def test_page_table_reuse_replays_current_groups_and_mapping(self):
        inputs = SimpleNamespace(is_target_verify=True)
        table_a = torch.arange(16, dtype=torch.int32, device="cuda").view(4, 4)
        table_b = table_a + 100
        ids = torch.tensor([0, 1, 1], dtype=torch.int32, device="cuda")
        side = torch.cuda.Stream()

        def forward():
            with reuse_adaptive_paged_mqa_tables(inputs):
                result_a = expand_adaptive_paged_mqa_table(table_a, ids)
                self.assertIs(expand_adaptive_paged_mqa_table(table_a, ids), result_a)
                result_b = expand_adaptive_paged_mqa_table(table_b, ids)
                main_stream = torch.cuda.current_stream()
                side.wait_stream(main_stream)
                with torch.cuda.stream(side):
                    side_a = expand_adaptive_paged_mqa_table(table_a, ids)
                    self.assertIsNot(side_a, result_a)
                main_stream.wait_stream(side)
                return result_a, result_b, side_a

        forward()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result_a, result_b, side_a = forward()
        for i in range(1, 4):
            table_a.copy_(table_a.roll(i, dims=1))
            table_b.copy_(table_b.roll(i, dims=0))
            ids.copy_((ids + i) % table_a.size(0))
            graph.replay()
            expected_a = table_a.index_select(0, ids.long())
            torch.testing.assert_close(result_a, expected_a)
            torch.testing.assert_close(side_a, expected_a)
            torch.testing.assert_close(result_b, table_b.index_select(0, ids.long()))

    def test_native_and_flattened_wrapper(self):
        for batch, next_n, draft, heads in [
            (1, 4, False, 32),
            (3, 3, True, 32),
            (16, 4, False, 32),
            (32, 4, False, 32),
            (64, 4, False, 32),
            (3, 5, True, 64),
            (3, 1, False, 32),
        ]:
            with self.subTest(batch=batch, next_n=next_n, draft=draft, heads=heads):
                args = self.fixture(batch, next_n, draft, heads)
                op, owner, inputs = args[:3]
                with patch.dict(os.environ, {"RTP_LLM_INDEXER_NATIVE_MTP": "0"}):
                    self.refresh(owner, inputs)
                    ref_topk = self.call(*args)
                    ref_logits = op.last_logits
                self.refresh(owner, inputs)
                actual = self.call(*args)
                valid = (
                    torch.arange(ref_logits.shape[1], device="cuda")[None, :]
                    < owner.fmha_params.expanded_seq_lens[:, None]
                )
                self.assertTrue(torch.equal(ref_logits[valid], op.last_logits[valid]))
                self.assertTrue(
                    torch.equal(ref_topk.sort().values, actual.sort().values)
                )
                self.assertEqual(
                    tuple(paged_mqa_context_lens(inputs, owner.fmha_params).shape),
                    (batch, next_n),
                )

    def test_graph_replay_refreshes_lengths_and_cache_groups(self):
        args = self.fixture(3, 4)
        op, owner, inputs = args[:3]
        self.refresh(owner, inputs)
        schedule_ptr = owner.fmha_params.schedule_metadata.data_ptr()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.call(*args)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured_topk = self.call(*args)
        captured_logits = op.last_logits
        for step in range(3):
            inputs.kv_cache_kernel_block_id_device.copy_(
                inputs.kv_cache_kernel_block_id_device.roll(1, 1)
            )
            owner.fmha_params.expanded_seq_lens.add_(1)
            self.refresh(owner, inputs, True)
            self.assertEqual(
                owner.fmha_params.schedule_metadata.data_ptr(), schedule_ptr
            )
            graph.replay()
            with patch.dict(os.environ, {"RTP_LLM_INDEXER_NATIVE_MTP": "0"}):
                ref_owner = SimpleNamespace(
                    fmha_params=SimpleNamespace(
                        expanded_seq_lens=owner.fmha_params.expanded_seq_lens,
                        kvlen_d=owner.fmha_params.kvlen_d,
                        schedule_metadata=None,
                    ),
                    seq_size_per_block=64,
                )
                self.refresh(ref_owner, inputs)
                ref = self.call(op, ref_owner, *args[2:])
            valid = (
                torch.arange(captured_logits.shape[1], device="cuda")[None, :]
                < owner.fmha_params.expanded_seq_lens[:, None]
            )
            self.assertTrue(torch.equal(captured_logits[valid], op.last_logits[valid]))
            self.assertTrue(torch.equal(captured_topk.sort().values, ref.sort().values))

    def test_request_id_storage_is_owned_by_bound_params(self):
        params = rtp_llm_ops.SparseMlaParams()
        self.assertIsNone(params.paged_mqa_request_ids)
        ids = torch.zeros(4, dtype=torch.int32)
        params.paged_mqa_request_ids = ids
        self.assertEqual(params.paged_mqa_request_ids.data_ptr(), ids.data_ptr())

    def test_adaptive_ragged_groups_match_flattened_after_graph_replay(self):
        multi_request_widths = [(4, 5, 6), (1, 6, 2), (6, 0, 3), (0, 0, 0)]
        for batch, replay_widths, context_offset in [
            (1, [(5,), (1,), (6,), (0,)], 0),
            (3, multi_request_widths, 0),
            (3, multi_request_widths, 2048),
        ]:
            with self.subTest(batch=batch, context_offset=context_offset), patch.dict(
                os.environ,
                {
                    "RTP_LLM_ADAPTIVE_VERIFICATION": "1",
                    "GLM5_INDEXER_TOPK_BACKEND": "topk_v3",
                },
            ):
                long_context = context_offset > 0
                args = self.fixture(
                    batch,
                    6,
                    blocks=36 if long_context else 20,
                    topk=2048 if long_context else 512,
                    use_configured_topk=long_context,
                )
                op, owner, inputs, cache, q, weights = args
                params = owner.fmha_params
                capacity = q.shape[0]
                params.batch_indice_d = torch.zeros(
                    capacity, device=q.device, dtype=torch.int32
                )

                def fill(widths):
                    batch_ids = [
                        request
                        for request, width in enumerate(widths)
                        for _ in range(width)
                    ]
                    lengths = [
                        context_offset + 63 + request * 61 + offset
                        for request, width in enumerate(widths)
                        for offset in range(1, width + 1)
                    ]
                    padding = capacity - len(lengths)
                    params.batch_indice_d.copy_(
                        torch.tensor(batch_ids + [0] * padding, dtype=torch.int32)
                    )
                    params.expanded_seq_lens.copy_(
                        torch.tensor(lengths + [0] * padding, dtype=torch.int32)
                    )

                fill((6,) * batch)
                self.refresh(owner, inputs)
                ids_ptr = params.paged_mqa_request_ids.data_ptr()
                schedule_ptr = params.schedule_metadata.data_ptr()
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        self.call(*args)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured_topk = self.call(*args)
                captured_logits = op.last_logits

                for widths in replay_widths:
                    fill(widths)
                    table = inputs.kv_cache_kernel_block_id_device
                    table.copy_(table.roll(1, 1))
                    self.refresh(owner, inputs, True)
                    self.assertEqual(params.paged_mqa_request_ids.data_ptr(), ids_ptr)
                    self.assertEqual(params.schedule_metadata.data_ptr(), schedule_ptr)
                    valid_rows = params.expanded_seq_lens > 0
                    self.assertTrue(
                        torch.all(params.paged_mqa_request_ids[~valid_rows] == -1)
                    )
                    graph.replay()
                    ref_owner = SimpleNamespace(
                        fmha_params=SimpleNamespace(
                            expanded_seq_lens=params.expanded_seq_lens,
                            batch_indice_d=params.batch_indice_d,
                            schedule_metadata=None,
                        ),
                        seq_size_per_block=64,
                    )
                    with patch.dict(os.environ, {"RTP_LLM_INDEXER_NATIVE_MTP": "0"}):
                        self.refresh(ref_owner, inputs)
                        reference = self.call(op, ref_owner, inputs, cache, q, weights)
                    if long_context:
                        selected = captured_topk[valid_rows]
                        self.assertTrue(torch.all(selected >= 0))
                        self.assertTrue(
                            torch.all(
                                selected < params.expanded_seq_lens[valid_rows, None]
                            )
                        )
                    valid_logits = (
                        torch.arange(captured_logits.shape[1], device=q.device)[None, :]
                        < params.expanded_seq_lens[:, None]
                    )
                    self.assertTrue(
                        torch.equal(
                            captured_logits[valid_logits], op.last_logits[valid_logits]
                        )
                    )
                    self.assertTrue(
                        torch.equal(
                            captured_topk[valid_rows].sort().values,
                            reference[valid_rows].sort().values,
                        )
                    )

    def test_single_token_decode_and_empty_queries(self):
        args = self.fixture(3, 1)
        op, owner, inputs, cache, q, weights = args
        inputs.is_prefill = inputs.is_target_verify = False
        self.assertEqual(
            tuple(paged_mqa_context_lens(inputs, owner.fmha_params).shape), (3, 1)
        )
        self.refresh(owner, inputs)
        self.call(*args)
        inputs.is_target_verify = True
        owner.fmha_params.expanded_seq_lens = torch.empty(
            0, dtype=torch.int32, device="cuda"
        )
        out = self.call(op, owner, inputs, cache, q[:0], weights[:0])
        self.assertEqual(tuple(out.shape), (0, 512))

    def test_invalid_request_layout_fails_before_kernel(self):
        _, owner, inputs, *_ = self.fixture(3, 4)
        owner.fmha_params.expanded_seq_lens = owner.fmha_params.expanded_seq_lens[:-1]
        with self.assertRaisesRegex(ValueError, "equal rows"):
            paged_mqa_context_lens(inputs, owner.fmha_params)

    def test_schedule_replay_requires_initialized_storage(self):
        _, owner, inputs, *_ = self.fixture(3, 4)
        with self.assertRaisesRegex(RuntimeError, "not captured"):
            self.refresh(owner, inputs, True)
        self.refresh(owner, inputs)
        owner.fmha_params.schedule_metadata = torch.empty(
            1, dtype=torch.int32, device="cuda"
        )
        with self.assertRaisesRegex(RuntimeError, "shape changed"):
            self.refresh(owner, inputs, True)

    def test_physical_index_tiles_match_at_dispatch_boundary(self):
        for tokens, topk in [
            *((tokens, 2048) for tokens in (4, 64, 127, 128, 129, 256)),
            *((128, topk) for topk in (256, 384, 512, 768)),
        ]:
            batch = (tokens + 3) // 4
            op = SparseMlaOp(64, 512, 64, 192, 64, 1.0, topk)
            op.block_table = torch.randint(
                1, 1000, (batch, 20), device="cuda", dtype=torch.int32
            )
            op.block_table[:, 3] = 0
            op.mla_params = SimpleNamespace(
                batch_indice_d=torch.arange(tokens, device="cuda", dtype=torch.int32)
                // 4
            )
            indices = torch.randint(
                -7, 1400, (tokens, topk), device="cuda", dtype=torch.int32
            )
            with patch.dict(os.environ, {"RTP_LLM_DSA_WIDE_INDEX_MAP": "0"}):
                reference = op._convert_topk_indices_to_global(indices)
            with patch.dict(os.environ, {"RTP_LLM_DSA_WIDE_INDEX_MAP": "1"}):
                actual = op._convert_topk_indices_to_global(indices)
            self.assertTrue(torch.equal(reference, actual))


if __name__ == "__main__":
    main()
