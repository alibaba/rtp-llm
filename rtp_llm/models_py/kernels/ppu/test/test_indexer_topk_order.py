"""PPU TopK ordering with real selection kernels and changing graph inputs."""

from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch

import torch

from rtp_llm.models_py.kernels.ppu.indexer_topk import stable_topk
from rtp_llm.models_py.modules.base.cuda import indexer_op
from rtp_llm.ops.compute_ops import LayerKVCache, PyAttentionInputs, rtp_llm_ops


class IndexerTopkOrderTest(TestCase):
    def make_op(self):
        return indexer_op.IndexerOp(1, 128, 2048, 0, stable_topk=True)

    def assert_selection(self, actual, starts, ends, relative=False):
        actual = actual.cpu().tolist()
        for row, start, end in zip(actual, starts, ends):
            expected = list(range(max(start, end - 2048), end))
            if relative:
                expected = [index - start for index in expected]
            expected += [-1] * (2048 - len(expected))
            self.assertEqual(row, expected)

    def test_prefill_order_and_request_masks(self):
        op = self.make_op()
        self.assertTrue(op._ppu_stable_topk)
        starts = [0, 0, 128, 512]
        ends = [17, 2048, 8192, 4096]
        ks = torch.tensor(starts, dtype=torch.int32, device="cuda")
        ke = torch.tensor(ends, dtype=torch.int32, device="cuda")
        logits = torch.arange(8192, dtype=torch.float32, device="cuda").repeat(4, 1)
        output = torch.empty(4, 2048, dtype=torch.int32, device="cuda")
        for _ in range(12):
            op._run_prefill_topk(logits, ks, ke, output, 8192)
            self.assert_selection(output, starts, ends, relative=True)
        # Identical ties may select a different set from another backend, but
        # within this native backend the same set must have a repeatable order.
        logits.zero_()
        ke.fill_(8192)
        ks.zero_()
        op._run_prefill_topk(logits, ks, ke, output, 8192)
        reference = output.cpu().clone()
        self.assertTrue(
            torch.equal(reference, torch.arange(2048, dtype=torch.int32).repeat(4, 1))
        )
        for _ in range(12):
            op._run_prefill_topk(logits, ks, ke, output, 8192)
            self.assertTrue(torch.equal(reference, output.cpu()))

    def test_paged_graph_replay_changes_lengths_and_scores(self):
        op = self.make_op()
        logits = torch.arange(8192, dtype=torch.float32, device="cuda").repeat(4, 1)
        lengths = torch.full((4,), 8192, dtype=torch.int32, device="cuda")
        params = SimpleNamespace(expanded_seq_lens=lengths, kvlen_d=lengths)
        inputs = SimpleNamespace(
            kv_cache_kernel_block_id_device=torch.zeros(
                4, 128, dtype=torch.int32, device="cuda"
            )
        )
        cache = SimpleNamespace(
            seq_size_per_block=64,
            kv_cache_base=torch.zeros(1, 64 * 132, dtype=torch.uint8, device="cuda"),
        )
        q = torch.zeros(4, 1, 128, device="cuda")
        weights = torch.ones(4, 1, 1, device="cuda")
        with patch.object(
            indexer_op, "_fp8_paged_mqa_logits_compat", return_value=logits
        ):
            for _ in range(3):
                op._get_topk_paged(q, weights, cache, params, inputs)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = op._get_topk_paged(q, weights, cache, params, inputs)
            for ends in ([17, 2048, 8192, 4096], [4096, 1, 8192, 1024]):
                lengths.copy_(torch.tensor(ends, dtype=torch.int32, device="cuda"))
                for _ in range(6):
                    graph.replay()
                    self.assert_selection(output, [0] * 4, ends)
            logits.neg_()
            lengths.fill_(8192)
            graph.replay()
            expected = torch.arange(2048, dtype=torch.int32).repeat(4, 1)
            self.assertTrue(torch.equal(output.cpu(), expected))

    def test_canonicalization_is_opt_in_on_ppu(self):
        for ppu in (False, True):
            for stable in (False, True):
                with patch.object(indexer_op, "is_ppu", return_value=ppu):
                    op = indexer_op.IndexerOp(1, 128, 2048, 0, stable_topk=stable)
                    self.assertEqual(op._ppu_stable_topk, ppu and stable)

    def test_wide_context_keeps_the_64_bit_path(self):
        for columns in (34817, 65537):
            with self.subTest(columns=columns):
                logits = torch.arange(columns, dtype=torch.float32, device="cuda")[None]
                starts = torch.zeros(1, dtype=torch.int32, device="cuda")
                ends = torch.full_like(starts, columns)
                output = torch.empty(1, 2048, dtype=torch.int32, device="cuda")
                with patch(
                    "rtp_llm.models_py.kernels.ppu.indexer_topk._stable_topk_small",
                    side_effect=AssertionError(
                        "wide context must retain legacy selection"
                    ),
                ):
                    stable_topk(logits, starts, ends, output)
                self.assert_selection(output, [0], [columns])

    def test_production_width_graph_switches_mixed_row_bounds_and_score_bits(self):
        columns = 34816
        scores = torch.randn(8, columns * 2, device="cuda")[:, ::2]
        starts = torch.zeros(8, dtype=torch.int32, device="cuda")
        ends = torch.full_like(starts, columns)
        output = torch.empty(8, 2048, dtype=torch.int32, device="cuda")
        for _ in range(3):
            stable_topk(scores, starts, ends, output)
        torch.cuda.synchronize()
        graph, stream = torch.cuda.CUDAGraph(), torch.cuda.Stream()
        with torch.cuda.graph(graph, stream=stream):
            stable_topk(scores, starts, ends, output)
        for repeat in range(3):
            scores.normal_()
            scores[1].zero_()
            scores[1, ::2] = -0.0
            scores[4, :2000] = 100
            scores.view(torch.int32)[6, ::7] = -1
            begin = [0, 1, 7, 0, 100, 0, -7, columns]
            end = [0, 18, 2055, 2049, 8293, 32833, columns, columns]
            if repeat % 2:
                begin, end = list(reversed(begin)), list(reversed(end))
            starts.copy_(torch.tensor(begin, dtype=torch.int32, device="cuda"))
            ends.copy_(torch.tensor(end, dtype=torch.int32, device="cuda"))
            torch.cuda.synchronize()
            graph.replay()
            torch.cuda.synchronize()
            # Independent CPU integer keys retain the original NaN/zero/tie
            # contract; the runtime never reads lengths or thresholds on CPU.
            col = torch.arange(columns)
            expected = []
            for bits, start, end in zip(scores.cpu().view(torch.int32), begin, end):
                normalized = bits.clone()
                normalized[(normalized & 0x7FFFFFFF) == 0] = 0
                ordered = torch.where(
                    normalized < 0, normalized ^ 0x7FFFFFFF, normalized
                )
                keys = (ordered.long() << 32) | (0xFFFFFFFF - col)
                keys[(col < start) | (col >= end)] = -9223372036854775808
                selected = torch.argsort(keys, descending=True)[:2048]
                selected = selected[keys[selected] != -9223372036854775808]
                indices = (selected - start).sort().values.tolist()
                expected.append(indices + [-1] * (2048 - len(indices)))
            self.assertEqual(output.cpu().tolist(), expected)

    def test_partial_ties_signed_scores_and_large_row_count(self):
        op = self.make_op()
        scores = torch.zeros(1030, 8192, device="cuda")
        scores[:, :2000] = 1
        starts = torch.zeros(1030, dtype=torch.int32, device="cuda")
        ends = torch.full((1030,), 8192, dtype=torch.int32, device="cuda")
        output = torch.empty(1030, 2048, dtype=torch.int32, device="cuda")
        expected = torch.arange(2048, dtype=torch.int32).repeat(1030, 1)
        for _ in range(4):
            op._run_prefill_topk(scores, starts, ends, output, 8192)
            self.assertTrue(torch.equal(output.cpu(), expected))
        scores = torch.linspace(-1, 1, 8192, device="cuda").repeat(4, 1)
        scores[1] = torch.arange(8192, device="cuda") % 255 - 128
        scores[2].zero_()
        scores[2, ::2] = -0.0
        scores[3] = (
            torch.arange(8192, dtype=torch.int32).view(torch.float32).to(scores.device)
        )
        starts, ends, output = starts[:4], ends[:4], output[:4]
        op._run_prefill_topk(scores, starts, ends, output, 8192)
        reference = (
            torch.argsort(scores.cpu(), descending=True, stable=True)[:, :2048]
            .sort(dim=-1)
            .values.int()
        )
        self.assertTrue(torch.equal(output.cpu(), reference))

    def test_small_batch_threshold_ties_ragged_bounds_and_strides(self):
        torch.manual_seed(34817)
        columns = 34817
        scores = torch.randn(4, columns * 2, device="cuda")[:, ::2]
        scores[0].fill_(float("-inf"))
        scores[2].zero_()
        scores[2, :2000] = 1
        scores[3].zero_()
        scores[3, ::2] = -0.0
        starts, ends = [-7, 123, 0, 32600], [0, 34000, 33000, columns]
        output = torch.empty(4, 2048, dtype=torch.int32, device="cuda")
        stable_topk(
            scores,
            torch.tensor(starts, dtype=torch.int32, device="cuda"),
            torch.tensor(ends, dtype=torch.int32, device="cuda"),
            output,
        )
        expected = []
        for row, start, end in zip(scores.cpu(), starts, ends):
            valid = torch.arange(max(start, 0), min(end, columns))
            selected = torch.argsort(row[valid], descending=True, stable=True)[:2048]
            indices = (valid[selected] - start).sort().values.tolist()
            expected.append(indices + [-1] * (2048 - len(indices)))
        self.assertEqual(output.cpu().tolist(), expected)

    def test_small_batch_nonfinite_keys_preserve_legacy_integer_order(self):
        columns = 8193
        scores = (
            torch.arange(columns, dtype=torch.int32).repeat(6, 1).view(torch.float32)
        )
        bits = scores.view(torch.int32)
        bits[0, ::3] = -1  # Negative NaN has the minimum 32-bit ordered key.
        bits[0, 1::3] = 0x7FC00001
        bits[1, ::2] = -2147483648  # Signed zero must tie positive zero.
        bits[1, 1::2] = 0
        scores[2].fill_(float("-inf"))
        scores[3].fill_(float("inf"))
        bits[4].fill_(-1)
        starts = [0, 17, 100, 0, 999, 0]
        ends = [columns, columns, 117, columns, 1016, 0]
        output = torch.empty(6, 2048, dtype=torch.int32, device="cuda")
        stable_topk(
            scores.cuda(),
            torch.tensor(starts, dtype=torch.int32, device="cuda"),
            torch.tensor(ends, dtype=torch.int32, device="cuda"),
            output,
        )
        # CPU lexicographic keys define the existing bit-level contract. The
        # candidate selects only a 32-bit threshold and resolves ties separately.
        expected = []
        col = torch.arange(columns)
        for row_bits, start, end in zip(bits, starts, ends):
            normalized = row_bits.clone()
            normalized[(normalized & 0x7FFFFFFF) == 0] = 0
            ordered = torch.where(normalized < 0, normalized ^ 0x7FFFFFFF, normalized)
            keys = (ordered.long() << 32) | (0xFFFFFFFF - col)
            keys[(col < start) | (col >= end)] = -9223372036854775808
            selected = torch.argsort(keys, descending=True)[:2048]
            selected = selected[keys[selected] != -9223372036854775808]
            indices = (selected - start).sort().values.tolist()
            expected.append(indices + [-1] * (2048 - len(indices)))
        self.assertEqual(output.cpu().tolist(), expected)

    def test_short_and_empty_rows_keep_minus_one_padding(self):
        op = self.make_op()
        for columns in (0, 17):
            scores = torch.zeros(3, columns, device="cuda")
            starts = torch.zeros(3, dtype=torch.int32, device="cuda")
            ends = torch.tensor(
                [0, min(1, columns), columns], dtype=torch.int32, device="cuda"
            )
            output = torch.empty(3, 2048, dtype=torch.int32, device="cuda")
            op._run_prefill_topk(scores, starts, ends, output, columns)
            self.assert_selection(output, [0] * 3, ends.cpu().tolist())

    def test_short_cp_batches_skip_logits_and_kv_assembly(self):
        op = self.make_op()
        starts = torch.tensor(
            [0, 0, 1000, 3000, 3000, 5000], device="cuda", dtype=torch.int32
        )
        ends = torch.tensor(
            [1, 999, 1100, 3005, 4999, 5000], device="cuda", dtype=torch.int32
        )
        offsets = starts.clone()
        ids = torch.arange(6, device="cuda", dtype=torch.int32)
        q = torch.empty(6, 1, 128, device="cuda")
        weights = torch.empty(6, 1, 1, device="cuda")
        params = SimpleNamespace(prefill_max_kv_len=2048)

        def forward():
            # Cache/table/CP assembly metadata are deliberately absent. A
            # short batch must return before accessing any of those inputs.
            return op._get_topk_ragged_cp(
                q,
                weights,
                None,
                params,
                None,
                ids,
                None,
                5000,
                starts,
                ends,
                ends - starts,
                offsets,
                kv_cache_sharded=True,
                cp_size=8,
                total_local_ids_is_identity=True,
            )

        def expected():
            columns = torch.arange(2048, device="cuda", dtype=torch.int32)
            return torch.where(
                columns[None, :] < (ends - starts)[:, None],
                columns[None, :] + offsets[:, None],
                -1,
            )

        with patch.object(
            indexer_op,
            "_fp8_mqa_logits_compat",
            side_effect=AssertionError("unneeded logits"),
        ):
            self.assertTrue(torch.equal(forward(), expected()))
            for _ in range(3):
                forward()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = forward()
            for _ in range(3):
                ends[1] -= 100
                offsets[2] += 1
                graph.replay()
                self.assertTrue(torch.equal(output, expected()))

    def test_short_rows_crossing_request_bases_match_score_selection(self):
        op = self.make_op()
        torch.manual_seed(2048)
        scores = torch.randn(5, 8192, device="cuda")
        starts = torch.tensor(
            [0, 1024, 2048, 4096, 8192], device="cuda", dtype=torch.int32
        )
        ends = torch.tensor(
            [0, 1041, 4096, 5120, 8192], device="cuda", dtype=torch.int32
        )
        full = torch.empty(5, 2048, dtype=torch.int32, device="cuda")
        fast = torch.empty_like(full)
        op._run_stable_topk(scores, starts, ends, full, 8192)
        op._run_stable_topk(scores, starts, ends, fast, 2048)
        self.assertTrue(torch.equal(full, fast))

    def test_cp_prefix_cache_and_native_bounds_at_topk_boundary(self):
        for prefix_a in (2044, 2045):
            with self.subTest(prefix_a=prefix_a):
                prefixes, lengths = [prefix_a, 8], [4, 4]
                kv_a = prefix_a + 4
                total_kv = kv_a + 12
                blocks = [list(range(33, 0, -1)), [0] * 33]
                inputs = PyAttentionInputs()
                inputs.is_prefill = True
                inputs.input_lengths = torch.tensor(lengths, dtype=torch.int32)
                inputs.prefix_lengths = torch.tensor(prefixes, dtype=torch.int32)
                inputs.sequence_lengths = torch.empty(0, dtype=torch.int32)
                inputs.context_total_kv_length = total_kv
                inputs.kv_cache_kernel_block_id = torch.tensor(
                    blocks, dtype=torch.int32
                )
                inputs.kv_cache_kernel_block_id_device = (
                    inputs.kv_cache_kernel_block_id.cuda()
                )
                inputs.cu_kv_seqlens_device = torch.tensor(
                    [0, kv_a, total_kv], dtype=torch.int32, device="cuda"
                )
                params = rtp_llm_ops.SparseMlaParams()
                params.fill_params(inputs, 64)
                self.assertEqual(params.prefill_max_kv_len, kv_a)
                op = indexer_op.IndexerOp(
                    32,
                    128,
                    2048,
                    0,
                    scale_fmt="none",
                    query_scale_fmt="ue8m0",
                    stable_topk=True,
                )
                cache = LayerKVCache(
                    torch.full((34, 64 * 132), 0xA5, dtype=torch.uint8, device="cuda"),
                    64,
                )
                for request, (prefix, length) in enumerate(zip(prefixes, lengths)):
                    for start, end in ((0, prefix), (prefix, prefix + length)):
                        slots = torch.tensor(
                            [
                                blocks[request][pos // 64] * 64 + pos % 64
                                for pos in range(start, end)
                            ],
                            dtype=torch.int64,
                            device="cuda",
                        )
                        # High-scoring new keys make the 2049-row selection
                        # differ from incorrectly truncating all indices at K.
                        keys = torch.full(
                            (end - start, 128),
                            2 if request == 0 and start == prefix else 1,
                            dtype=torch.bfloat16,
                            device="cuda",
                        )
                        op.quant_k_only(keys, cache, slots)
                cache_before = cache.kv_cache_base.clone()
                q = torch.ones(8, 32, 128, device="cuda").to(torch.float8_e4m3fn)
                weights = torch.ones(8, 32, 1, device="cuda")
                result = op._get_topk_ragged_cp(
                    q,
                    weights,
                    cache,
                    params,
                    inputs,
                    torch.arange(8, device="cuda"),
                    inputs.cu_kv_seqlens_device,
                    total_kv,
                    params.ks,
                    params.ke,
                    params.expanded_seq_lens,
                    params.topk_indices_offset,
                    total_local_ids_is_identity=True,
                )
                for row, visible in enumerate(
                    [prefix_a + i + 1 for i in range(4)] + [8 + i + 1 for i in range(4)]
                ):
                    count = min(visible, 2048)
                    expected = torch.full((2048,), -1, dtype=torch.int32, device="cuda")
                    indices = (
                        list(range(2048 - (visible - prefix_a)))
                        + list(range(prefix_a, visible))
                        if visible > 2048
                        else list(range(visible))
                    )
                    expected[:count] = torch.tensor(
                        indices, dtype=torch.int32, device="cuda"
                    )
                    self.assertTrue(torch.equal(result[row], expected))
                    if visible > 2048:
                        self.assertFalse(
                            torch.equal(
                                result[row],
                                torch.arange(2048, dtype=torch.int32, device="cuda"),
                            )
                        )
                self.assertTrue(torch.equal(cache.kv_cache_base, cache_before))


if __name__ == "__main__":
    main()
