"""Hopper packed-FP4 scoring against an independent BF16 rounding oracle."""

import json
import os
import unittest

import torch

from rtp_llm.models_py.modules.dsv41.fp8 import _v41_decode_topk as decode_topk
from rtp_llm.models_py.modules.dsv41.fp8 import _v41_prefill_candidates as candidate_ops
from rtp_llm.models_py.modules.dsv41.fp8 import _v41_prefill_topk as prefill_topk
from rtp_llm.models_py.modules.dsv41.fp8 import _v41_sparse_prefill_indexer as sparse
from rtp_llm.models_py.modules.dsv41.fp8._v41_prefill_indexer import (
    _fp4_rows_torch,
    score_indexer_chunk,
)
from rtp_llm.models_py.modules.dsv41.fp8._v41_sm90_indexer import (
    candidate_score,
    dense_score,
    paged_score,
)


def reference(q, keys, weights):
    dots = (q.double() @ keys.double().T).bfloat16().float().relu()
    weighted = (dots * weights.bfloat16().float()[:, None]).bfloat16()
    return weighted.float().sum(0).bfloat16().float()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class SM90IndexerTest(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() != (9, 0):
            self.skipTest("requires an SM90 GPU")

    def test_candidate_only_exact_dense_layout_cp_positions_and_ties(self):
        # Preserve the original dense-width topk dispatch and tie membership.
        # Query positions intentionally mimic disjoint CP zigzag halves.
        torch.manual_seed(4200)
        rows, width = 7, 1031
        q = torch.randn(rows, 32, 128, device="cuda", dtype=torch.bfloat16) * 0.2
        keys = torch.randn(width, 128, device="cuda", dtype=torch.bfloat16) * 0.2
        qp, qs, qr = _fp4_rows_torch(q)
        kp, ks, kr = _fp4_rows_torch(keys)
        qp, qs = qp.view(rows, 32, 64), qs.view(rows, 32)
        weights = torch.randn(rows, 32, device="cuda") / 64
        # Entirely tied scores, alongside random scores, test exact indices.
        weights[3].zero_()
        candidates = torch.full((rows, 128), -1, device="cuda", dtype=torch.int32)
        candidates[:, :80] = torch.arange(80, device="cuda", dtype=torch.int32)
        candidates[:, 80:84] = torch.tensor([128, 128, -1, 999], device="cuda")
        # One nonempty visible row deliberately has no valid candidate.
        candidates[2].fill_(-1)
        positions = torch.tensor(
            [1029, 0, 14, 1024, -1, 2061, 2059],
            device="cuda",
            dtype=torch.int64,
        )
        for ratio in (1, 2):
            visible = ((positions + 1) // ratio).clamp(0, width).int()
            plan = sparse.prepare_plan(
                candidates, positions, width, positions_ratio=ratio, sm90=True
            )
            self.assertIsNotNone(plan)
            actual = candidate_score(qp, qs, kp, ks, weights, plan)
            dense = dense_score(
                qp, qs, kp, ks, weights, torch.zeros_like(visible), visible
            )
            columns = torch.arange(width, device="cuda")
            allowed = (columns[None, :, None] // 8 == candidates[:, None]).any(-1)
            expected = dense.masked_fill(~allowed, -torch.inf)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            # Independent double-accumulation reference validates rounding;
            # exact equality above is the actual baseline acceptance gate.
            oracle = torch.stack(
                [
                    reference(
                        qr.view(rows, 32, 128)[row].bfloat16(),
                        kr.bfloat16(),
                        weights[row],
                    )
                    for row in range(rows)
                ]
            )
            oracle.masked_fill_(
                ~allowed | (columns[None] >= visible[:, None]), -torch.inf
            )
            torch.testing.assert_close(actual, oracle, rtol=0.01, atol=0.001)
            torch.testing.assert_close(
                prefill_topk.try_select_tokens(actual, visible),
                prefill_topk.try_select_tokens(expected, visible),
                rtol=0,
                atol=0,
            )

    @unittest.skipUnless(
        os.environ.get("DSV41_SM90_INDEXER_BENCH") == "1",
        "explicit 128K microbenchmark opt-in required",
    )
    def test_candidate_128k_benchmark(self):
        import triton

        torch.manual_seed(4201)
        rows, width, blocks = 512, 131072, 2048
        q = torch.randn(rows, 32, 128, device="cuda", dtype=torch.bfloat16) * 0.2
        keys = torch.randn(width, 128, device="cuda", dtype=torch.bfloat16) * 0.2
        qp, qs, _ = _fp4_rows_torch(q)
        kp, ks, _ = _fp4_rows_torch(keys)
        qp, qs = qp.view(rows, 32, 64), qs.view(rows, 32)
        weights = torch.rand(rows, 32, device="cuda") / 64
        candidates = torch.randperm(width // 8, device="cuda")[:blocks].int()
        candidates = candidates.expand(rows, -1).contiguous()
        visible = torch.full((rows,), width, device="cuda", dtype=torch.int32)
        starts = torch.zeros_like(visible)
        plan = sparse.prepare_plan(candidates, visible, width, sm90=True)
        flags = candidate_ops.build_flags(candidates, width, 8)
        self.assertIsNotNone(flags)

        def baseline():
            logits = dense_score(qp, qs, kp, ks, weights, starts, visible)
            self.assertTrue(candidate_ops.mask_candidates(logits, flags, 8))
            return logits

        def optimized():
            return candidate_score(qp, qs, kp, ks, weights, plan)

        torch.testing.assert_close(optimized(), baseline(), rtol=0, atol=0)
        results = {
            "rows": rows,
            "keys": width,
            "candidate_blocks": blocks,
            "baseline_score_mask_ms": triton.testing.do_bench(baseline),
            "candidate_score_dense_layout_ms": triton.testing.do_bench(optimized),
            "plan_prepare_ms": triton.testing.do_bench(
                lambda: sparse.prepare_plan(candidates, visible, width, sm90=True)
            ),
        }
        print("SM90_CANDIDATE_BENCH " + json.dumps(results), flush=True)

        if os.environ.get("DSV41_SM90_INDEXER_PROFILE") == "1":
            # Component-only trace, not model prefill/decode or end-to-end
            # timing. Keep compilation, allocations' first use, and all
            # correctness synchronization outside the captured region.
            for _ in range(3):
                baseline()
                warm_plan = sparse.prepare_plan(candidates, visible, width, sm90=True)
                candidate_score(qp, qs, kp, ks, weights, warm_plan)
            torch.cuda.synchronize()
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
            ) as profiler:
                for iteration in range(3):
                    with torch.profiler.record_function(
                        f"round1_indexer_component_ab/round_{iteration}"
                    ):
                        with torch.profiler.record_function("baseline_score_mask"):
                            baseline()
                        with torch.profiler.record_function("candidate_plan"):
                            profile_plan = sparse.prepare_plan(
                                candidates, visible, width, sm90=True
                            )
                        with torch.profiler.record_function("candidate_score"):
                            candidate_score(qp, qs, kp, ks, weights, profile_plan)
                torch.cuda.synchronize()
            output_dir = os.environ["TEST_UNDECLARED_OUTPUTS_DIR"]
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "round1_indexer_component_ab.json")
            profiler.export_chrome_trace(output_path)
            print("SM90_INDEXER_COMPONENT_TRACE " + output_path, flush=True)

    def _check_topk(self, actual, logits, lengths, *, decode=False, k=512):
        # Independent CPU selection; scores are unique except masked/nonfinite
        # entries. Compare sets because native TopK deliberately leaves order
        # unspecified, while decode short rows must be ascending even for NaN.
        scores = logits.cpu()
        ends = lengths.cpu().long().clamp(0, scores.shape[1])
        columns = torch.arange(scores.shape[1])
        masked = scores.masked_fill(columns[None] >= ends[:, None], -torch.inf)
        values, expected = masked.topk(k, dim=-1)
        expected = expected.masked_fill(~values.isfinite(), -1).int()
        got = actual.cpu()
        if decode:
            short = ends <= k
            prefix = torch.arange(k).expand(scores.shape[0], k)
            prefix = prefix.masked_fill(prefix >= ends[:, None], -1).int()
            expected = torch.where(short[:, None], prefix, expected)
            torch.testing.assert_close(got[short], prefix[short], rtol=0, atol=0)
        try:
            torch.testing.assert_close(
                got.sort(-1).values, expected.sort(-1).values, rtol=0, atol=0
            )
        except AssertionError as error:
            diagnostics = []
            for row in range(scores.shape[0]):
                valid = got[row][(got[row] >= 0) & (got[row] < scores.shape[1])]
                selected_scores = scores[row, valid.long()]
                wanted = expected[row][expected[row] >= 0]
                wanted_scores = scores[row, wanted.long()]
                diagnostics.append(
                    {
                        "row": row,
                        "length": int(ends[row]),
                        "valid_count": valid.numel(),
                        "unique_count": valid.unique().numel(),
                        "set_overlap": torch.isin(valid, wanted).sum().item(),
                        "index_head": got[row, :16].tolist(),
                        "score_head": selected_scores[:16].tolist(),
                        "score_range": (
                            [selected_scores.min().item(), selected_scores.max().item()]
                            if selected_scores.numel()
                            else []
                        ),
                        "expected_score_range": (
                            [wanted_scores.min().item(), wanted_scores.max().item()]
                            if wanted_scores.numel()
                            else []
                        ),
                    }
                )
            raise AssertionError(
                f"{error}\nTopK row diagnostics: {diagnostics}"
            ) from error

    def test_native_topk_prefill_decode_boundaries_and_nonfinite(self):
        # Full-model smoke first reached this native selector at 783 prompt
        # tokens. Include K-1/K/K+1, zero/negative/oversized lengths, and an
        # unaligned strided view: missing SM90 cubins must fail this test.
        rows, width = 9, 783
        generator = torch.Generator().manual_seed(4194)
        storage = torch.empty((rows, width + 7), device="cuda")
        logits = storage[:, 1 : width + 1]
        logits.copy_(
            torch.stack(
                [torch.randperm(width, generator=generator) for _ in range(rows)]
            ).float()
        )
        logits[:, 0] = torch.nan
        logits[:, 1] = torch.inf
        logits[:, 2] = -torch.inf
        lengths = torch.tensor(
            [0, 1, 511, 512, 513, width, width + 100, -1, width - 1],
            dtype=torch.int32,
            device="cuda",
        )
        self.assertTrue(prefill_topk.is_supported(logits, lengths))
        self.assertTrue(decode_topk.is_supported(logits, lengths, 512))
        selected = prefill_topk.try_select_tokens(logits, lengths)
        self.assertIsNotNone(selected)
        self._check_topk(selected, logits, lengths)
        self._check_topk(
            decode_topk.select_tokens(logits, lengths, 512),
            logits,
            lengths,
            decode=True,
        )

        # The producer can publish raw native indices and finish filtering in
        # a later epilogue. Keep this output-alias contract on SM90 as well.
        ends = lengths.clamp(0, width)
        bounds = (torch.zeros_like(ends), ends)
        out = torch.empty((rows, 512), device="cuda", dtype=torch.int32)
        raw = prefill_topk.try_select_tokens(
            logits, lengths, bounds=bounds, out=out, filter_finite=False
        )
        self.assertIs(raw, out)
        prefill_topk.finish_tokens(logits, ends, raw)
        self._check_topk(raw, logits, lengths)

    def test_native_topk_cluster_and_graph_replay(self):
        # Width > 16384 and batch <= 8 selects the real 8-CTA cluster kernel,
        # not just the small-row register kernel exercised by the smoke.
        rows, width = 3, 32768
        generator = torch.Generator().manual_seed(4195)
        logits = (
            torch.stack(
                [torch.randperm(width, generator=generator) for _ in range(rows)]
            )
            .float()
            .cuda()
        )
        lengths = torch.tensor([0, 513, width], device="cuda", dtype=torch.int32)

        def run():
            return (
                prefill_topk.try_select_tokens(logits, lengths),
                decode_topk.select_tokens(logits, lengths, 512),
            )

        # Validate the same cluster dispatch eagerly before capture, then
        # compare both replay and eager execution on changed inputs. subTest
        # preserves every assertion while collecting all phases in one run.
        before = run()  # Also warm workspace and finite-filter kernels.
        for phase, selected in zip(("prefill", "decode"), before):
            with self.subTest(execution="eager_before", phase=phase):
                self.assertIsNotNone(selected)
                self._check_topk(selected, logits, lengths, decode=phase == "decode")
        original_lengths = lengths.clone()
        lengths.fill_(width)
        for phase, selected in zip(("prefill", "decode"), run()):
            with self.subTest(execution="eager_all_long_positive", phase=phase):
                self.assertIsNotNone(selected)
                self._check_topk(selected, logits, lengths, decode=phase == "decode")
        lengths.copy_(original_lengths)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            prefill, decode = run()
        lengths.copy_(
            torch.tensor([512, width, width - 17], device="cuda", dtype=torch.int32)
        )
        logits.mul_(-1)
        graph.replay()
        for phase, selected in zip(("prefill", "decode"), (prefill, decode)):
            with self.subTest(execution="graph_replay", phase=phase):
                self.assertIsNotNone(selected)
                self._check_topk(selected, logits, lengths, decode=phase == "decode")
        after = run()
        for phase, selected in zip(("prefill", "decode"), after):
            with self.subTest(execution="eager_after", phase=phase):
                self.assertIsNotNone(selected)
                self._check_topk(selected, logits, lengths, decode=phase == "decode")
        lengths.fill_(width)
        for phase, selected in zip(("prefill", "decode"), run()):
            with self.subTest(execution="eager_all_long_negative", phase=phase):
                self.assertIsNotNone(selected)
                self._check_topk(selected, logits, lengths, decode=phase == "decode")

    def test_native_topk_cluster4_and_cluster2_output_sinks(self):
        # At 32769 columns the public dispatcher selects Cluster4 for batch13
        # and Cluster2 for batch40. The persistent dispatch branch is currently
        # unreachable; do not change routing just to claim test coverage.
        width = 32769
        generator = torch.Generator().manual_seed(4196)
        for rows, cluster_size in ((13, 4), (40, 2)):
            logits = (
                torch.stack(
                    [torch.randperm(width, generator=generator) for _ in range(rows)]
                )
                .float()
                .cuda()
            )
            lengths = torch.full((rows,), width, device="cuda", dtype=torch.int32)
            for negative in (False, True):
                if negative:
                    logits.neg_()
                for phase in ("prefill", "decode"):
                    with self.subTest(
                        cluster_size=cluster_size, negative=negative, phase=phase
                    ):
                        selected = (
                            prefill_topk.try_select_tokens(logits, lengths)
                            if phase == "prefill"
                            else decode_topk.select_tokens(logits, lengths, 512)
                        )
                        self.assertIsNotNone(selected)
                        self._check_topk(
                            selected, logits, lengths, decode=phase == "decode"
                        )

    def test_dense_prefill_causal_bounds_and_fp4_rounding(self):
        torch.manual_seed(4192)
        q = torch.randn(3, 32, 128, device="cuda", dtype=torch.bfloat16) * 0.2
        keys = torch.randn(137, 128, device="cuda", dtype=torch.bfloat16) * 0.2
        qp, qs, qr = _fp4_rows_torch(q)
        kp, ks, kr = _fp4_rows_torch(keys)
        weights = torch.randn(3, 32, device="cuda") / 64
        starts = torch.tensor([0, 17, 0], device="cuda", dtype=torch.int32)
        ends = torch.tensor([0, 79, 137], device="cuda", dtype=torch.int32)
        actual = score_indexer_chunk(
            qp.view(3, 32, 64),
            qs.view(3, 32),
            kp,
            ks,
            weights,
            ends,
            bounds=(starts, ends),
        )
        expected = torch.stack(
            [
                reference(qr.view(3, 32, 128)[i].bfloat16(), kr.bfloat16(), weights[i])
                for i in range(3)
            ]
        )
        columns = torch.arange(137, device="cuda")
        visible = (columns[None] >= starts[:, None]) & (columns[None] < ends[:, None])
        expected.masked_fill_(~visible, -torch.inf)
        torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.001)

    def test_paged_half_pages_zero_sentinel_and_graph(self):
        for capacity in (137, 1025, 16385):
            with self.subTest(capacity=capacity):
                self._check_paged_half_pages_zero_sentinel_and_graph(capacity)

    def _check_paged_half_pages_zero_sentinel_and_graph(self, capacity):
        torch.manual_seed(4193)
        batch, sequence, heads = 2, 2, 32
        physical, logical = 128, 64
        pages_per_request = (capacity + logical - 1) // logical
        page_count = 1 + batch * pages_per_request
        table = torch.arange(1, page_count, device="cuda", dtype=torch.int32).reshape(
            batch, pages_per_request
        )
        table[0] = table[0].flip(0)
        table[1, 1] = 0
        pool = torch.full(
            (page_count, physical, 68), 255, device="cuda", dtype=torch.uint8
        )
        keys = (
            torch.randn(batch, capacity, 128, device="cuda", dtype=torch.bfloat16) * 0.2
        )
        kp, ks, kr = _fp4_rows_torch(keys)
        columns = torch.arange(capacity, device="cuda")
        pages = table[:, columns // logical].long()
        offsets = columns.remainder(logical).expand(batch, -1)
        live = pages > 0
        raw = pool.view(page_count, physical * 68)
        payload = raw[:, : physical * 64].view(page_count, physical, 64)
        scales = raw[:, physical * 64 :].view(torch.int32)
        payload[pages[live], offsets[live]] = kp.view(batch, capacity, 64)[live].view(
            torch.uint8
        )
        scales[pages[live], offsets[live]] = ks.view(batch, capacity)[live]
        kr = kr.view(batch, capacity, 128).bfloat16()
        kr[~live] = 0
        q = (
            torch.randn(
                batch, sequence, heads, 128, device="cuda", dtype=torch.bfloat16
            )
            * 0.2
        )
        qp, qs, qr = _fp4_rows_torch(q)
        qp, qs = qp.view(batch, sequence, heads, 64), qs.view(batch, sequence, heads)
        weights = torch.randn(batch, sequence, heads, device="cuda") / 64
        lengths = torch.tensor(
            [[0, 73], [129, capacity]], device="cuda", dtype=torch.int32
        )

        def run():
            return paged_score(qp, qs, pool, weights, table, lengths, capacity, logical)

        def check(actual):
            qref = qr.view(batch, sequence, heads, 128).bfloat16()
            expected = torch.stack(
                [
                    reference(qref[b, s], kr[b], weights[b, s])
                    for b in range(batch)
                    for s in range(sequence)
                ]
            )
            expected.masked_fill_(
                columns[None] >= lengths.flatten()[:, None], -torch.inf
            )
            torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.001)

        check(run())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run()
        lengths.copy_(
            torch.tensor([[capacity, 27], [0, 81]], device="cuda", dtype=torch.int32)
        )
        table[1, 1] = -1  # Negative and zero page sentinels both represent zero K.
        graph.replay()
        check(captured)


if __name__ == "__main__":
    unittest.main()
