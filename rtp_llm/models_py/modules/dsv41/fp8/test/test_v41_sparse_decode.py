"""Sparse decode gating, candidate holes, paged selection and graph replay.

The dense GPU oracle uses the same BF16 head reduction/logits as the sparse
kernel. This tests indexing exactly, not equality to the old FP32 arithmetic.
"""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import torch

from rtp_llm.models_py.modules.dsv41.fp8 import _v41_decode_indexer as indexer
from rtp_llm.models_py.modules.dsv41.fp8 import _v41_decode_topk as topk
from rtp_llm.models_py.modules.dsv41.fp8 import _v41_sparse_decode as sparse
from rtp_llm.models_py.modules.dsv41.fp8._indexer_score import (
    fp8_fp4_paged_indexer_score,
)
from rtp_llm.models_py.modules.dsv41.fp8.attention_v41 import AttentionV41FP8
from rtp_llm.models_py.modules.dsv41.fp8.test.test_v41_decode_attention_fusion import (
    _fixture,
    _original_paged_score,
    _select,
)


class SparseDecodeTopKRoutingCPU(unittest.TestCase):
    def setUp(self):
        self.logits = torch.empty(2, 4096, dtype=torch.bfloat16)
        self.plan = sparse.SparseDecodePlan(
            torch.empty(2, 512, dtype=torch.int32),
            torch.empty(2, 512, dtype=torch.bool),
            torch.empty(2, 64, dtype=torch.int32),
            torch.empty(2, dtype=torch.int32),
            None,
        )
        for name in ("_clean_logits_kernel", "_restore_indices_kernel"):
            mocked = patch.object(sparse, name, MagicMock())
            mocked.start()
            self.addCleanup(mocked.stop)
        workspace = patch.object(sparse, "_get_topk_workspace", return_value=None)
        workspace.start()
        self.addCleanup(workspace.stop)

    def test_deepselect_default_switch_and_fallback(self):
        for switch, available in ((None, True), ("1", True), ("0", True), (None, False)):
            with self.subTest(switch=switch, available=available), patch.dict(os.environ):
                os.environ.pop("DSV41_DECODE_DEEPSELECT", None)
                if switch is not None:
                    os.environ["DSV41_DECODE_DEEPSELECT"] = switch

                def select(scores, lengths, *, out, filter_finite, enabled):
                    self.assertEqual(scores.dtype, torch.float32)
                    self.assertEqual(scores.shape, (2, 4096))
                    self.assertEqual(lengths.shape, (2,))
                    self.assertFalse(filter_finite)
                    self.assertEqual(enabled, switch != "0")
                    return out if enabled and available else None

                with patch.object(
                    sparse.deepselect, "try_select_tokens", side_effect=select
                ) as native, patch.object(
                    sparse.rtp_llm_ops, "topk_v3"
                ) as fallback:
                    output = sparse.select_tokens(self.logits, self.plan)
                native.assert_called_once()
                self.assertIs(native.call_args.kwargs["out"], output)
                self.assertEqual(fallback.call_count, int(switch == "0" or not available))
                if fallback.called:
                    self.assertIs(fallback.call_args.args[2], output)
                restore = sparse._restore_indices_kernel.__getitem__.return_value
                self.assertIs(restore.call_args.args[0], output)

    def test_deepselect_errors_do_not_fall_back(self):
        with patch.object(
            sparse.deepselect, "try_select_tokens", side_effect=RuntimeError("DeepSelect failed")
        ), patch.object(sparse.rtp_llm_ops, "topk_v3") as fallback:
            with self.assertRaisesRegex(RuntimeError, "DeepSelect failed"):
                sparse.select_tokens(self.logits, self.plan)
        fallback.assert_not_called()
        sparse._restore_indices_kernel.__getitem__.assert_not_called()


class SparseDecodeSupportCPU(unittest.TestCase):
    def test_backend_and_candidate_contract(self):
        candidates = SimpleNamespace(
            is_cuda=True,
            ndim=2,
            dtype=torch.int32,
            shape=(6, 512),
            device=torch.device("cuda"),
            is_contiguous=lambda: True,
        )
        backend = SimpleNamespace(
            get_paged_sparse_mqa_logits_metadata=Mock(),
            fp8_fp4_paged_sparse_mqa_logits=Mock(),
        )
        with patch.object(sparse, "deep_gemm", backend), patch.object(
            sparse, "rtp_llm_ops", SimpleNamespace(topk_v3=Mock())
        ), patch.object(torch.cuda, "get_device_capability", return_value=(10, 3)):
            for k in (8, 511, 512, 513, 1024, 2048, 2049):
                with self.subTest(k=k):
                    candidates.shape = (6, k)
                    self.assertEqual(
                        sparse.is_supported(candidates), k in (512, 1024, 2048)
                    )
            candidates.shape = (6, 512)
            for attr, value in (
                ("is_cuda", False),
                ("ndim", 1),
                ("dtype", torch.int64),
                ("is_contiguous", lambda: False),
            ):
                with self.subTest(attr=attr), patch.object(candidates, attr, value):
                    self.assertFalse(sparse.is_supported(candidates))
            for name in vars(backend):
                with self.subTest(api=name), patch.object(backend, name, None):
                    self.assertFalse(sparse.is_supported(candidates))
            with patch.object(sparse, "deep_gemm", None):
                self.assertFalse(sparse.is_supported(candidates))
            with patch.object(sparse.rtp_llm_ops, "topk_v3", None):
                self.assertFalse(sparse.is_supported(candidates))
            with patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)):
                self.assertFalse(sparse.is_supported(candidates))
            backend.get_paged_sparse_mqa_logits_metadata.assert_not_called()
            backend.fp8_fp4_paged_sparse_mqa_logits.assert_not_called()

    def test_unsupported_path_preserves_dense_selection(self):
        attn, _, x, qr, positions = _fixture("cpu", 1, candidate=True)
        with patch.object(
            indexer, "score_decode_indexer", side_effect=_original_paged_score
        ):
            _select(attn, x, qr, positions)
            attn.layer_id = 24
            expected = _select(attn, x, qr, positions)
            attn.decode_sparse_mqa = True
            with patch.object(sparse, "make_plan") as plan:
                got = _select(attn, x, qr, positions)
            plan.assert_not_called()
            torch.testing.assert_close(got, expected, rtol=0, atol=0)

    def test_kernel_errors_are_not_hidden_by_dense_fallback(self):
        attn, _, x, qr, positions = _fixture("cpu", 1, candidate=True)
        attn.layer_id = 24
        attn.decode_sparse_mqa = True
        attn._shared_attention["candidates"] = torch.zeros(12, 512, dtype=torch.int32)
        with patch.object(sparse, "is_supported", return_value=True), patch.object(
            sparse, "make_plan", side_effect=RuntimeError("sparse kernel failed")
        ), patch.object(indexer, "score_decode_indexer") as dense:
            with self.assertRaisesRegex(RuntimeError, "sparse kernel failed"):
                _select(attn, x, qr, positions)
        dense.assert_not_called()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class SparseDecodeCUDA(unittest.TestCase):
    def setUp(self):
        candidates = torch.empty(1, 512, device="cuda", dtype=torch.int32)
        if not sparse.is_supported(candidates):
            self.skipTest("SM10x and DeepGEMM paged sparse MQA required")
        self.context = torch.inference_mode()
        self.context.__enter__()
        self.addCleanup(self.context.__exit__, None, None, None)

    def _check_plan(self, plan, candidates, lengths, table, query_len):
        torch.testing.assert_close(
            plan.table,
            table.repeat_interleave(query_len, 0).clamp_min(0),
            rtol=0,
            atol=0,
        )
        for row, length in enumerate(lengths.tolist()):
            expected = min((length + 7) // 8, candidates.shape[1])
            prefix = plan.blocks[row, :expected]
            self.assertEqual(prefix.unique().numel(), expected)
            self.assertTrue(bool((prefix[1:] > prefix[:-1]).all()))
            self.assertTrue(bool(((prefix >= 0) & (prefix < (length + 7) // 8)).all()))
            wanted = candidates[row][candidates[row] >= 0].sort().values
            torch.testing.assert_close(
                plan.blocks[row][plan.active[row]], wanted, rtol=0, atol=0
            )
            self.assertFalse(bool(plan.active[row, expected:].any()))

    def _check_selection(self, result, candidates, lengths, dense):
        for row, length in enumerate(lengths.tolist()):
            blocks = candidates[row][candidates[row] >= 0].long()
            allowed = (blocks[:, None] * 8 + torch.arange(8, device="cuda")).flatten()
            allowed = allowed[allowed < length].sort().values
            selected = result[row][result[row] >= 0].long()
            values = dense[row, allowed]
            finite = values.isfinite()
            allowed, values = allowed[finite], values[finite]
            self.assertEqual(selected.unique().numel(), selected.numel())
            self.assertTrue(
                bool(torch.isin(selected, allowed).all()), "candidate/filler leak"
            )
            self.assertEqual(selected.numel(), min(512, allowed.numel()))
            # Tie identities/order may differ; selected score multisets must not.
            expected = values.topk(min(512, values.numel())).values.sort().values
            torch.testing.assert_close(
                dense[row, selected].sort().values, expected, rtol=0, atol=0
            )
            if length <= 512:
                torch.testing.assert_close(selected, allowed, rtol=0, atol=0)

    def _paged_case(self, batch, capacity):
        from rtp_llm.models_py.modules.dsv41.fp8._v41_fp4_triton import (
            quantize_indexer_k_fp4,
        )

        cols = capacity // 128
        # Padded request rows and shuffled physical pages must both survive B*S expansion.
        storage = torch.zeros(batch, cols + 3, device="cuda", dtype=torch.int32)
        table = storage[:, :cols]
        table.copy_(
            (torch.randperm(batch * cols, device="cuda", dtype=torch.int32) + 1).view(
                batch, cols
            )
        )
        table[-1, 0] = -1
        table[-1, 1] = 0
        pool = torch.zeros(batch * cols + 1, 128, 68, device="cuda", dtype=torch.uint8)
        pos = torch.arange(capacity, device="cuda")
        for request in range(batch):
            keys = torch.randn(capacity, 128, device="cuda", dtype=torch.bfloat16)
            slots = table[request, pos // 128].clamp_min(0).long() * 128 + pos % 128
            quantize_indexer_k_fp4(keys, slots, pool)
        # Page zero contains real nonzero bytes: the adapter, not zero storage,
        # must implement RTP's unallocated-page semantics.
        return table, pool

    def test_candidate_holes_match_dense_and_graph_replay(self):
        from rtp_llm.models_py.modules.dsv41.fp8._v41_fp4_triton import quantize_rows_fp4

        dg = sparse.deep_gemm
        for k in (512, 1024, 2048):
            for query_len in (1, 3):
                with self.subTest(k=k, query_len=query_len):
                    torch.manual_seed(93 + k + query_len)
                    rows, capacity = 12, 32768
                    table, pool = self._paged_case(rows // query_len, capacity)
                    lengths = torch.tensor(
                        [0, 1, 7, 9, 511, 512, 513, 8191, 16383, 16384, 16385, 32761],
                        device="cuda",
                        dtype=torch.int32,
                    )
                    source = torch.randn(rows, capacity, device="cuda")
                    source[3::3] = -torch.inf
                    source[1::3, : capacity // 2] = torch.nan
                    source[2::3, 8:capacity:16] = torch.nan
                    source[2::3, 9:capacity:16] = -torch.inf
                    candidates, _ = topk.select_candidates(source, lengths, 8, k)
                    candidates[7, ::2] = -1
                    candidates[9] = -1
                    plan = sparse.make_plan(
                        candidates, lengths, table, query_len=query_len
                    )
                    self._check_plan(plan, candidates, lengths, table, query_len)
                    query = quantize_rows_fp4(
                        torch.randn(
                            rows, 1, 32, 128, device="cuda", dtype=torch.bfloat16
                        )
                    )
                    weights = (torch.randn(rows, 32, device="cuda") / 64).bfloat16()
                    expanded = (
                        table.repeat_interleave(query_len, 0).clamp_min(0).contiguous()
                    )
                    dense = fp8_fp4_paged_indexer_score(
                        query[0],
                        query[1],
                        pool,
                        weights,
                        expanded,
                        lengths[:, None],
                        128,
                        capacity,
                        return_bf16=True,
                    ).float()
                    pos = torch.arange(capacity, device="cuda")
                    dense = torch.where(expanded[:, pos // 128] > 0, dense, 0)

                    def chain():
                        current = sparse.make_plan(
                            candidates, lengths, table, query_len=query_len
                        )
                        logits = dg.fp8_fp4_paged_sparse_mqa_logits(
                            query,
                            pool.unsqueeze(2),
                            weights,
                            current.metadata,
                            k,
                            8,
                        )
                        return sparse.select_tokens(logits, current)

                    self._check_selection(chain(), candidates, lengths, dense)
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        chain()
                    torch.cuda.current_stream().wait_stream(stream)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        result = chain()
                    for mode in ("original", "more_holes", "empty"):
                        if mode == "more_holes":
                            candidates[:, 1::2] = -1
                        elif mode == "empty":
                            candidates.fill_(-1)
                        graph.replay()
                        self._check_selection(result, candidates, lengths, dense)

    def test_nonfinite_logits_are_filtered(self):
        k, length = 512, 8192
        candidates = torch.arange(k, device="cuda", dtype=torch.int32)[None] * 2
        lengths = torch.tensor([length], device="cuda", dtype=torch.int32)
        table = torch.ones(1, length // 128, device="cuda", dtype=torch.int32)
        plan = sparse.make_plan(candidates, lengths, table, query_len=1)
        raw = torch.arange(k * 8, device="cuda").bfloat16()[None]
        raw[:, ::7] = torch.nan
        raw[:, 1::7] = torch.inf
        raw[:, 2::7] = -torch.inf
        result = sparse.select_tokens(raw, plan)
        selected = result[result >= 0].long()
        compact = (selected // 16) * 8 + selected % 8
        self.assertTrue(bool(raw[0, compact].isfinite().all()))
        self.assertEqual(selected.unique().numel(), selected.numel())
        # Compare to the existing dense adapter's post-TopK finite filtering,
        # rather than introducing a new "filter first" numerical contract.
        reference = topk.select_tokens(
            raw.float(), torch.tensor([k * 8], device="cuda", dtype=torch.int32), 512
        )
        reference = reference[reference >= 0].long()
        torch.testing.assert_close(
            raw[0, compact].sort().values,
            raw[0, reference].sort().values,
            rtol=0,
            atol=0,
        )

    def test_deepselect_and_fallback_masked_selection_graph(self):
        if not sparse.deepselect.is_available(torch.device("cuda"), enabled=True):
            self.skipTest("Native FP32 DeepSelect required")
        for k in (512, 1024, 2048):
            capacity = k * 16
            lengths = torch.tensor(
                [0, 1, 511, 512, 513, capacity - 1, capacity, capacity],
                device="cuda", dtype=torch.int32,
            )
            rows = lengths.numel()
            candidates = (
                torch.arange(k, device="cuda", dtype=torch.int32)[None].repeat(rows, 1) * 2
            )
            candidates.masked_fill_(candidates * 8 >= lengths[:, None], -1)
            candidates[4:, 1::3] = -1
            candidates[-1] = -1
            table = torch.ones(rows, capacity // 128, device="cuda", dtype=torch.int32)
            table[-2, 0] = 0
            plan = sparse.make_plan(candidates, lengths, table, query_len=1)
            tokens = plan.blocks[:, :, None].long() * 8 + torch.arange(8, device="cuda")
            raw = tokens.reshape(rows, k * 8).bfloat16()
            dense = (
                torch.arange(capacity, device="cuda").bfloat16().float()[None].repeat(rows, 1)
            )
            dense[-2, :128] = 0  # Unallocated page, regardless of raw logits.
            for switch in ("0", "1"):
                with self.subTest(k=k, deepselect=switch), patch.dict(
                    os.environ, {"DSV41_DECODE_DEEPSELECT": switch}
                ), patch.object(
                    sparse.rtp_llm_ops, "topk_v3", wraps=sparse.rtp_llm_ops.topk_v3
                ) as fallback:
                    self._check_selection(
                        sparse.select_tokens(raw, plan), candidates, lengths, dense
                    )
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        sparse.select_tokens(raw, plan)
                    torch.cuda.current_stream().wait_stream(stream)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        result = sparse.select_tokens(raw, plan)
                    self.assertEqual(fallback.call_count, 3 if switch == "0" else 0)
                    for sign in (1, -1):
                        raw.copy_(tokens.reshape(rows, k * 8).bfloat16() * sign)
                        graph.replay()
                        self._check_selection(result, candidates, lengths, dense * sign)
                    raw.copy_(tokens.reshape(rows, k * 8).bfloat16())

    def test_missing_backend_and_unsupported_k_keep_dense_path(self):
        attn, _, x, qr, positions = _fixture("cuda", 1, candidate=True)
        with patch.object(
            indexer, "score_decode_indexer", side_effect=_original_paged_score
        ):
            _select(attn, x, qr, positions)
            attn.layer_id = 24
            expected = _select(attn, x, qr, positions)
            attn.decode_sparse_mqa = True
            with patch.object(sparse, "make_plan") as plan:
                got = _select(attn, x, qr, positions)
            plan.assert_not_called()  # K=8: same regression as candidate publication.
            torch.testing.assert_close(
                got.sort(-1).values, expected.sort(-1).values, rtol=0, atol=0
            )
            # Pad IDs to a supported width so missing APIs, not K, reject sparse.
            candidates = attn._shared_attention["candidates"]
            padded = torch.full((12, 512), -1, device="cuda", dtype=torch.int32)
            padded[:, : candidates.shape[1]] = candidates
            attn._shared_attention["candidates"] = padded
            _, flags, block_size = attn._shared_attention["candidate_mask"]
            attn._shared_attention["candidate_mask"] = (padded, flags, block_size)
            for api in (
                "get_paged_sparse_mqa_logits_metadata",
                "fp8_fp4_paged_sparse_mqa_logits",
            ):
                with self.subTest(api=api), patch.object(
                    sparse.deep_gemm, api, None
                ), patch.object(sparse, "make_plan") as plan:
                    got = _select(attn, x, qr, positions)
                    plan.assert_not_called()
                    # Backend rejection must preserve the existing dense route.
                    torch.testing.assert_close(
                        got.sort(-1).values, expected.sort(-1).values, rtol=0, atol=0
                    )

    def test_reindex_layers_share_plan_and_refresh_on_graph_replay(self):
        torch.manual_seed(31415)
        batch, query_len, capacity = 2, 3, 32768
        rows = batch * query_len
        table, pool = self._paged_case(batch, capacity)
        keys = indexer.DecodeIndexerKeys(pool, table, capacity, 128)
        shared = {"layers": {0: None}}
        first_layer = SimpleNamespace(layer_id=0, _shared_attention=shared)
        x = torch.randn(batch, query_len, 32, device="cuda", dtype=torch.bfloat16)
        qr = torch.randn(batch, query_len, 16, device="cuda", dtype=torch.bfloat16)
        positions = torch.tensor(
            [20000, 20001, 20002, 30000, 30001, 30002], device="cuda"
        )
        source = torch.randn(rows, capacity, device="cuda")
        frequencies = torch.polar(
            torch.ones(capacity, 32, device="cuda"),
            torch.randn(capacity, 32, device="cuda"),
        )
        layers = []
        for layer_id in (24, 28, 32, 36):
            layer = AttentionV41FP8.__new__(AttentionV41FP8)
            torch.nn.Module.__init__(layer)
            layer.layer_id = layer.index_source_layer_id = layer_id
            layer.kv_source_layer_id = 20
            layer.is_index_source = layer.decode_sparse_mqa = True
            layer.compress_ratio = 1
            layer.index_n_heads, layer.index_head_dim, layer.index_topk = 32, 128, 512
            layer.rope_head_dim = 64
            layer._shared_attention = shared
            layer.v41_config = dict(
                candidate_source_layer_id=20,
                candidate_topk_blocks=2048,
                candidate_block_size=8,
            )
            layer.index_wq = torch.nn.Linear(
                16, 4096, bias=False, device="cuda", dtype=torch.bfloat16
            )
            layer.index_weights = torch.randn(
                32, 32, device="cuda", dtype=torch.bfloat16
            )
            layer.freqs_cis = frequencies
            layers.append(layer)

        def forward():
            AttentionV41FP8._begin_forward(first_layer)
            self.assertNotIn("decode_sparse_plan", shared)
            shared["global"] = {20: keys}
            candidates, flags = topk.select_candidates(
                source, (positions + 1).int(), 8, 2048
            )
            shared["candidates"] = candidates
            shared["candidate_mask"] = (candidates, flags, 8)
            outputs, plan = [], None
            for layer in layers:
                outputs.append(layer._select_indices_decode(x, qr, positions))
                current = shared["decode_sparse_plan"]
                if plan is not None:
                    self.assertIs(current, plan)
                plan = current
            return outputs, candidates

        def check(outputs, candidates):
            lengths = (positions + 1).int()
            expanded = table.repeat_interleave(query_len, 0).clamp_min(0).contiguous()
            columns = torch.arange(capacity, device="cuda")
            for layer, selected in zip(layers, outputs):
                q = layer.index_wq(qr.reshape(rows, 16)).view(batch, query_len, 32, 128)
                weights = torch.nn.functional.linear(x, layer.index_weights)
                payload, scales, scaled = indexer.prepare_indexer_q_and_weights(
                    q, weights, frequencies, positions, 64,
                    weights_out_dtype=torch.bfloat16,
                )
                dense = fp8_fp4_paged_indexer_score(
                    payload.reshape(rows, 1, 32, 64),
                    scales.reshape(rows, 1, 32),
                    pool,
                    scaled.reshape(rows, 32),
                    expanded,
                    lengths[:, None],
                    128,
                    capacity,
                    return_bf16=True,
                ).float()
                dense = torch.where(expanded[:, columns // 128] > 0, dense, 0)
                self._check_selection(selected, candidates, lengths, dense)

        with patch.object(
            indexer,
            "score_decode_indexer",
            side_effect=AssertionError("unexpected dense fallback"),
        ):
            with patch.object(sparse, "make_plan", wraps=sparse.make_plan) as planner:
                check(*forward())
                self.assertEqual(planner.call_count, 1)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                forward()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                outputs, candidates = forward()
            for delta in (1, 17, 3):
                positions.add_(delta)
                source.normal_()
                if delta == 17:
                    source[:, :28000] = torch.nan
                elif delta == 3:
                    source.fill_(-torch.inf)
                qr.normal_()
                graph.replay()
                check(outputs, candidates)


if __name__ == "__main__":
    unittest.main()
