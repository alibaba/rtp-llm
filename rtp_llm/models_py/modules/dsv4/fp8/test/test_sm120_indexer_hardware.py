"""SM120 hardware gate for indexer prefill/decode and topk_v3."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from rtp_llm.models_py.modules.dsv4 import _indexer_score_triton
from rtp_llm.models_py.modules.dsv4.fp8 import indexer
from rtp_llm.models_py.modules.dsv4.fp8._indexer_q_quant_triton import (
    indexer_q_fp8_quant_fold,
)
from rtp_llm.models_py.modules.dsv4.fp8._indexer_quant_triton import (
    INDEXER_ENTRY_BYTES,
    INDEXER_HEAD_DIM,
    dequantize_indexer_k,
    quantize_indexer_k,
)
from rtp_llm.models_py.modules.dsv4.fp8._indexer_score import (
    fp8_mqa_indexer_score,
    fp8_paged_indexer_score,
    has_fp8_mqa_logits,
    has_fp8_paged_mqa_logits,
)
from rtp_llm.models_py.modules.dsv4.fp8.attention import AttentionFP8
from rtp_llm.models_py.utils.arch import is_sm120
from rtp_llm.ops.compute_ops import rtp_llm_ops


class Sm120IndexerHardwareTest(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("SM120 hardware gate was scheduled without a CUDA device")
        if not is_sm120():
            self.skipTest(
                "SM120 hardware gate was scheduled on "
                f"compute capability {torch.cuda.get_device_capability()}"
            )
        self.device = torch.device("cuda", torch.cuda.current_device())
        torch.manual_seed(23)

    def test_paged_prefill_maps_multiple_requests_and_invalid_slots(self) -> None:
        from rtp_llm.models_py.modules.dsv4.fp8._sm120_prefill_indices import (
            split_sm120_combined_tables,
        )

        combined = torch.tensor(
            [
                [0, 2, 3, -1, -1],
                [1, 4, 5, -1, -1],
                [6, 8, 9, 10, -1],
                [11, -1, -1, -1, -1],
            ],
            dtype=torch.int32,
            device=self.device,
        )
        lengths = torch.tensor([3, 3, 4, 1], dtype=torch.int32, device=self.device)
        (
            swa_indices,
            swa_lens,
            extra_indices,
            extra_lens,
            extra_page_size,
        ) = split_sm120_combined_tables(
            combined,
            lengths,
            M=6,
            N=2,
            window_size=3,
            extra_width=2,
            ratio=4,
            device=self.device,
        )
        self.assertEqual(extra_page_size, 64)
        torch.testing.assert_close(
            extra_lens,
            torch.tensor([1, 1, 1, 0], dtype=torch.int32, device=self.device),
            rtol=0,
            atol=0,
        )
        expected_extra = torch.zeros(4, 64, dtype=torch.int32, device=self.device)
        expected_extra[0, 0] = 0
        expected_extra[1, 0] = 1
        expected_extra[2, 0] = 2
        self.assertEqual(extra_indices.shape, expected_extra.shape)
        torch.testing.assert_close(extra_indices, expected_extra, rtol=0, atol=0)
        torch.testing.assert_close(
            swa_lens,
            torch.tensor([2, 2, 3, 1], dtype=torch.int32, device=self.device),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            swa_indices,
            torch.tensor(
                [[0, 1, 0], [2, 3, 0], [4, 5, 6], [7, 0, 0]],
                dtype=torch.int32,
                device=self.device,
            ),
            rtol=0,
            atol=0,
        )

    def test_paged_prefill_ring_residency_uses_real_slot_producer(self) -> None:
        from rtp_llm.models_py.modules.dsv4.fp8 import attention
        from rtp_llm.models_py.modules.dsv4.fp8._swa_ops_triton import (
            compute_swa_slot_mapping,
        )
        from rtp_llm.models_py.modules.dsv4.kv_cache_utils import HCA_KV, SWA_KV

        tokens, ring, dim = 256, 128, 512
        bt = torch.tensor([[1, 2]], dtype=torch.int32, device=self.device)
        qsl = torch.tensor([0, tokens], dtype=torch.int32, device=self.device)
        lengths = torch.tensor([tokens], dtype=torch.int32, device=self.device)
        zeros = torch.zeros(1, dtype=torch.int32, device=self.device)
        for block_span in (128, 2048):
            with self.subTest(block_span=block_span):
                slots = compute_swa_slot_mapping(
                    bt,
                    qsl,
                    lengths,
                    tokens,
                    pool_entries_per_block=ring,
                    tokens_per_block_for_block_table=block_span,
                    ring_entries=ring,
                )
                obj = SimpleNamespace(
                    _kv_cache=object(),
                    _block_tables_by_type={SWA_KV: bt, HCA_KV: bt},
                    compress_ratio=128,
                    window_size=128,
                    head_dim=dim,
                    dim=dim,
                    _swa_entries_per_block=lambda: ring,
                    _pool_entries_per_block=lambda *_: ring,
                    _swa_cp_byte_sliced=lambda: False,
                    _build_swa_cp_byte_compaction=lambda *a, **k: None,
                )
                with mock.patch.object(
                    attention, "_dsv4_pool_tokens_per_block", return_value=block_span
                ):
                    wm = AttentionFP8._build_workspace_meta(
                        obj,
                        tokens,
                        0,
                        self.device,
                        True,
                        use_varlen=True,
                        cu_seqlens=qsl,
                        input_lengths=lengths,
                        prefix_lengths=zeros,
                        position_ids=torch.arange(tokens, device=self.device),
                        req_id_per_token=torch.zeros(
                            tokens, dtype=torch.int64, device=self.device
                        ),
                    )
                if block_span == ring:
                    self.assertTrue(bool((slots >= 0).all()))
                else:
                    self.assertTrue(bool((slots[:128] == -1).all()))
                pool = torch.zeros(3, ring, 584, dtype=torch.uint8, device=self.device)
                obj._pool_view_3d_fp8 = lambda *_: pool
                obj._should_async_workspace_reads_for_prefill = lambda *_: False
                wm = wm._replace(
                    cmp_reader=SimpleNamespace(
                        fill=lambda **kw: kw["out"][:, : wm.N].zero_()
                    )
                )
                q = (
                    torch.randn(
                        tokens, 1, dim, device=self.device, dtype=torch.bfloat16
                    )
                    / 8
                )
                kv = torch.randn(tokens, dim, device=self.device, dtype=torch.bfloat16)
                original_kv = kv.clone()

                def reference_projection(**kwargs):
                    indices = kwargs["indices"].view(tokens, -1).long()
                    valid = torch.arange(indices.size(1), device=self.device)[
                        None, :
                    ] < kwargs["topk_length"].view(-1, 1)
                    keys = kwargs["kv"].view(-1, dim)[indices.clamp_min(0)].float()
                    scores = (
                        (kwargs["q"].float() * keys)
                        .sum(-1)
                        .masked_fill(~valid, -torch.inf)
                    )
                    return (scores.softmax(-1)[..., None] * keys).sum(1)

                obj._flash_mla_sparse_fwd_chunked_projected = reference_projection
                actual = AttentionFP8._attn_via_workspace(
                    obj,
                    SimpleNamespace(q=q, kv_full=kv),
                    SimpleNamespace(
                        use_varlen=True,
                        batch_size=1,
                        any_cont=False,
                        cp_on=False,
                        freqs_cis=None,
                        workspace=None,
                    ),
                    workspace_meta=wm,
                    cmp_topk_runtime=None,
                )
                expected = []
                for pos in range(tokens):
                    keys = torch.cat(
                        [
                            torch.zeros((pos + 1) // 128, dim, device=self.device),
                            original_kv[max(0, pos - 127) : pos + 1].float(),
                        ]
                    )
                    scores = keys @ q[pos, 0].float()
                    expected.append(scores.softmax(0) @ keys)
                torch.testing.assert_close(
                    actual, torch.stack(expected), rtol=1e-4, atol=1e-5
                )

    def test_oom_recovery_uses_resolved_defrag_flag(self) -> None:
        for retry in (True, False):
            with self.subTest(oom_defrag_retry=retry), mock.patch.object(
                torch.cuda, "empty_cache"
            ) as empty_cache:
                output = torch.zeros(1, 8, device=self.device)
                forward = mock.Mock(
                    side_effect=[torch.OutOfMemoryError("injected"), output]
                )
                obj = SimpleNamespace(
                    oom_defrag_retry=retry,
                    _flash_mla_sparse_fwd_chunked_projected=forward,
                )
                obj._run_with_oom_defrag_retry = (
                    lambda device, fn, _obj=obj: AttentionFP8._run_with_oom_defrag_retry(
                        _obj, device, fn
                    )
                )
                qkv = SimpleNamespace(q=output.unsqueeze(1), kv_full=output.clone())
                common = SimpleNamespace(
                    cp_on=False,
                    freqs_cis=None,
                    workspace=None,
                    topk_idxs=torch.zeros(1, 1, dtype=torch.int32, device=self.device),
                    swa_meta=SimpleNamespace(
                        topk_length_kv_full=torch.ones(
                            1, dtype=torch.int32, device=self.device
                        )
                    ),
                )
                if retry:
                    actual = AttentionFP8._attn_fp8_swa_via_kv_full(obj, qkv, common)
                    self.assertIs(actual, output)
                    empty_cache.assert_called_once()
                    self.assertEqual(forward.call_count, 2)
                else:
                    with self.assertRaises(torch.OutOfMemoryError):
                        AttentionFP8._attn_fp8_swa_via_kv_full(obj, qkv, common)
                    empty_cache.assert_not_called()
                    self.assertEqual(forward.call_count, 1)

    def test_indexer_tuned_tiles_preserve_scores(self) -> None:
        q = torch.randn(1, 17, 64, 128, dtype=torch.bfloat16, device=self.device)
        kv = torch.randn(1, 257, 128, dtype=torch.bfloat16, device=self.device)
        weights = torch.randn(1, 17, 64, dtype=torch.float32, device=self.device)
        expected = (
            torch.einsum("bshd,btd->bsht", q.float(), kv.float()).relu_()
            * weights.unsqueeze(-1)
        ).sum(dim=2)
        for config in ((16, 256, 4, 2), (16, 256, 8, 3)):
            with self.subTest(config=config), mock.patch.object(
                _indexer_score_triton, "_INDEXER_TILE", config
            ):
                actual = _indexer_score_triton.v4_indexer_score(q, kv, weights)
                torch.testing.assert_close(actual, expected, rtol=1e-4, atol=2e-3)

    def test_decode_weight_projection_is_batch_invariant(self) -> None:
        dim, heads = 4096, 64
        weight = (
            torch.randn(heads, dim, dtype=torch.bfloat16, device=self.device) * 0.01
        )
        token = torch.randn(1, 1, dim, dtype=torch.bfloat16, device=self.device)
        expected = indexer._project_decode_weights(token, weight)[0, 0]
        reference = torch.nn.functional.linear(token.double(), weight.double())[0, 0]
        torch.testing.assert_close(expected.double(), reference, rtol=8e-3, atol=2e-4)

        def fixed_tile_reference(x):
            flat = x.reshape(-1, dim)
            tiles = []
            for start in range(0, flat.shape[0], 8):
                tile = flat[start : start + 8]
                count = tile.shape[0]
                if count < 8:
                    tile = torch.nn.functional.pad(tile, (0, 0, 0, 8 - count))
                projected = torch.mm(tile, weight.t(), out_dtype=torch.float32)
                tiles.append(projected[:count].to(x.dtype))
            return torch.cat(tiles).reshape(*x.shape[:-1], heads)

        for rows in (1, 2, 3, 4, 7, 8, 9, 15, 16, 31, 64, 65, 127):
            with self.subTest(rows=rows):
                # Change both row position and neighboring tokens, not just shape.
                data = torch.randn(
                    rows, 1, dim * 2, dtype=torch.bfloat16, device=self.device
                )
                x = data[..., ::2]
                positions = sorted({0, rows // 2, rows - 1})
                for position in positions:
                    x[position].copy_(token[0])
                actual = indexer._project_decode_weights(x, weight)
                torch.testing.assert_close(
                    actual, fixed_tile_reference(x), rtol=0, atol=0
                )
                full_reference = torch.nn.functional.linear(x.double(), weight.double())
                torch.testing.assert_close(
                    actual.double(), full_reference, rtol=8e-3, atol=2e-4
                )
                torch.testing.assert_close(
                    actual[positions, 0],
                    expected.expand(len(positions), -1),
                    rtol=0,
                    atol=0,
                )
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = indexer._project_decode_weights(x, weight)
                for _ in range(2):
                    x.copy_(torch.randn_like(x))
                    for position in positions:
                        x[position].copy_(token[0])
                    graph.replay()
                    torch.testing.assert_close(
                        captured, fixed_tile_reference(x), rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        captured[positions, 0],
                        expected.expand(len(positions), -1),
                        rtol=0,
                        atol=0,
                    )
                del graph
        empty = torch.empty(0, 1, dim, dtype=torch.bfloat16, device=self.device)
        self.assertEqual(
            indexer._project_decode_weights(empty, weight).shape, (0, 1, heads)
        )

    def test_topk_v3_binding_and_values(self) -> None:
        if not hasattr(rtp_llm_ops, "topk_v3"):
            self.fail("SM120 build is missing the required rtp_llm_ops.topk_v3 binding")

        rows, width, topk = 2, 640, 512
        logits = torch.randn(rows, width, dtype=torch.float32, device=self.device)
        lengths = torch.tensor([640, 513], dtype=torch.int32, device=self.device)
        output = torch.full((rows, topk), -1, dtype=torch.int32, device=self.device)
        workspace = torch.empty(1024 * 1024, dtype=torch.uint8, device=self.device)

        def assert_values() -> None:
            for row, length in enumerate(lengths.cpu().tolist()):
                actual = logits[row, output[row].long()].sort().values
                expected = (
                    logits[row, :length].topk(topk, sorted=False).values.sort().values
                )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        rtp_llm_ops.topk_v3(logits, lengths, output, workspace, topk, width)
        assert_values()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            rtp_llm_ops.topk_v3(logits, lengths, output, workspace, topk, width)
        logits.copy_(torch.randn_like(logits))
        graph.replay()
        torch.cuda.synchronize(self.device)
        assert_values()

    def test_topk_v3_selection_is_stable_during_graph_replay(self) -> None:
        width = 8192
        lengths = torch.tensor(
            [8192, 4097, 37, 0], dtype=torch.int32, device=self.device
        )
        # Unique scores make the selected set deterministic, so sorted
        # comparison separates selection membership from emission order.
        logits = torch.stack(
            [torch.randperm(width, device=self.device) for _ in range(4)]
        ).float()

        for topk in (512, 1024, 2048):
            with self.subTest(topk=topk):
                output = torch.empty((4, topk), dtype=torch.int32, device=self.device)
                self.assertTrue(
                    indexer._run_topk_v3(logits, lengths, output, topk, width)
                )
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    indexer._run_topk_v3(logits, lengths, output, topk, width)
                for _ in range(2):
                    expected = torch.full_like(output, -1)
                    for row, length in enumerate(lengths.cpu().tolist()):
                        count = min(topk, length)
                        if count:
                            indices = logits[row, :length].topk(count).indices
                            expected[row, :count] = indices.sort().values.int()
                    for _ in range(16):
                        graph.replay()
                        torch.testing.assert_close(
                            output.sort(dim=-1).values,
                            expected.sort(dim=-1).values,
                            rtol=0,
                            atol=0,
                        )
                    logits.neg_()

    def test_prefill_and_decode_topk_respect_row_bounds(self) -> None:
        for width in (1024, 2048, 8192, 32768):
            for topk in (512, 1024, 2048):
                for prefill in (False, True):
                    with self.subTest(width=width, topk=topk, prefill=prefill):
                        scores = torch.randint(
                            -16, 17, (4, width), device=self.device
                        ).float()
                        scores[0].zero_()
                        starts = torch.tensor(
                            [0, 17, 5, 100] if prefill else [0] * 4,
                            dtype=torch.int32,
                            device=self.device,
                        )
                        lengths = torch.tensor(
                            [width, width - 17, 37, 0],
                            dtype=torch.int32,
                            device=self.device,
                        )
                        ends = starts + lengths
                        output = torch.empty(
                            (4, topk), dtype=torch.int32, device=self.device
                        )

                        def forward():
                            if prefill:
                                indexer._run_prefill_topk(
                                    scores, starts, ends, output, topk, 4
                                )
                            else:
                                self.assertTrue(
                                    indexer._run_topk_v3(
                                        scores, ends, output, topk, width
                                    )
                                )

                        forward()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            forward()
                        for variant in range(3):
                            for _ in range(8):
                                graph.replay()
                                for row, (start, end) in enumerate(
                                    zip(starts.cpu().tolist(), ends.cpu().tolist())
                                ):
                                    count = min(topk, end - start)
                                    row_out = output[row]
                                    valid = row_out[row_out >= 0]
                                    self.assertEqual(valid.numel(), count)
                                    self.assertTrue(
                                        bool(
                                            (
                                                (row_out == -1)
                                                | (
                                                    (row_out >= 0)
                                                    & (row_out < end - start)
                                                )
                                            ).all()
                                        )
                                    )
                            if variant == 0:
                                scores.neg_()
                            else:
                                # Whole-row -inf leaves the radix prefill
                                # kernel's smem partially unfilled (out of its
                                # contract); keep finite extremes with
                                # distinct values instead.
                                scores[0].fill_(float("inf"))
                                scores[1:].mul_(-1e30)

    def test_paged_decode_uses_sm120_fallback_with_eight_heads(self) -> None:
        self.assertTrue(has_fp8_paged_mqa_logits(self.device))
        batch, query_len, heads = 2, 1, 8
        tokens, block_size = 19, 8
        query = (
            torch.randn(
                batch,
                query_len,
                heads,
                INDEXER_HEAD_DIM,
                dtype=torch.bfloat16,
                device=self.device,
            )
            * 0.25
        )
        router_weights = torch.randn(
            batch, query_len, heads, dtype=torch.float32, device=self.device
        )
        query_fp8, folded_weights = indexer_q_fp8_quant_fold(
            query.contiguous(), router_weights
        )

        blocks_per_request = (tokens + block_size - 1) // block_size
        total_blocks = 1 + batch * blocks_per_request
        pool = torch.zeros(
            total_blocks,
            block_size,
            INDEXER_ENTRY_BYTES,
            dtype=torch.uint8,
            device=self.device,
        )
        block_table = torch.empty(
            batch, blocks_per_request, dtype=torch.int32, device=self.device
        )
        dequantized = []
        for batch_id in range(batch):
            source = (
                torch.randn(
                    tokens,
                    INDEXER_HEAD_DIM,
                    dtype=torch.bfloat16,
                    device=self.device,
                )
                * 0.25
            )
            first_block = 1 + batch_id * blocks_per_request
            block_table[batch_id] = torch.arange(
                first_block,
                first_block + blocks_per_request,
                dtype=torch.int32,
                device=self.device,
            )
            slots = (
                torch.arange(tokens, dtype=torch.int64, device=self.device)
                + first_block * block_size
            )
            quantize_indexer_k(source, slots, pool)
            dequantized.append(
                dequantize_indexer_k(pool, slots, out_dtype=torch.float32)
            )

        live_lengths = torch.tensor(
            [[tokens], [tokens - 5]], dtype=torch.int32, device=self.device
        )

        def forward() -> torch.Tensor:
            return fp8_paged_indexer_score(
                query_fp8,
                folded_weights.view(batch * query_len, heads),
                pool.view(-1, INDEXER_ENTRY_BYTES),
                block_table,
                live_lengths,
                block_size,
                max_ctx_len=tokens,
            ).view(batch, query_len, tokens)

        keys = torch.stack(dequantized).float()

        def reference() -> torch.Tensor:
            result = torch.einsum("bshd,btd->bsht", query_fp8.float(), keys)
            result = (torch.relu(result) * folded_weights.float().unsqueeze(-1)).sum(
                dim=2
            )
            positions = torch.arange(tokens, device=self.device).view(1, 1, -1)
            result.masked_fill_(positions >= live_lengths.unsqueeze(-1), float("-inf"))
            return result

        torch.testing.assert_close(forward(), reference(), rtol=2e-3, atol=2e-3)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = forward()
        folded_weights.mul_(-0.75)
        graph.replay()
        torch.cuda.synchronize(self.device)
        torch.testing.assert_close(graph_output, reference(), rtol=2e-3, atol=2e-3)

    def test_paged_scores_are_batch_invariant(self) -> None:
        tokens, page = 1025, 64
        pages = (tokens + page - 1) // page
        keys = torch.randn(pages, page, 128, device=self.device).to(torch.float8_e4m3fn)
        scales = torch.rand(pages, page, device=self.device) * 0.02
        pool = torch.empty(pages, page * 132, dtype=torch.uint8, device=self.device)
        pool[:, : page * 128] = keys.view(torch.uint8).reshape(pages, -1)
        pool[:, page * 128 :] = scales.view(torch.uint8).reshape(pages, -1)
        table = torch.randperm(pages, device=self.device).int().reshape(1, -1)
        ordered_keys = keys.float()[table[0].long()].reshape(-1, 128)[:tokens]
        ordered_scales = scales[table[0].long()].reshape(-1)[:tokens]

        for heads in (8, 64):
            token = torch.randn(1, 1, heads, 128, device=self.device).to(
                torch.float8_e4m3fn
            )
            weight = torch.randn(1, heads, device=self.device) * 0.01
            length = torch.full((1, 1), tokens, dtype=torch.int32, device=self.device)

            def score(q, w, lengths, tables):
                return fp8_paged_indexer_score(
                    q, w, pool.reshape(-1, 132), tables, lengths, page, tokens
                )

            baseline = score(token, weight, length, table)
            reference = (
                (
                    token[0, 0].double()
                    @ (ordered_keys.double() * ordered_scales.double()[:, None]).t()
                ).relu()
                * weight[0].double()[:, None]
            ).sum(0)
            torch.testing.assert_close(
                baseline[0].double(), reference, rtol=2e-3, atol=2e-3
            )
            for batch, next_n in ((1, 4), (2, 3), (3, 7), (1, 33)):
                with self.subTest(heads=heads, batch=batch, next_n=next_n):
                    rows = batch * next_n
                    q = torch.randn(batch, next_n, heads, 128, device=self.device).to(
                        torch.float8_e4m3fn
                    )
                    weights = torch.randn(rows, heads * 2, device=self.device)[:, ::2]
                    positions = sorted({0, rows // 2, rows - 1})
                    for position in positions:
                        q.view(rows, heads, 128)[position].copy_(token[0, 0])
                        weights[position].copy_(weight[0])
                    lengths = length.expand(batch, next_n).contiguous()
                    tables = table.expand(batch, -1).contiguous()

                    def forward():
                        return score(q, weights, lengths, tables)

                    actual = forward()
                    torch.testing.assert_close(
                        actual[positions],
                        baseline.expand(len(positions), -1),
                        rtol=0,
                        atol=0,
                    )
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        captured = forward()
                    for _ in range(2):
                        weights.neg_()
                        graph.replay()
                        torch.testing.assert_close(captured, forward(), rtol=0, atol=0)
                    torch.testing.assert_close(
                        captured[positions],
                        baseline.expand(len(positions), -1),
                        rtol=0,
                        atol=0,
                    )

    def test_paged_scores_mask_invalid_pages_and_idle_rows(self) -> None:
        from unittest.mock import patch

        import rtp_llm.models_py.modules.dsv4.fp8._indexer_score as idx_score

        page = 64
        for heads in (8, 32):
            with self.subTest(heads=heads):
                torch.manual_seed(100 + heads)
                k_vals = (
                    torch.randn(2, page, INDEXER_HEAD_DIM, device=self.device) * 0.25
                ).to(torch.float8_e4m3fn)
                k_scales = (
                    torch.rand(2, page, dtype=torch.float32, device=self.device) + 0.1
                )
                pool_blk = torch.empty(
                    2,
                    page * INDEXER_ENTRY_BYTES,
                    dtype=torch.uint8,
                    device=self.device,
                )
                pool_blk[:, : page * INDEXER_HEAD_DIM] = k_vals.view(
                    torch.uint8
                ).reshape(2, -1)
                pool_blk[:, page * INDEXER_HEAD_DIM :] = k_scales.view(
                    torch.uint8
                ).reshape(2, -1)
                pool = pool_blk.view(-1, INDEXER_ENTRY_BYTES)
                q = (
                    torch.randn(2, 3, heads, INDEXER_HEAD_DIM, device=self.device)
                    * 0.25
                ).to(torch.float8_e4m3fn)
                weights = torch.randn(6, heads, dtype=torch.float32, device=self.device)
                table = torch.tensor(
                    [[0, -1, 1], [1, 2, 0]], dtype=torch.int32, device=self.device
                )
                lengths = torch.tensor(
                    [[0, 65, 192], [1, 129, 191]], dtype=torch.int32, device=self.device
                )

                def reference():
                    k_fp32 = (k_vals.float() * k_scales.unsqueeze(-1)).reshape(
                        2 * page, INDEXER_HEAD_DIM
                    )
                    columns = torch.arange(192, device=self.device)
                    page_ids = table.repeat_interleave(3, 0)[:, columns // page]
                    valid = (
                        (page_ids >= 0)
                        & (page_ids < 2)
                        & (columns.unsqueeze(0) < lengths.reshape(-1, 1))
                    )
                    safe_slots = (page_ids.clamp(0, 1) * page) + (columns % page)
                    k_gathered = k_fp32[safe_slots]
                    q_flat = q.float().reshape(6, heads, INDEXER_HEAD_DIM)
                    dots = torch.einsum("rhd,rkd->rhk", q_flat, k_gathered).relu_()
                    scores = (dots * weights.unsqueeze(-1)).sum(dim=1)
                    return scores.masked_fill(~valid, -float("inf"))

                native_paged = idx_score._deep_gemm.fp8_fp4_paged_mqa_logits
                calls = []
                with patch.object(
                    idx_score._deep_gemm,
                    "fp8_fp4_paged_mqa_logits",
                    side_effect=lambda *a, **kw: calls.append(1)
                    or native_paged(*a, **kw),
                ):
                    def forward():
                        return fp8_paged_indexer_score(
                            q, weights, pool, table, lengths, page, 192
                        )

                    eager = forward()
                    ref_eager = reference()
                    torch.testing.assert_close(eager, ref_eager, rtol=2e-2, atol=2e-2)
                    self.assertTrue(
                        torch.equal(torch.isneginf(eager), torch.isneginf(ref_eager))
                    )

                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        captured = forward()
                    lengths.zero_()
                    graph.replay()
                    torch.testing.assert_close(captured, reference(), rtol=0, atol=0)

                    lengths.copy_(
                        torch.tensor(
                            [[128, 192, 64], [192, 64, 128]],
                            dtype=torch.int32,
                            device=self.device,
                        )
                    )
                    table.copy_(
                        torch.tensor(
                            [[-1, 0, 1], [0, 1, 9]],
                            dtype=torch.int32,
                            device=self.device,
                        )
                    )
                    graph.replay()
                    ref_mut = reference()
                    torch.testing.assert_close(captured, ref_mut, rtol=2e-2, atol=2e-2)
                    self.assertTrue(
                        torch.equal(torch.isneginf(captured), torch.isneginf(ref_mut))
                    )

                if heads == 32:
                    self.assertEqual(len(calls), 2)
                else:
                    self.assertEqual(len(calls), 0)

    def test_nonpaged_prefill_uses_sm120_fallback_with_eight_heads(self) -> None:
        self.assertTrue(has_fp8_mqa_logits(self.device))
        query_tokens, key_tokens, heads = 7, 23, 8
        query = (
            torch.randn(
                1,
                query_tokens,
                heads,
                INDEXER_HEAD_DIM,
                dtype=torch.bfloat16,
                device=self.device,
            )
            * 0.25
        )
        router_weights = torch.randn(
            1, query_tokens, heads, dtype=torch.float32, device=self.device
        )
        query_fp8, folded_weights = indexer_q_fp8_quant_fold(
            query.contiguous(), router_weights
        )
        key = torch.randn(
            key_tokens,
            INDEXER_HEAD_DIM,
            dtype=torch.float32,
            device=self.device,
        )
        key_scale = key.abs().amax(dim=-1).clamp_min(1e-8) / 448.0
        key_fp8 = (key / key_scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
        starts = torch.tensor(
            [0, 1, 2, 3, 0, 5, 7], dtype=torch.int32, device=self.device
        )
        ends = torch.tensor(
            [8, 9, 11, 13, 17, 21, 23], dtype=torch.int32, device=self.device
        )

        # The SM120 fallback feeds BF16 tensor cores with raw FP8 Q/K operands
        # (exact in BF16) and applies the FP32 key scale in FP32 afterward.
        dequantized_key = key_fp8.to(torch.bfloat16).float() * key_scale.unsqueeze(-1)
        dequantized_query = query_fp8.to(torch.bfloat16)

        def forward() -> torch.Tensor:
            return fp8_mqa_indexer_score(
                query_fp8.view(query_tokens, heads, INDEXER_HEAD_DIM),
                folded_weights.view(query_tokens, heads),
                key_fp8,
                key_scale,
                starts,
                ends,
                clean_logits=True,
            )

        def reference() -> torch.Tensor:
            result = torch.einsum(
                "mhd,td->mht",
                dequantized_query.float().view(query_tokens, heads, INDEXER_HEAD_DIM),
                dequantized_key.float(),
            )
            result = (
                torch.relu(result)
                * folded_weights.float().view(query_tokens, heads).unsqueeze(-1)
            ).sum(dim=1)
            positions = torch.arange(key_tokens, device=self.device).unsqueeze(0)
            valid = (positions >= starts.long().unsqueeze(1)) & (
                positions < ends.long().unsqueeze(1)
            )
            result.masked_fill_(~valid, float("-inf"))
            return result

        torch.testing.assert_close(forward(), reference(), rtol=2e-3, atol=2e-3)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = forward()
        folded_weights.mul_(0.5)
        graph.replay()
        torch.cuda.synchronize(self.device)
        torch.testing.assert_close(graph_output, reference(), rtol=2e-3, atol=2e-3)

    def test_sm120_native_paged_and_prefill_mqa_logits(self) -> None:
        from unittest.mock import patch

        from rtp_llm.models_py.modules.dsv4.fp8 import _indexer_score as idx_score

        self.assertTrue(
            idx_score.supports_dsv4_paged_mqa_logits(self.device, 64, num_heads=64)
        )
        self.assertTrue(idx_score.supports_dsv4_mqa_logits(self.device, num_heads=64))
        self.assertFalse(idx_score.supports_paged_mqa_logits(self.device, 64))
        self.assertFalse(idx_score.supports_mqa_logits(self.device))
        torch.manual_seed(2026)

        # Paged H=64, block_size=64 with CUDA Graph replay via native fp8_fp4_paged_mqa_logits
        B, next_n, H, block_size, T_ctx = 2, 2, 64, 64, 128
        pages = B * (T_ctx // block_size)
        keys = torch.randn(pages, block_size, INDEXER_HEAD_DIM, device=self.device).to(
            torch.float8_e4m3fn
        )
        scales = torch.rand(pages, block_size, device=self.device) * 0.25
        pool = torch.empty(
            pages, block_size * INDEXER_ENTRY_BYTES, dtype=torch.uint8, device=self.device
        )
        pool[:, : block_size * INDEXER_HEAD_DIM] = keys.view(torch.uint8).reshape(
            pages, -1
        )
        pool[:, block_size * INDEXER_HEAD_DIM :] = scales.view(torch.uint8).reshape(
            pages, -1
        )
        pool_flat = pool.view(-1, INDEXER_ENTRY_BYTES)
        block_table = torch.tensor(
            [[0, 1], [2, 3]], dtype=torch.int32, device=self.device
        )
        q_paged = (
            torch.randn(B, next_n, H, INDEXER_HEAD_DIM, device=self.device) * 0.25
        ).to(torch.float8_e4m3fn)
        w_paged = torch.randn(B * next_n, H, dtype=torch.float32, device=self.device)
        lengths = torch.full((B, next_n), T_ctx, dtype=torch.int32, device=self.device)

        native_paged = idx_score._deep_gemm.fp8_fp4_paged_mqa_logits
        paged_calls = []
        with patch.object(
            idx_score._deep_gemm,
            "fp8_fp4_paged_mqa_logits",
            side_effect=lambda *a, **kw: paged_calls.append(1) or native_paged(*a, **kw),
        ):
            out_paged = fp8_paged_indexer_score(
                q_paged, w_paged, pool_flat, block_table, lengths, block_size, T_ctx
            )
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                cg_paged = fp8_paged_indexer_score(
                    q_paged, w_paged, pool_flat, block_table, lengths, block_size, T_ctx
                )
            w_paged.mul_(0.5)
            graph.replay()
            torch.cuda.synchronize(self.device)
            expected_paged = fp8_paged_indexer_score(
                q_paged, w_paged, pool_flat, block_table, lengths, block_size, T_ctx
            )
        self.assertGreaterEqual(len(paged_calls), 3)
        torch.testing.assert_close(cg_paged, expected_paged, rtol=0, atol=0)
        self.assertEqual(tuple(out_paged.shape), (B * next_n, T_ctx))

        # Prefill H=64 max_seqlen_k=0 and max_seqlen_k>0 with unclamped bounds + CUDA Graph
        M, N = 4, 32
        q_pre = (torch.randn(M, H, INDEXER_HEAD_DIM, device=self.device) * 0.25).to(
            torch.float8_e4m3fn
        )
        w_pre = torch.randn(M, H, dtype=torch.float32, device=self.device)
        k_pre = (torch.randn(N, INDEXER_HEAD_DIM, device=self.device) * 0.25).to(
            torch.float8_e4m3fn
        )
        ks_scale = torch.rand(N, dtype=torch.float32, device=self.device) + 0.1
        starts = torch.tensor([-1, 10, 4, 36], dtype=torch.int32, device=self.device)
        ends = torch.tensor([6, 8, 12, 40], dtype=torch.int32, device=self.device)

        def _prefill_ref(q_t, w_t, k_t, s_t, st, en, max_k):
            k_fp32 = k_t.float() * s_t.unsqueeze(1)
            logits = (
                torch.einsum("mhd,nd->mhn", q_t.float(), k_fp32).relu_()
                * w_t.unsqueeze(-1)
            ).sum(dim=1)
            cols = torch.arange(k_t.shape[0], device=q_t.device)
            if max_k == 0:
                valid = (cols.unsqueeze(0) >= st.unsqueeze(1)) & (
                    cols.unsqueeze(0) < en.unsqueeze(1)
                )
                return logits.masked_fill(~valid, -float("inf"))
            rel = torch.arange(max_k, device=q_t.device).unsqueeze(0)
            abs_pos = st.long().unsqueeze(1) + rel
            valid = (
                (abs_pos >= st.long().unsqueeze(1))
                & (abs_pos < en.long().unsqueeze(1))
                & (abs_pos >= 0)
                & (abs_pos < k_t.shape[0])
            )
            gathered = logits.gather(1, abs_pos.clamp(0, k_t.shape[0] - 1))
            return gathered.masked_fill(~valid, -float("inf"))

        native_mqa = idx_score._deep_gemm.fp8_fp4_mqa_logits
        mqa_calls = []
        with patch.object(
            idx_score._deep_gemm,
            "fp8_fp4_mqa_logits",
            side_effect=lambda *a, **kw: mqa_calls.append(1) or native_mqa(*a, **kw),
        ):
            out_abs = fp8_mqa_indexer_score(
                q_pre,
                w_pre,
                k_pre,
                ks_scale,
                starts,
                ends,
                clean_logits=True,
                max_seqlen_k=0,
            )
            out_rel = fp8_mqa_indexer_score(
                q_pre,
                w_pre,
                k_pre,
                ks_scale,
                starts,
                ends,
                clean_logits=True,
                max_seqlen_k=4,
            )
            graph_pre = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph_pre):
                cg_rel = fp8_mqa_indexer_score(
                    q_pre,
                    w_pre,
                    k_pre,
                    ks_scale,
                    starts,
                    ends,
                    clean_logits=True,
                    max_seqlen_k=4,
                )
            w_pre.mul_(0.5)
            graph_pre.replay()
            torch.cuda.synchronize(self.device)
            expected_rel = fp8_mqa_indexer_score(
                q_pre,
                w_pre,
                k_pre,
                ks_scale,
                starts,
                ends,
                clean_logits=True,
                max_seqlen_k=4,
            )
        self.assertGreaterEqual(len(mqa_calls), 4)
        self.assertEqual(tuple(out_abs.shape), (M, N))
        self.assertEqual(tuple(out_rel.shape), (M, 4))
        ref_abs = _prefill_ref(q_pre, w_pre * 2.0, k_pre, ks_scale, starts, ends, 0)
        ref_rel = _prefill_ref(q_pre, w_pre * 2.0, k_pre, ks_scale, starts, ends, 4)
        torch.testing.assert_close(out_abs, ref_abs, rtol=2e-2, atol=2e-2)
        self.assertTrue(torch.equal(torch.isneginf(out_abs), torch.isneginf(ref_abs)))
        torch.testing.assert_close(out_rel, ref_rel, rtol=2e-2, atol=2e-2)
        self.assertTrue(torch.equal(torch.isneginf(out_rel), torch.isneginf(ref_rel)))
        torch.testing.assert_close(cg_rel, expected_rel, rtol=0, atol=0)

    @unittest.skipIf(
        torch.cuda.device_count() < 2, "Requires at least 2 CUDA devices"
    )
    def test_sm120_native_mqa_nondefault_device(self) -> None:
        dev1 = torch.device("cuda", 1)
        prev_dev = torch.cuda.current_device()
        try:
            torch.cuda.set_device(0)
            torch.manual_seed(303)
            H, page = 32, 64

            # Paged H=32 on cuda:1 with CPU host context_lens & block_table
            k_vals = (torch.randn(2, page, INDEXER_HEAD_DIM, device=dev1) * 0.25).to(
                torch.float8_e4m3fn
            )
            k_scales = torch.rand(2, page, dtype=torch.float32, device=dev1) + 0.1
            pool_blk = torch.empty(
                2, page * INDEXER_ENTRY_BYTES, dtype=torch.uint8, device=dev1
            )
            pool_blk[:, : page * INDEXER_HEAD_DIM] = k_vals.view(torch.uint8).reshape(
                2, -1
            )
            pool_blk[:, page * INDEXER_HEAD_DIM :] = k_scales.view(torch.uint8).reshape(
                2, -1
            )
            pool = pool_blk.view(-1, INDEXER_ENTRY_BYTES)
            q_paged = (
                torch.randn(2, 2, H, INDEXER_HEAD_DIM, device=dev1) * 0.25
            ).to(torch.float8_e4m3fn)
            w_paged = torch.randn(4, H, dtype=torch.float32, device=dev1)
            table_cpu = torch.tensor([[0, -1], [1, 0]], dtype=torch.int32)
            lengths_cpu = torch.tensor([[0, 65], [1, 128]], dtype=torch.int32)

            out_paged = fp8_paged_indexer_score(
                q_paged, w_paged, pool, table_cpu, lengths_cpu, page, 128
            )
            self.assertEqual(torch.cuda.current_device(), 0)
            self.assertEqual(out_paged.device, dev1)

            k_fp32 = (k_vals.float() * k_scales.unsqueeze(-1)).reshape(
                2 * page, INDEXER_HEAD_DIM
            )
            cols = torch.arange(128, device=dev1)
            bt_rows = table_cpu.to(dev1).repeat_interleave(2, 0)
            page_ids = bt_rows[:, cols // page]
            valid_p = (
                (page_ids >= 0)
                & (page_ids < 2)
                & (cols.unsqueeze(0) < lengths_cpu.to(dev1).reshape(-1, 1))
            )
            safe_slots = (page_ids.clamp(0, 1) * page) + (cols % page)
            dots_p = torch.einsum(
                "rhd,rkd->rhk", q_paged.float().reshape(4, H, INDEXER_HEAD_DIM), k_fp32[safe_slots]
            ).relu_()
            ref_paged = (dots_p * w_paged.unsqueeze(-1)).sum(dim=1).masked_fill(
                ~valid_p, -float("inf")
            )
            torch.testing.assert_close(out_paged, ref_paged, rtol=2e-2, atol=2e-2)
            self.assertTrue(
                torch.equal(torch.isneginf(out_paged), torch.isneginf(ref_paged))
            )

            # Prefill H=32 on cuda:1 with CPU host unclamped bounds
            M, N = 4, 32
            q_pre = (torch.randn(M, H, INDEXER_HEAD_DIM, device=dev1) * 0.25).to(
                torch.float8_e4m3fn
            )
            w_pre = torch.randn(M, H, dtype=torch.float32, device=dev1)
            k_pre = (torch.randn(N, INDEXER_HEAD_DIM, device=dev1) * 0.25).to(
                torch.float8_e4m3fn
            )
            ks_scale = torch.rand(N, dtype=torch.float32, device=dev1) + 0.1
            starts_cpu = torch.tensor([-1, 10, 4, 36], dtype=torch.int32)
            ends_cpu = torch.tensor([6, 8, 12, 40], dtype=torch.int32)

            out_abs = fp8_mqa_indexer_score(
                q_pre, w_pre, k_pre, ks_scale, starts_cpu, ends_cpu, clean_logits=True, max_seqlen_k=0
            )
            out_rel = fp8_mqa_indexer_score(
                q_pre, w_pre, k_pre, ks_scale, starts_cpu, ends_cpu, clean_logits=True, max_seqlen_k=4
            )
            self.assertEqual(torch.cuda.current_device(), 0)
            self.assertEqual(out_abs.device, dev1)
            self.assertEqual(out_rel.device, dev1)

            k_pre_fp32 = k_pre.float() * ks_scale.unsqueeze(1)
            base_logits = (
                torch.einsum("mhd,nd->mhn", q_pre.float(), k_pre_fp32).relu_()
                * w_pre.unsqueeze(-1)
            ).sum(dim=1)
            st_d, en_d = starts_cpu.to(dev1), ends_cpu.to(dev1)
            cols_n = torch.arange(N, device=dev1)
            valid_abs = (cols_n.unsqueeze(0) >= st_d.unsqueeze(1)) & (
                cols_n.unsqueeze(0) < en_d.unsqueeze(1)
            )
            ref_abs = base_logits.masked_fill(~valid_abs, -float("inf"))
            torch.testing.assert_close(out_abs, ref_abs, rtol=2e-2, atol=2e-2)
            self.assertTrue(
                torch.equal(torch.isneginf(out_abs), torch.isneginf(ref_abs))
            )

            rel = torch.arange(4, device=dev1).unsqueeze(0)
            abs_pos = st_d.long().unsqueeze(1) + rel
            valid_rel = (
                (abs_pos >= st_d.long().unsqueeze(1))
                & (abs_pos < en_d.long().unsqueeze(1))
                & (abs_pos >= 0)
                & (abs_pos < N)
            )
            ref_rel = base_logits.gather(1, abs_pos.clamp(0, N - 1)).masked_fill(
                ~valid_rel, -float("inf")
            )
            torch.testing.assert_close(out_rel, ref_rel, rtol=2e-2, atol=2e-2)
            self.assertTrue(
                torch.equal(torch.isneginf(out_rel), torch.isneginf(ref_rel))
            )

            # Verify persistent TopK (dsv4_persistent_topk) + SM120 sparse decode
            # (pack_slots_to_paged + FlashInfer) work on cuda:1 while cuda:0 is current.
            from rtp_llm.models_py.modules.dsv4.fp8 import sm120_sparse_mla
            from rtp_llm.models_py.modules.dsv4.indexer_topk import select_indexer_topk

            scores_dev1 = torch.randn((2, 2048), device=dev1, dtype=torch.float32)
            lens_dev1 = torch.tensor([1024, 1500], device=dev1, dtype=torch.int32)
            topk_idx = select_indexer_topk(scores_dev1, 512, lengths=lens_dev1)
            self.assertEqual(torch.cuda.current_device(), 0)
            self.assertEqual(topk_idx.device, dev1)
            self.assertEqual(tuple(topk_idx.shape), (2, 512))

            # Minimal SM120 sparse decode on cuda:1 with zeroed cache + out-of-pool guard
            decode_cache = torch.zeros((2, 64, 584), dtype=torch.uint8, device=dev1)
            decode_q = torch.randn((2, 1, 16, 512), dtype=torch.bfloat16, device=dev1)
            decode_sink = torch.zeros((16,), dtype=torch.float32, device=dev1)
            decode_slots = torch.full((2, 128), -1, dtype=torch.int32, device=dev1)
            decode_slots[0, 0] = 0
            decode_slots[1, 0] = 64
            decode_out = sm120_sparse_mla.forward_decode(
                decode_q, decode_cache, decode_sink, decode_slots, softmax_scale=0.1
            )
            self.assertEqual(torch.cuda.current_device(), 0)
            self.assertEqual(decode_out.device, dev1)
            self.assertTrue(torch.isfinite(decode_out).all())
        finally:
            torch.cuda.set_device(prev_dev)


if __name__ == "__main__":
    unittest.main()
