"""SM120 hardware gate for indexer prefill/decode and topk_v3."""

from __future__ import annotations

import os
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
            self.fail("SM120 hardware gate was scheduled without a CUDA device")
        if not is_sm120():
            self.fail(
                "SM120 hardware gate was scheduled on "
                f"compute capability {torch.cuda.get_device_capability()}"
            )
        self.device = torch.device("cuda", torch.cuda.current_device())
        torch.manual_seed(23)

    def test_failed_tail_launch_drains_submitted_work(self) -> None:
        x = torch.ones(128, device=self.device)
        result = torch.empty_like(x)
        tail = torch.cuda.Stream()
        done = torch.cuda.Event()
        state = SimpleNamespace(x=x)

        def fail_after_launch(*_):
            torch.cuda._sleep(20_000_000)
            torch.add(x, 1, out=result)
            done.record()
            self.assertFalse(done.query())
            raise RuntimeError("injected tail failure")

        obj = SimpleNamespace(
            _prefill_score_prefix=lambda *args, **kwargs: state,
            _prefill_score_tail=fail_after_launch,
        )
        with self.assertRaisesRegex(RuntimeError, "injected tail failure"):
            indexer.IndexerFP8.forward_prefill_tail_on_stream(
                obj, x, x, None, workspace=None, tail_stream=tail
            )
        self.assertTrue(done.query())
        torch.testing.assert_close(result, torch.full_like(result, 2))

    def test_failed_consumer_preparation_drains_pending_tail(self) -> None:
        for phase in ("compressor", "materialize"):
            with self.subTest(phase=phase):
                x = torch.ones(128, device=self.device)
                tail = torch.cuda.Stream()
                tail.wait_stream(torch.cuda.current_stream())
                done = torch.cuda.Event()
                with torch.cuda.stream(tail):
                    torch.cuda._sleep(20_000_000)
                    output = x + 1
                    done.record()
                pending = indexer._PendingIndexerTopk(
                    output, done, tail, SimpleNamespace(x=x)
                )

                def fail(*args, **kwargs):
                    self.assertFalse(done.query())
                    raise RuntimeError("injected consumer failure")

                obj = SimpleNamespace(
                    compressor=fail if phase == "compressor" else lambda *a, **k: None,
                    _materialize_prefill_q=fail,
                )
                common = SimpleNamespace(sp_int=0, workspace=None)
                with self.assertRaisesRegex(RuntimeError, "injected consumer failure"):
                    AttentionFP8._forward_prefill_compressed(
                        obj, x, None, common, pending, None, None
                    )
                self.assertTrue(done.query())
                torch.testing.assert_close(output, torch.full_like(output, 2))

    def test_paged_prefill_maps_multiple_requests_and_invalid_slots(self) -> None:
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
        wm = SimpleNamespace(
            M=6,
            N=2,
            dense_cmp_topk=torch.zeros(4, 2, dtype=torch.int32, device=self.device),
            cmp_pool_slot_mapping=torch.tensor([[7, 2], [11, 5]], device=self.device),
            swa_pool_slot_mapping=torch.tensor(
                [[19, 3, 17, 4], [8, 21, 6, 14]], device=self.device
            ),
            qsl=None,
            swa_seq_lens=None,
            swa_gather_lens=None,
        )
        qkv = SimpleNamespace(
            q=torch.zeros(4, 1, 8, device=self.device),
            kv_full=torch.zeros(4, 8, device=self.device),
        )
        common = SimpleNamespace(cp_on=False, freqs_cis=None, workspace=None)
        projected = mock.Mock(side_effect=lambda **kwargs: kwargs)
        obj = SimpleNamespace(
            compress_ratio=4,
            window_size=3,
            dim=8,
            _flash_mla_sparse_fwd_chunked_projected=projected,
        )
        pool = torch.empty(24, 1, 8, device=self.device)
        with mock.patch(
            "rtp_llm.models_py.modules.dsv4.fp8._swa_ops_triton.combine_topk_swa_indices",
            return_value=(combined.unsqueeze(1), lengths),
        ):
            actual = AttentionFP8._attn_via_sm120_paged_cache(
                obj,
                qkv=qkv,
                common=common,
                workspace_meta=wm,
                cmp_topk_runtime=None,
                swa_pool_3d=pool,
                cmp_pool_3d=pool,
            )
        for key, expected in (
            ("sm120_extra_indices", [[7, -1], [2, -1], [11, -1], [-1, -1]]),
            ("sm120_extra_lens", [1, 1, 1, 0]),
            ("sm120_swa_indices", [[19, 3, -1], [17, 4, -1], [8, 21, 6], [14, -1, -1]]),
            ("sm120_swa_lens", [2, 2, 3, 1]),
        ):
            torch.testing.assert_close(
                actual[key],
                torch.tensor(expected, dtype=torch.int32, device=self.device),
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
                with mock.patch.dict(
                    os.environ, {"DSV4_SM120_PAGED_CACHE_PREFILL": "1"}
                ), mock.patch.object(
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
                        swa_slot_mapping=slots,
                    )
                if block_span == ring:
                    self.assertTrue(bool((slots >= 0).all()))
                    torch.testing.assert_close(wm.swa_pool_slot_mapping[0], slots)
                    continue
                self.assertTrue(bool((slots[:128] == -1).all()))
                self.assertIsNone(wm.swa_pool_slot_mapping)
                self.assertIsNone(wm.cmp_pool_slot_mapping)
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

    def test_oom_recovery_uses_resolved_moe_strategy(self) -> None:
        from rtp_llm.models_py.model_desc.deepseek_v4_model import (
            _resolve_dsv4_moe_strategy,
        )

        for requested, alias, retry in (
            ("sm120_decode", "", True),
            ("auto", "sm120_decode", True),
            ("sm120_fused_moe", "sm120_decode", False),
        ):
            with self.subTest(requested=requested, alias=alias), mock.patch.dict(
                os.environ, {"DSV4_MOE_STRATEGY": alias}
            ), mock.patch.object(torch.cuda, "empty_cache") as empty_cache:
                output = torch.zeros(1, 8, device=self.device)
                forward = mock.Mock(
                    side_effect=[torch.OutOfMemoryError("injected"), output]
                )
                obj = SimpleNamespace(
                    moe_strategy=_resolve_dsv4_moe_strategy(
                        SimpleNamespace(moe_strategy=requested)
                    ),
                    _flash_mla_sparse_fwd_chunked_projected=forward,
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

    def test_canonical_topk_is_stable_during_graph_replay(self) -> None:
        width = 8192
        lengths = torch.tensor(
            [8192, 4097, 37, 0], dtype=torch.int32, device=self.device
        )
        # Unique scores separate output-order stability from cutoff tie policy.
        logits = torch.stack(
            [torch.randperm(width, device=self.device) for _ in range(4)]
        ).float()

        with mock.patch.dict(
            os.environ,
            {"DSV4_TOPK_V3": "1", "DSV4_INDEXER_TOPK_CANONICALIZE": "1"},
        ):
            for topk in (512, 1024, 2048):
                with self.subTest(topk=topk):
                    output = torch.empty(
                        (4, topk), dtype=torch.int32, device=self.device
                    )
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
                            torch.testing.assert_close(output, expected, rtol=0, atol=0)
                        logits.neg_()

    def test_prefill_and_decode_canonicalize_cutoff_ties(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"DSV4_TOPK_V3": "1", "DSV4_INDEXER_TOPK_CANONICALIZE": "1"},
        ):
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
                                expected = torch.full_like(output, -1)
                                for row, (start, end) in enumerate(
                                    zip(starts.cpu().tolist(), ends.cpu().tolist())
                                ):
                                    count = min(topk, end - start)
                                    if count:
                                        indices = torch.argsort(
                                            scores[row, start:end],
                                            descending=True,
                                            stable=True,
                                        )[:count]
                                        expected[row, :count] = (
                                            indices.sort().values.int()
                                        )
                                for _ in range(8):
                                    graph.replay()
                                    torch.testing.assert_close(
                                        output, expected, rtol=0, atol=0
                                    )
                                if variant == 0:
                                    scores.neg_()
                                else:
                                    scores[0].fill_(float("inf"))
                                    scores[1:].fill_(float("-inf"))

    def test_canonical_topk_fallback_preserves_lowest_index_ties(self) -> None:
        for width, stride in ((1025, 2), (32769, 1)):
            with self.subTest(width=width, stride=stride):
                scores = torch.zeros((2, width * stride), device=self.device)[
                    :, ::stride
                ]
                starts = torch.tensor([3, 17], dtype=torch.int32, device=self.device)
                ends = torch.full_like(starts, width)
                output = torch.zeros((2, 512), dtype=torch.int32, device=self.device)
                indexer._canonicalize_fp8_topk(scores, output, ends, starts)
                expected = torch.arange(512, device=self.device, dtype=torch.int32)
                torch.testing.assert_close(
                    output, expected.expand(2, -1), rtol=0, atol=0
                )
                scores.fill_(float("-inf"))
                ends[1] = starts[1] + 37
                indexer._canonicalize_fp8_topk(scores, output, ends, starts)
                expected_rows = expected.expand(2, -1).clone()
                expected_rows[1, 37:] = -1
                torch.testing.assert_close(output, expected_rows, rtol=0, atol=0)

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
        page, heads = 64, 8
        pool = torch.zeros(2 * page, 132, dtype=torch.uint8, device=self.device)
        q = torch.ones(2, 3, heads, 128, device=self.device).to(torch.float8_e4m3fn)
        weights = torch.ones(6, heads, device=self.device)
        table = torch.tensor(
            [[0, -1, 1], [1, 2, 0]], dtype=torch.int32, device=self.device
        )
        lengths = torch.tensor(
            [[0, 65, 192], [1, 129, 191]], dtype=torch.int32, device=self.device
        )

        def forward():
            return fp8_paged_indexer_score(q, weights, pool, table, lengths, page, 192)

        def reference():
            columns = torch.arange(192, device=self.device)
            valid_page = table.repeat_interleave(3, 0)[:, columns // page]
            valid = (
                (valid_page >= 0)
                & (valid_page < 2)
                & (columns < lengths.reshape(-1, 1))
            )
            return torch.zeros(6, 192, device=self.device).masked_fill(
                ~valid, -float("inf")
            )

        torch.testing.assert_close(forward(), reference(), rtol=0, atol=0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = forward()
        lengths.zero_()
        graph.replay()
        torch.testing.assert_close(captured, reference(), rtol=0, atol=0)

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

        # The SM120 fallback intentionally feeds BF16 tensor cores.  Build the
        # oracle from the same post-FP8, post-scale BF16 values instead of the
        # higher-precision intermediate that the production kernel never sees.
        dequantized_key = (key_fp8.float() * key_scale.unsqueeze(-1)).to(torch.bfloat16)
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


if __name__ == "__main__":
    unittest.main()
