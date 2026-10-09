"""RTX PRO 5000 gate for SM120 indexer prefill/decode and topk_v3."""

from __future__ import annotations

import os
import unittest
from unittest import mock

import torch

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
