"""Exact discrete/representable-value probes, not floating model acceptance."""

import hashlib
import json
import os
import unittest
from pathlib import Path

import torch

from rtp_llm.models_py.modules.dsv41.cache_layout import CacheRegion
from rtp_llm.models_py.modules.dsv41.compact_reader import CompactPages
from rtp_llm.models_py.modules.dsv41.compact_writer import encode_compact
from rtp_llm.models_py.modules.dsv41.indexer import (
    CANDIDATE_BLOCKS,
    QUERY_TILE,
    score_candidate_tile,
    select_index_positions,
)


def ints(values):
    return torch.tensor(values, dtype=torch.int32, device="cuda")


class IndexerGpuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert os.geteuid() != 0
        assert torch.cuda.is_available()
        assert torch.cuda.get_device_capability()[0] == 10
        assert torch.version.cuda.startswith("13.")
        cls.observations = []

    @classmethod
    def tearDownClass(cls):
        output = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
        if output:
            path = Path(output) / "indexer_components.json"
            path.write_text(
                json.dumps(
                    {
                        "scope": "component exact values and state; not model numerical or 1M generation acceptance",
                        "torch": str(torch.__version__),
                        "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
                        "observations": cls.observations,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

    def equal(self, actual, expected):
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(
            torch.equal(actual, expected),
            {
                "actual": actual[..., :16].tolist(),
                "expected": expected[..., :16].tolist(),
            },
        )

        def digest(tensor):
            return hashlib.sha256(
                tensor.cpu()
                .contiguous()
                .reshape(-1)
                .view(torch.uint8)
                .numpy()
                .tobytes()
            ).hexdigest()

        self.observations.append(
            {
                "test": self.id(),
                "shape": list(actual.shape),
                "actual_sha256": digest(actual),
                "expected_sha256": digest(expected),
            }
        )

    def fixture(self, values, rows=1):
        entries, width = 128, 68
        count = (len(values) + entries - 1) // entries
        stride = ((entries * width + 511) // 512) * 512
        storage = torch.zeros((count + 1, stride), dtype=torch.uint8, device="cuda")
        pages = CompactPages(
            storage[:, : entries * width], CacheRegion.INDEX_K, entries
        )
        dense = torch.zeros((count * entries, 128), dtype=torch.bfloat16, device="cuda")
        dense[: len(values)] = torch.as_tensor(
            values, dtype=torch.bfloat16, device="cuda"
        )[:, None]
        encoded = encode_compact(dense, CacheRegion.INDEX_K)
        encoded.check()
        pages.data[1:].copy_(encoded.output.reshape(count, entries * width))
        table = torch.arange(1, count + 1, dtype=torch.int32, device="cuda")[None, :]
        query = torch.ones((rows, 32, 128), dtype=torch.bfloat16, device="cuda")
        weights = torch.zeros((rows, 32), dtype=torch.bfloat16, device="cuda")
        weights[:, 0] = 1 / 128
        return (
            query,
            weights,
            pages,
            table,
            torch.zeros(rows, dtype=torch.int32, device="cuda"),
        )

    def candidates(self, lengths):
        ids = torch.arange(CANDIDATE_BLOCKS, dtype=torch.int32, device="cuda")[
            None, :
        ].expand(len(lengths), -1)
        return torch.where(ids * 8 < ints(lengths)[:, None], ids, -1).contiguous()

    def test_exact_scores_and_verify_query_causality(self):
        lengths = [0, 1, 3, 7, 8, 9]
        args = self.fixture([1] * 16, rows=6)
        result = score_candidate_tile(
            *args, ints(lengths), self.candidates(lengths), layer=24
        )
        self.equal(result.status, ints([0] * 6))
        positions = torch.arange(
            CANDIDATE_BLOCKS * 8, dtype=torch.int32, device="cuda"
        )[None, :].expand(6, -1)
        valid = positions < ints(lengths)[:, None]
        self.equal(result.positions, torch.where(valid, positions, -1))
        self.equal(
            result.logits,
            torch.where(
                valid, torch.tensor(1, dtype=torch.bfloat16, device="cuda"), -torch.inf
            ),
        )

    def test_top512_has_exact_margin_and_position_order(self):
        values = [0] * 700
        values[85:597] = [1] * 512
        args = self.fixture(values)
        result = select_index_positions(
            *args, ints([700]), layer=2, max_visible_length=700
        )
        result.check()
        self.equal(
            result.topk,
            torch.arange(85, 597, dtype=torch.int32, device="cuda")[None, :],
        )
        self.assertEqual(result.key_owner, 2)
        self.assertIsNone(result.candidate_blocks)

    def test_l20_block_scan_crosses2048_and_pins_partial(self):
        length = 2048 * 8 + 3
        values = [0] * 8 + [1] * (length - 8)
        args = self.fixture(values)
        result = select_index_positions(
            *args, ints([length]), layer=20, max_visible_length=length
        )
        result.check()
        self.equal(
            result.candidate_blocks,
            torch.arange(1, 2049, dtype=torch.int32, device="cuda")[None, :],
        )
        self.assertEqual(result.scorer_calls, 2)
        self.assertEqual(result.max_logits_elements, CANDIDATE_BLOCKS * 8)
        self.assertEqual(result.topk.shape, (1, 512))
        self.assertTrue(torch.all((result.topk >= 8) & (result.topk < length)).item())
        # Reindex only these blocks, with a different query owner and same K owner.
        other = select_index_positions(
            *args,
            ints([length]),
            layer=28,
            max_visible_length=length,
            candidate_blocks=result.candidate_blocks
        )
        other.check()
        self.assertEqual(other.scorer_calls, 1)
        self.assertEqual(other.key_owner, 20)
        self.assertTrue(torch.all(other.topk >= 8).item())

    def test_padding_short_context_and_empty_local_rows(self):
        args = self.fixture([1] * 8)
        result = select_index_positions(*args, ints([3]), layer=8, max_visible_length=8)
        result.check()
        expected = torch.full((1, 512), -1, dtype=torch.int32, device="cuda")
        expected[0, :3] = ints([0, 1, 2])
        self.equal(result.topk, expected)
        query, weights, pages, table, requests = args
        empty = select_index_positions(
            query[:0],
            weights[:0],
            pages,
            table,
            requests[:0],
            ints([]),
            layer=20,
            max_visible_length=8,
        )
        empty.check()
        self.assertEqual(empty.topk.shape, (0, 512))
        self.assertEqual(empty.candidate_blocks.shape, (0, 2048))

    def test_invalid_metadata_is_rejected_without_vendor_oob(self):
        args = self.fixture([1] * 16)
        for ids in ([1, 0], [0, 0], [0, -1, 1], [999999], [-2]):
            with self.subTest(ids=ids):
                candidates = torch.full(
                    (1, CANDIDATE_BLOCKS), -1, dtype=torch.int32, device="cuda"
                )
                candidates[0, : len(ids)] = ints(ids)
                result = score_candidate_tile(*args, ints([16]), candidates, layer=24)
                self.equal(result.status, ints([1]))
                self.assertTrue(torch.all(result.logits == -torch.inf).item())
        query, weights, pages, table, requests = args
        table.zero_()
        result = select_index_positions(
            query,
            weights,
            pages,
            table,
            requests,
            ints([16]),
            layer=14,
            max_visible_length=16,
        )
        with self.assertRaisesRegex(RuntimeError, "rejected"):
            result.check()
        self.assertTrue(torch.all(result.topk == -1).item())

    def test_graph_reads_new_pages_queries_lengths_and_candidates(self):
        args = self.fixture([1] * 128 + [2] * 128, rows=6)
        query, weights, pages, table, requests = args
        lengths = ints([1, 3, 7, 8, 9, 16])
        candidates = self.candidates(lengths.tolist())
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                score_candidate_tile(*args, lengths, candidates, layer=36)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            result = score_candidate_tile(*args, lengths, candidates, layer=36)
        table[0, 0] = 2
        lengths.copy_(ints([2, 4, 8, 9, 10, 17]))
        candidates.copy_(self.candidates(lengths.tolist()))
        query.mul_(2)
        for _ in range(3):
            graph.replay()
            torch.cuda.synchronize()
            eager = score_candidate_tile(*args, lengths, candidates, layer=36)
            self.equal(result.status, ints([0] * 6))
            self.equal(result.positions, eager.positions)
            self.equal(result.logits, eager.logits)
            valid = result.positions >= 0
            self.equal(result.logits[valid], torch.full_like(result.logits[valid], 4))

    def test_owner_capacity_and_candidate_contracts(self):
        args = self.fixture([1] * 8)
        with self.assertRaisesRegex(ValueError, "eight"):
            select_index_positions(*args, ints([8]), layer=21, max_visible_length=8)
        with self.assertRaisesRegex(ValueError, "require"):
            select_index_positions(*args, ints([8]), layer=24, max_visible_length=8)
        query, weights, pages, table, requests = args
        with self.assertRaisesRegex(ValueError, "32 rows"):
            select_index_positions(
                query.expand(QUERY_TILE + 1, -1, -1).contiguous(),
                weights,
                pages,
                table,
                requests,
                ints([8]),
                layer=20,
                max_visible_length=8,
            )
        result = select_index_positions(
            *args, ints([9]), layer=20, max_visible_length=8
        )
        with self.assertRaisesRegex(RuntimeError, "rejected"):
            result.check()


if __name__ == "__main__":
    unittest.main()
