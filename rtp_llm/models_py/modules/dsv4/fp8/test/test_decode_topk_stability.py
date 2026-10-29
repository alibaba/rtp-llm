"""GLM53 decode selection must preserve ties, padding and graph replay order."""

import os
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4.fp8.indexer import _run_decode_topk


def oracle(scores, lengths, k):
    # CPU stable argsort independently defines score-descending/index-ascending
    # membership; attention consumes the selected indices in token order.
    expected = torch.full((scores.shape[0], k), -1, dtype=torch.int32)
    for row, length in enumerate(lengths.cpu().tolist()):
        length = min(max(length, 0), scores.shape[1])
        selected = torch.argsort(
            scores[row, :length].cpu(), descending=True, stable=True
        )[:k]
        expected[row, : selected.numel()] = selected.sort().values.to(torch.int32)
    return expected


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class DecodeTopkStabilityTest(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(
            os.environ,
            {"DSV4_INDEXER_TOPK_CANONICALIZE": "1", "DSV4_PERSISTENT_TOPK": "1"},
        )
        self.env.start()
        self.addCleanup(self.env.stop)
        self.generator = torch.Generator(device="cuda").manual_seed(177)

    def scores(self, rows, width):
        # Positive discrete values create many boundary ties without conflating
        # the radix kernel's signed-zero ordering with the numeric CPU oracle.
        return torch.randint(
            1, 17, (rows, width), device="cuda", generator=self.generator
        ).float()

    def check(self, scores, lengths, k, backend="topk_v3_tie_break", repeats=1):
        out = torch.full((scores.shape[0], k), -7, device="cuda", dtype=torch.int32)
        expected = oracle(scores, lengths, k)
        original = scores.clone()
        for _ in range(repeats):
            _run_decode_topk(scores, lengths, out, backend)
            torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
            torch.testing.assert_close(scores, original, rtol=0, atol=0)

    def test_boundary_ties_are_repeatable_for_every_native_width(self):
        for width in (13000, 65537, 262144):
            for k in (512, 1024, 2048):
                with self.subTest(width=width, topk=k):
                    scores = self.scores(4, width)
                    lengths = torch.tensor(
                        [width, width - 1, k - 1, 0], device="cuda", dtype=torch.int32
                    )
                    # Invalid context slots must never displace valid scores.
                    scores[2, k - 1 :] = 1e9
                    self.check(scores, lengths, k, repeats=5)

    def test_native_short_capture_width_and_empty_batch(self):
        lengths = torch.tensor([32, 1, 0], device="cuda", dtype=torch.int32)
        for k in (512, 1024, 2048):
            with self.subTest(topk=k):
                self.check(self.scores(3, 32), lengths, k)
                self.check(self.scores(0, 32), lengths[:0], k)

    def test_unique_scores_and_legacy_backends_keep_the_selected_set(self):
        width, k = 13000, 1024
        scores = torch.stack(
            [
                torch.randperm(width, device="cuda", generator=self.generator).float()
                for _ in range(4)
            ]
        )
        lengths = torch.tensor([width, 8000, 511, 0], device="cuda", dtype=torch.int32)
        for backend in ("topk_v3_tie_break", "topk_v3", "dsv4_persistent"):
            with self.subTest(backend=backend):
                self.check(scores, lengths, k, backend)

    def test_torch_fallback_preserves_stable_ties_and_padding(self):
        with patch.dict(os.environ, {"DSV4_PERSISTENT_TOPK": "0"}):
            for width, k in ((32, 512), (13000, 1024), (13000, 7)):
                with self.subTest(width=width, topk=k):
                    scores = self.scores(4, width)
                    lengths = torch.tensor(
                        [width, min(k - 1, width), 1, 0],
                        device="cuda",
                        dtype=torch.int32,
                    )
                    self.check(scores, lengths, k, repeats=3)

    def test_graph_replay_uses_changed_scores_and_lengths_including_mtp_rows(self):
        width, k = 32768, 1024
        for rows in (1, 4, 8, 16):
            with self.subTest(rows=rows):
                scores = self.scores(rows, width)
                lengths = torch.full((rows,), width, device="cuda", dtype=torch.int32)
                out = torch.empty(rows, k, device="cuda", dtype=torch.int32)
                for _ in range(3):
                    _run_decode_topk(scores, lengths, out, "topk_v3_tie_break")
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    _run_decode_topk(scores, lengths, out, "topk_v3_tie_break")
                for replay in range(4):
                    changed = self.scores(rows, width)
                    new_lengths = torch.tensor(
                        [
                            (
                                0
                                if (i + replay) % 4 == 0
                                else (
                                    k - 1
                                    if (i + replay) % 4 == 1
                                    else width - replay * 13 - i
                                )
                            )
                            for i in range(rows)
                        ],
                        device="cuda",
                        dtype=torch.int32,
                    )
                    scores.copy_(changed)
                    lengths.copy_(new_lengths)
                    out.fill_(-7)
                    graph.replay()
                    torch.testing.assert_close(
                        out.cpu(), oracle(changed, new_lengths, k), rtol=0, atol=0
                    )


if __name__ == "__main__":
    unittest.main()
