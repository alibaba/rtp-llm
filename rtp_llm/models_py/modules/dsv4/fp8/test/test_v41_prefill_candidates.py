"""Exact prefill candidate selection, packed flags, and graph replay tests."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.dsv4.fp8 import _v41_prefill_candidates as fused


def reference_select(logits, visible, block_size, topk_blocks):
    """Frozen old attention_v41.select_candidate_blocks, including sorted=True."""
    nblocks = (logits.shape[1] + block_size - 1) // block_size
    padded = F.pad(
        logits, (0, nblocks * block_size - logits.shape[1]), value=-torch.inf
    )
    scores = padded.view(logits.shape[0], nblocks, block_size).amax(-1)
    newest = ((visible - 1).clamp_min(0) // block_size).long()
    scores.scatter_(
        1, newest[:, None], torch.where(visible > 0, torch.inf, -torch.inf)[:, None]
    )
    values, ids = scores.topk(min(topk_blocks, nblocks), dim=-1)
    return torch.where(values > -torch.inf, ids, -1).int()


def reference_mask(logits, candidates, block_size):
    nblocks = (logits.shape[1] + block_size - 1) // block_size
    allowed = torch.zeros(
        logits.shape[0], nblocks + 1, dtype=torch.bool, device=logits.device
    )
    ids = torch.where(candidates >= 0, candidates, nblocks).long()
    allowed.scatter_(1, ids, True)
    columns = torch.arange(logits.shape[1], device=logits.device) // block_size
    return logits.masked_fill_(~allowed[:, columns], -torch.inf)


def make_case(rows, width, seed=41):
    torch.manual_seed(seed)
    storage = torch.randn(rows, width + 256, device="cuda")
    logits = storage[:, :width]
    visible_storage = torch.randint(0, width + 1, (rows * 2,), device="cuda")
    visible = visible_storage[::2]
    if rows >= 3:
        visible[:3] = torch.tensor([0, 1, width], device="cuda")
    logits.masked_fill_(
        torch.arange(width, device="cuda")[None, :] >= visible[:, None], -torch.inf
    )
    return logits, visible


def metadata(rows=17, width=32768):
    logits = SimpleNamespace(
        is_cuda=True,
        device=torch.device("cuda"),
        dtype=torch.float32,
        ndim=2,
        shape=(rows, width),
        stride=lambda dim: width + 256 if dim == 0 else 1,
        requires_grad=False,
    )
    visible = SimpleNamespace(
        device=logits.device,
        ndim=1,
        dtype=torch.int64,
        numel=lambda: rows,
        stride=lambda dim: 2,
    )
    return logits, visible


class V41PrefillCandidatesCPU(unittest.TestCase):
    def test_default_gate_and_environment_disable(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(fused.is_supported(*metadata(), 8, 2048))
        with patch.dict(os.environ, {"DSV41_FUSED_PREFILL_CANDIDATES": "0"}):
            self.assertFalse(fused.is_supported(*metadata(), 8, 2048))
        x = torch.empty(3, 41)
        self.assertIsNone(fused.select_candidates(x, torch.ones(3).long(), 8, 5))
        self.assertFalse(fused.mask_candidates(x, torch.ones(3, 1).int(), 8))

    def test_layout_dtype_shape_and_grad_gates(self):
        for field, value in (
            ("dtype", torch.bfloat16),
            ("ndim", 3),
            ("shape", (0, 32768)),
            ("shape", (17, 0)),
            ("shape", (17, (1 << 20) + 1)),
            ("stride", lambda dim: 2),
            ("requires_grad", True),
        ):
            x, visible = metadata()
            setattr(x, field, value)
            with torch.enable_grad(), self.subTest(field=field):
                self.assertFalse(fused.is_supported(x, visible, 8, 2048))
        x, visible = metadata()
        visible.dtype = torch.float32
        self.assertFalse(fused.is_supported(x, visible, 8, 2048))
        self.assertFalse(fused.is_supported(*metadata(), 3, 2048))
        self.assertFalse(fused.is_supported(*metadata(), 8, 0))

    def test_full_cache_memory_budget(self):
        with patch.dict(os.environ, {}, clear=True):
            for tokens, mib in (
                (32768, 4),
                (65536, 16),
                (131072, 64),
                (262144, 256),
                (524288, 1024),
            ):
                self.assertEqual(
                    fused.bitmap_size_bytes(tokens // 4, tokens, 8), mib * 1024**2
                )
                self.assertEqual(
                    fused.bitmap_is_bounded(tokens // 4, tokens, 8), mib <= 256
                )
        with patch.dict(os.environ, {"DSV41_PREFILL_CANDIDATE_FLAGS_MAX_BYTES": "0"}):
            self.assertFalse(fused.bitmap_is_bounded(1, 8, 8))


class V41PrefillCandidatesCUDA(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        self.env = patch.dict(os.environ, {"DSV41_FUSED_PREFILL_CANDIDATES": "1"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def check_case(self, logits, visible, block=8, k=2048):
        before = logits.clone()
        expected = reference_select(logits, visible, block, k)
        result = fused.select_candidates(logits, visible, block, k)
        self.assertIsNotNone(result)
        actual, flags = result
        self.assertEqual(actual.dtype, torch.int32)
        self.assertEqual(flags.dtype, torch.int32)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        rebuilt = fused.build_flags(actual, logits.shape[1], block)
        torch.testing.assert_close(rebuilt, flags, rtol=0, atol=0)
        torch.testing.assert_close(
            logits.view(torch.int32), before.view(torch.int32), rtol=0, atol=0
        )
        reference = reference_mask(before, expected, block)
        self.assertTrue(fused.mask_candidates(logits, flags, block))
        torch.testing.assert_close(
            logits.view(torch.int32), reference.view(torch.int32), rtol=0, atol=0
        )

    @torch.no_grad()
    def test_real_chunk_shapes(self):
        for rows, width in ((1024, 65536), (2048, 32768), (4096, 16384)):
            with self.subTest(rows=rows, width=width):
                self.check_case(*make_case(rows, width))

    @torch.no_grad()
    def test_boundaries_nan_infinity_ties_and_partial_blocks(self):
        for width, block, k in (
            (1, 1, 3),
            (37, 8, 7),
            (2059, 8, 257),
            (32769, 8, 2048),
        ):
            with self.subTest(width=width):
                logits, visible = make_case(11, width)
                visible[:] = width
                visible[0] = 0
                logits[0] = -torch.inf
                logits[1] = 0
                logits[2] = -torch.inf
                logits[3] = float("nan")
                logits[4] = torch.inf
                logits[5, ::2] = -0.0
                logits[5, 1::2] = 0.0
                logits[6, : min(width, 33)] = float("nan")
                self.check_case(logits, visible, block, k)

    @torch.no_grad()
    def test_sliced_outputs_extra_words_and_padding(self):
        rows, width = 17, 2059
        logits, visible = make_case(rows, width)
        reference = reference_select(logits, visible, 8, 257)
        out_base = torch.full((rows + 4, 280), 123, dtype=torch.int32, device="cuda")
        flag_base = torch.full((rows + 4, 23), 123, dtype=torch.int32, device="cuda")
        out, flags = out_base[2:-2, :270], flag_base[2:-2, :19]
        result = fused.select_candidates(logits, visible, 8, 257, out=out, flags=flags)
        self.assertIs(result[0], out)
        self.assertIs(result[1], flags)
        torch.testing.assert_close(out[:, :257], reference, rtol=0, atol=0)
        self.assertTrue(torch.all(out[:, 257:] == -1).item())
        self.assertTrue(torch.all(flags[:, fused.bitmap_words(width, 8) :] == 0).item())
        self.assertTrue(torch.all(out_base[:2] == 123).item())
        self.assertTrue(torch.all(flag_base[-2:] == 123).item())
        self.assertTrue(torch.all(flag_base[:, 19:] == 123).item())
        expected = reference_mask(logits.clone(), reference, 8)
        self.assertTrue(fused.mask_candidates(logits, flags, 8))
        torch.testing.assert_close(logits, expected, rtol=0, atol=0)

    @torch.no_grad()
    def test_dynamic_cuda_graph_replay(self):
        rows, width = 17, 2059
        logits, visible = make_case(rows, width)
        out = torch.empty(rows, 257, dtype=torch.int32, device="cuda")
        flags = torch.empty(
            rows, fused.bitmap_words(width, 8), dtype=torch.int32, device="cuda"
        )
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                fused.select_candidates(logits, visible, 8, 257, out=out, flags=flags)
                fused.mask_candidates(logits, flags, 8)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = fused.select_candidates(
                logits, visible, 8, 257, out=out, flags=flags
            )
            fused.mask_candidates(logits, flags, 8)
        for seed in (7, 19, 41, 103):
            source, lengths = make_case(rows, width, seed)
            expected_ids = reference_select(source, lengths, 8, 257)
            expected_logits = reference_mask(source.clone(), expected_ids, 8)
            logits.copy_(source)
            visible.copy_(lengths)
            graph.replay()
            torch.testing.assert_close(captured[0], expected_ids, rtol=0, atol=0)
            torch.testing.assert_close(logits, expected_logits, rtol=0, atol=0)

    @torch.no_grad()
    def test_512k_chunk_bitmap_rebuild_and_duplicate_ids(self):
        # Full-query caching is gated at 1 GiB; chunk rebuilding stays 8 MiB.
        rows, width = 1024, 524288
        storage = torch.randint(
            0, width // 8, (rows + 2, 2053), device="cuda", dtype=torch.int32
        )
        candidates = storage[1:-1, :2048]
        candidates[:, :8] = torch.tensor(
            [-1, 0, 31, 32, 63, 64, 65535, 31], device="cuda", dtype=torch.int32
        )
        flags = fused.build_flags(candidates, width, 8)
        self.assertEqual(flags.numel() * flags.element_size(), 8 * 1024**2)
        # Compare bitmap bits directly without allocating a 2 GiB logits tensor.
        bit_ids = torch.arange(width // 8, device="cuda")
        unpacked = ((flags[:, bit_ids // 32].long() >> (bit_ids % 32)) & 1).bool()
        expected = torch.zeros(rows, width // 8 + 1, dtype=torch.bool, device="cuda")
        expected.scatter_(
            1, torch.where(candidates >= 0, candidates, width // 8).long(), True
        )
        torch.testing.assert_close(unpacked, expected[:, :-1], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
