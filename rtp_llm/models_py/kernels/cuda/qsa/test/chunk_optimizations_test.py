"""Independent numerical checks for QSA's optional cache specializations."""

import itertools
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.kernels.cuda.qsa import chunk_paged_gqa as chunk


@triton.jit
def _convert_codes(X, Y):
    i = tl.arange(0, 256)
    x = tl.load(X + i)
    tl.store(Y + i, chunk._fp8_to_bf16(x))


class ChunkToolchainTest(unittest.TestCase):
    def test_unsupported_triton_rejected_before_reading_inputs(self):
        for version in ("3.4.0", "3.4.0+vendor", "3.5.1"):
            with self.subTest(version=version):
                with patch.object(triton, "__version__", version):
                    with self.assertRaisesRegex(
                        RuntimeError, r"Triton >= 3\.6"
                    ) as error:
                        chunk.chunk_paged_gqa(None, None, None, None, None)
                    self.assertIn(version, str(error.exception))
                    self.assertIn("CUDA 13", str(error.exception))

    def test_unsupported_triton_rejected_before_legacy_ptxas_query(self):
        # Triton 3.4 exposes the no-argument API. The version error must precede
        # that API call, rather than leaking a get_ptxas() TypeError.
        def legacy_get_ptxas():
            return SimpleNamespace(version="12.9")

        with patch.object(triton, "__version__", "3.4.0"), patch(
            "triton.backends.nvidia.compiler.get_ptxas", legacy_get_ptxas
        ):
            with self.assertRaisesRegex(RuntimeError, r"Triton >= 3\.6"):
                chunk._require_fp8_toolchain()

    def test_supported_triton_checks_selected_ptxas(self):
        for version in ("3.6.0", "3.6.0+vendor", "3.10.0"):
            for ptxas in ("12.9", "13.1", "13.2"):
                with self.subTest(version=version, ptxas=ptxas):
                    with patch.object(triton, "__version__", version), patch(
                        "triton.backends.nvidia.compiler.get_ptxas",
                        return_value=SimpleNamespace(version=ptxas),
                    ):
                        if ptxas == "13.2":
                            chunk._require_fp8_toolchain()
                        else:
                            with self.assertRaisesRegex(
                                RuntimeError, r"CUDA 13\.2"
                            ) as error:
                                chunk._require_fp8_toolchain()
                            self.assertIn(
                                "TRITON_PTXAS_BLACKWELL_PATH", str(error.exception)
                            )


class ChunkOptimizationsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (
            12,
            0,
        ):
            raise unittest.SkipTest("SM120 is required")
        chunk._require_fp8_toolchain()

    def fixture(self, packed=True, unit=False):
        torch.manual_seed(749)
        b, h, qlen, d = 2, 12, 17, 256
        q = torch.randn(b, h, qlen, d, device="cuda", dtype=torch.bfloat16)
        # HybridPool's page stride can be larger than its HND payload.
        storage = torch.randn(7, 2 * 128 * d + 256, device="cuda").to(
            torch.float8_e4m3fn
        )
        cache = storage[:, : 2 * 128 * d].view(7, 2, 1, 128, d)
        scale = torch.rand(7, 2, 1, 128, device="cuda") * 0.8 + 0.2
        if unit:
            scale.fill_(1)
        table = torch.tensor([[3, 6, 1], [5, 2, 4]], device="cuda", dtype=torch.int32)
        lengths = torch.full((b, qlen), 367, device="cuda", dtype=torch.int32)
        counts = [0, 1, 2, 3, 4, 5, 31, 32, 33, 63, 64, 65, 127, 128, 129, 2048, 2051]
        selected = torch.full(
            (b, qlen, 2052 if packed else 2051), -1, device="cuda", dtype=torch.int32
        )
        for row, count in enumerate(counts):
            blocks, tail = divmod(count, 4)
            ids = ((torch.arange(blocks, device="cuda") * 31 + 31) % 91) * 4
            selected[:, row, : blocks * 4] = (
                ids[:, None] + torch.arange(4, device="cuda")
            ).flatten()
            tail_start = blocks * 4 if packed else 2048
            selected[:, row, tail_start : tail_start + tail] = 364 + torch.arange(
                tail, device="cuda"
            )
            if packed:
                selected[:, row, -1] = count
        return q, cache, table, lengths, selected, scale

    def reference(self, args, packed, unit=False):
        q, cache, table, lengths, selected, scale = args
        out = torch.zeros_like(q, dtype=torch.float64)
        for b in range(q.shape[0]):
            for row in range(q.shape[2]):
                ids = selected[b, row, :2051]
                if packed:
                    ids = ids[: int(selected[b, row, -1])]
                ids = ids[(ids >= 0) & (ids < lengths[b, row])].long()
                page = table[b, ids // 128].long()
                good = (page > 0) & (page < cache.shape[0])
                ids, page = ids[good], page[good]
                if not ids.numel():
                    continue
                k = cache[page, 0, 0, ids % 128].double()
                v = cache[page, 1, 0, ids % 128].double()
                if not unit:
                    k *= scale[page, 0, 0, ids % 128, None].double()
                    v *= scale[page, 1, 0, ids % 128, None].double()
                out[b, :, row] = (q[b, :, row].double() @ k.T / 16).softmax(-1) @ v
        return out

    def run_chunk(self, args, packed=True, **kwargs):
        return chunk.chunk_paged_gqa(
            *args[:5],
            scale=args[5],
            selected_has_count=packed,
            block_n=64,
            num_warps=2,
            num_stages=1,
            **kwargs,
        )

    def test_direct_conversion_preserves_all_finite_fp8_codes(self):
        raw = torch.arange(256, device="cuda", dtype=torch.int32).to(torch.uint8)
        x = raw.view(torch.float8_e4m3fn)
        actual = torch.empty(256, device="cuda", dtype=torch.bfloat16)
        _convert_codes[(1,)](x, actual)
        expected = x.bfloat16()
        finite = ~expected.isnan()
        torch.testing.assert_close(
            actual.view(torch.int16)[finite],
            expected.view(torch.int16)[finite],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(actual.isnan(), expected.isnan(), rtol=0, atol=0)

    def test_fp8_rejects_old_ptxas_before_launch(self):
        args = self.fixture(unit=True)
        with patch(
            "triton.backends.nvidia.compiler.get_ptxas",
            return_value=SimpleNamespace(version="12.9"),
        ):
            with self.assertRaisesRegex(RuntimeError, r"CUDA 13\.2"):
                self.run_chunk(args, unit_scale=True)

    def test_fp8_rejects_non_sm120_device(self):
        args = self.fixture(unit=True)
        with patch("torch.cuda.get_device_capability", return_value=(9, 0)):
            with self.assertRaisesRegex(RuntimeError, "SM120"):
                self.run_chunk(args, unit_scale=True)

    def test_cache_contracts_preserve_partial_blocks_scales_and_padded_pages(self):
        for packed, unit in itertools.product((False, True), repeat=2):
            args = self.fixture(packed, unit)
            # A live length can bisect a selected block. Missing pages mask out.
            args[3][0, 5] = 126
            args[2][1, 1] = 0
            expected = self.reference(args, packed)
            baseline = self.run_chunk(args, packed)
            # Single BF16 PV can cross a pointwise bound on tiny selections;
            # these optimizations must preserve the baseline bits exactly.
            self.assertLess(
                float((baseline.double() - expected).norm() / expected.norm()), 8e-3
            )
            if unit:
                with self.subTest(packed=packed):
                    actual = self.run_chunk(args, packed, unit_scale=True)
                    torch.testing.assert_close(actual, baseline, rtol=0, atol=0)

    def test_unit_scale_needs_no_scale_buffer_and_graph_reads_live_metadata(self):
        args = list(self.fixture(True, True))
        args[5] = None
        gate = torch.randn(2, 17, 12, 256, device="cuda", dtype=torch.bfloat16)
        options = dict(unit_scale=True)
        self.run_chunk(args, output_gate=gate, **options)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = self.run_chunk(args, output_gate=gate, **options)
        for length in (129, 367, 0):
            args[2].copy_(args[2].flip(1))
            args[3].fill_(length)
            args[0].mul_(0.75)
            if length == 367:
                # Refresh both block identity and packed count in-place.
                args[4][..., :2051].fill_(-1)
                args[4][..., :4] = torch.arange(4, 8, device="cuda")
                args[4][..., -1].fill_(4)
            elif length == 0:
                args[4][..., :2051].fill_(-1)
                args[4][..., -1].zero_()
            graph.replay()
            ungated = self.run_chunk(args, **options)
            expected = self.reference(args, True, unit=True)
            if length:
                baseline_args = [*args[:5], torch.ones(7, 2, 1, 128, device="cuda")]
                baseline = self.run_chunk(baseline_args)
                torch.testing.assert_close(ungated, baseline, rtol=0, atol=0)
                self.assertLess(
                    float((ungated.double() - expected).norm() / expected.norm()), 8e-3
                )
            else:
                torch.testing.assert_close(
                    ungated, torch.zeros_like(ungated), rtol=0, atol=0
                )
            torch.testing.assert_close(
                actual, ungated * gate.transpose(1, 2).sigmoid(), rtol=0, atol=0
            )
        graph.reset()

    def test_public_wrapper_obeys_explicit_unit_scale_contract(self):
        from rtp_llm.models_py.kernels.cuda.qsa.sparse_paged_fmha import (
            sparse_paged_gqa_attn,
        )

        args = self.fixture(True, True)
        baseline = sparse_paged_gqa_attn(
            *args[:5],
            page_size=128,
            kv_scale=args[5],
            selected_has_count=True,
        )
        # A unity declaration makes scale storage irrelevant to this reader.
        args[5].fill_(float("nan"))
        actual = sparse_paged_gqa_attn(
            *args[:5],
            page_size=128,
            kv_scale=args[5],
            selected_has_count=True,
            kv_scale_is_unity=True,
        )
        torch.testing.assert_close(actual, baseline, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
