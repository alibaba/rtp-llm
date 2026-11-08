"""SM120 CP request batching must preserve standalone compressed-K numerics."""

import unittest
from types import SimpleNamespace

import torch

from rtp_llm.models_py.modules.dsv4.fp8.compressor import CompressorFP8
from rtp_llm.models_py.utils.arch import is_sm120


class SM120CPCompressorProjectionTest(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available() or not is_sm120(torch.device("cuda")):
            self.skipTest("SM120 required")
        torch.manual_seed(20260930)

    def test_batch_matches_standalone_requests(self):
        for output_dim in (512, 1024, 2048):
            weight = torch.randn(output_dim, 4096, device="cuda", dtype=torch.bfloat16)
            for lengths in ((10, 10), (2, 10, 18, 34), (10,) * 8, (10, 258), (0, 10)):
                with self.subTest(output_dim=output_dim, lengths=lengths):
                    inputs = [
                        torch.randn(n, 4096, device="cuda", dtype=torch.bfloat16)
                        for n in lengths
                    ]
                    compressor = SimpleNamespace(
                        _cp_ctx=SimpleNamespace(chunk_lengths_per_req=lengths),
                        _wkv_wgate_fused=weight,
                    )
                    expected = torch.cat(
                        [
                            CompressorFP8._project_prefill(
                                SimpleNamespace(_cp_ctx=None, _wkv_wgate_fused=weight),
                                x,
                            )
                            for x in inputs
                        ]
                    )
                    actual = CompressorFP8._project_prefill(
                        compressor, torch.cat(inputs)
                    )
                    self.assertEqual(actual.dtype, torch.float32)
                    self.assertTrue(torch.equal(actual, expected))

    def test_cached_suffix_matches_full_prefill(self):
        x = torch.randn(129, 4096, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(2048, 4096, device="cuda", dtype=torch.bfloat16)
        compressor = SimpleNamespace(_cp_ctx=None, _wkv_wgate_fused=weight)
        expected = CompressorFP8._project_prefill(compressor, x)
        for rows in (1, 4, 17, 63, 64, 65):
            for cp_ctx in (None, SimpleNamespace(chunk_lengths_per_req=(rows,))):
                compressor._cp_ctx = cp_ctx
                self.assertTrue(
                    torch.equal(
                        CompressorFP8._project_prefill(compressor, x[-rows:]),
                        expected[-rows:],
                    )
                )

    def test_rejects_inconsistent_request_chunks(self):
        x = torch.zeros(10, 16, device="cuda", dtype=torch.bfloat16)
        for lengths in ((2, 4), (-2, 12)):
            compressor = SimpleNamespace(
                _cp_ctx=SimpleNamespace(chunk_lengths_per_req=lengths),
                _wkv_wgate_fused=torch.zeros(
                    8, 16, device="cuda", dtype=torch.bfloat16
                ),
            )
            with self.assertRaisesRegex(ValueError, "CP request chunks"):
                CompressorFP8._project_prefill(compressor, x)


if __name__ == "__main__":
    unittest.main()
