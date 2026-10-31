"""RTX PRO 5000 gate: native FP4 prefill compressed logits give the same TopK as dense logits."""

from __future__ import annotations

import os
import unittest
from unittest import mock

import torch

from rtp_llm.models_py.modules.dsv4.fp8 import indexer
from rtp_llm.models_py.utils.arch import is_sm120


def _batch(device, requests, rows_per_request, keys_per_request, heads=64, seed=0):
    """Random FP4 Q/K with request-concatenated keys and causal-like row windows."""
    gen = torch.Generator(device=device).manual_seed(seed)
    rows = requests * rows_per_request
    keys = requests * keys_per_request
    q_payload = torch.randint(
        0, 256, (rows, heads, 64), dtype=torch.uint8, device=device, generator=gen
    )
    q_scales = torch.randint(
        118, 130, (rows, heads, 4), dtype=torch.uint8, device=device, generator=gen
    )
    k_payload = torch.randint(
        0, 256, (keys, 64), dtype=torch.uint8, device=device, generator=gen
    )
    k_scales = torch.randint(
        118, 130, (keys, 4), dtype=torch.uint8, device=device, generator=gen
    )
    weights = torch.randn(
        (rows, heads), dtype=torch.float32, device=device, generator=gen
    )
    starts = torch.empty(rows, dtype=torch.int32, device=device)
    ends = torch.empty(rows, dtype=torch.int32, device=device)
    tail = torch.arange(rows_per_request, device=device) // 4
    for b in range(requests):
        lo, hi = b * rows_per_request, (b + 1) * rows_per_request
        starts[lo:hi] = b * keys_per_request
        window = (keys_per_request - rows_per_request // 4 + tail).clamp(
            max=keys_per_request
        )
        ends[lo:hi] = b * keys_per_request + window.to(torch.int32)
    return (q_payload, q_scales), weights, k_payload, k_scales, starts, ends


class Fp4PrefillCompressedLogitsTest(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.fail("SM120 hardware gate was scheduled without a CUDA device")
        if not is_sm120():
            self.fail(
                "SM120 hardware gate was scheduled on "
                f"compute capability {torch.cuda.get_device_capability()}"
            )
        self.device = torch.device("cuda", torch.cuda.current_device())

    def _topk(self, compressed, q, weights, k_payload, k_scales, starts, ends):
        from rtp_llm.models_py.modules.dsv4.fp4.native import prefill_score

        top_k = 512
        out = torch.empty(
            (starts.numel(), top_k), dtype=torch.int32, device=self.device
        )
        if compressed:
            width = int((ends - starts).max())
            logits = prefill_score(
                q, weights, k_payload, k_scales, starts, ends, max_seqlen_k=width
            )
            self.assertEqual(logits.shape[1], width)
            window = (ends - starts).contiguous()
            indexer._run_prefill_topk(
                logits, torch.zeros_like(window), window, out, top_k, 4
            )
        else:
            logits = prefill_score(q, weights, k_payload, k_scales, starts, ends)
            self.assertEqual(logits.shape[1], k_payload.shape[0])
            indexer._run_prefill_topk(logits, starts, ends, out, top_k, 4)
        return out

    def test_compressed_topk_matches_dense(self) -> None:
        # 4 requests stay under the 32768-wide Triton canonicalize limit in the
        # dense layout; 6 and 8 requests exceed it and take the torch fallback.
        with mock.patch.dict(os.environ, {"DSV4_INDEXER_TOPK_CANONICALIZE": "1"}):
            for requests, rows_per_request in ((4, 512), (6, 256), (8, 256)):
                with self.subTest(requests=requests):
                    case = _batch(self.device, requests, rows_per_request, 8191)
                    dense = self._topk(False, *case)
                    compressed = self._topk(True, *case)
                    self.assertTrue(torch.equal(dense, compressed))

    def test_switch_defaults_off(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DSV4_FP4_PREFILL_COMPRESSED_LOGITS", None)
            self.assertFalse(indexer._prefill_compressed_logits_enabled())
        with mock.patch.dict(os.environ, {"DSV4_FP4_PREFILL_COMPRESSED_LOGITS": "1"}):
            self.assertTrue(indexer._prefill_compressed_logits_enabled())


if __name__ == "__main__":
    unittest.main()
