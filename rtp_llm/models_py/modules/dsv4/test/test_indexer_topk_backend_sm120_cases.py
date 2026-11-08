"""SM120 cases split out of test_indexer_topk_backend.py.
These cases live in this sibling file, leaving the original test file
untouched."""

import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4.indexer_topk import (
    AutoIndexerTopKBackend,
    FastIndexerTopKBackend,
    PersistentIndexerTopKBackend,
    TorchIndexerTopKBackend,
    get_indexer_topk_backend,
)
from rtp_llm.models_py.utils.arch import is_sm120


def _assert_topk_sets(testcase, got, ref):
    testcase.assertEqual(got.shape, ref.shape)
    testcase.assertTrue(
        torch.equal(torch.sort(got, dim=-1).values, torch.sort(ref, dim=-1).values)
    )


class TestIndexerTopKBackendSm120(unittest.TestCase):
    """Sibling of ``TestIndexerTopKBackend`` carrying the SM120 cases."""

    def test_auto_sm120_missing_persistent_provider_fails_loudly(self):
        score = torch.randn(2, 4096, device="cuda", dtype=torch.float32)
        backend = AutoIndexerTopKBackend()
        with patch(
            "rtp_llm.models_py.modules.dsv4.indexer_topk.is_sm120", return_value=True
        ), patch.object(
            backend._persistent,
            "select",
            side_effect=AttributeError("provider missing"),
        ):
            with self.assertRaisesRegex(RuntimeError, "dsv4_persistent_topk"):
                backend.select(score, 2048)

    def test_backend_selection_is_always_auto(self):
        self.assertIsInstance(get_indexer_topk_backend(), AutoIndexerTopKBackend)

    def test_fast_backend_topk_2048_architecture_contract(self):
        torch.manual_seed(3)
        score = torch.randn(4, 4096, device="cuda", dtype=torch.float32)
        lengths = torch.tensor([4096, 3000, 2048, 17], device="cuda", dtype=torch.int32)
        if is_sm120(score.device):
            with patch(
                "rtp_llm.models_py.kernels.cuda.fast_topk.fast_topk_v2"
            ) as kernel, self.assertRaisesRegex(RuntimeError, "SM120 shared memory"):
                FastIndexerTopKBackend().select(score, 2048, lengths=lengths)
            kernel.assert_not_called()
            return
        fast = FastIndexerTopKBackend().select(score, 2048, lengths=lengths)
        ref = TorchIndexerTopKBackend().select(score, 2048, lengths=lengths)
        _assert_topk_sets(self, fast.cpu(), ref.cpu())

    def test_persistent_provider_is_scoped_to_sm120(self):
        from rtp_llm.ops.compute_ops import rtp_llm_ops

        score = torch.randn(2, 1024, device="cuda", dtype=torch.float32)
        for sm120 in (False, True):
            with self.subTest(sm120=sm120), patch(
                "rtp_llm.models_py.modules.dsv4.indexer_topk.is_sm120",
                return_value=sm120,
            ), patch.object(rtp_llm_ops, "persistent_topk") as legacy, patch.object(
                rtp_llm_ops, "dsv4_persistent_topk"
            ) as specialized:

                def fill_output(_score, _lengths, out, *_args):
                    out.fill_(7)

                selected = specialized if sm120 else legacy
                selected.side_effect = fill_output
                result = PersistentIndexerTopKBackend().select(score, 512)
                selected.assert_called_once()
                (legacy if sm120 else specialized).assert_not_called()
                torch.testing.assert_close(result, torch.full_like(result, 7))


if __name__ == "__main__":
    import unittest

    unittest.main()
