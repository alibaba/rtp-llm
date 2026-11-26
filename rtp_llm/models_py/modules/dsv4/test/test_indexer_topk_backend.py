import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4.indexer_topk import (
    AutoIndexerTopKBackend,
    FastIndexerTopKBackend,
    PersistentIndexerTopKBackend,
    TorchIndexerTopKBackend,
    get_indexer_topk_backend,
    select_indexer_topk,
)
from rtp_llm.models_py.utils.arch import is_sm120


def _assert_topk_sets(testcase, got, ref):
    testcase.assertEqual(got.shape, ref.shape)
    testcase.assertTrue(
        torch.equal(torch.sort(got, dim=-1).values, torch.sort(ref, dim=-1).values)
    )


def _assert_valid_indices_compact(testcase, got, lengths):
    got_cpu = got.cpu()
    lengths_cpu = lengths.cpu().view(-1)
    for row, length in zip(got_cpu, lengths_cpu):
        valid_mask = row >= 0
        valid_count = int(valid_mask.sum().item())
        testcase.assertTrue(torch.all(row[:valid_count] >= 0))
        testcase.assertTrue(torch.all(row[valid_count:] == -1))
        testcase.assertTrue(torch.all(row[:valid_count] < int(length.item())))


class TestIndexerTopKBackend(unittest.TestCase):
    def test_torch_backend_masks_short_rows(self):
        score = torch.tensor(
            [
                [1.0, 4.0, 3.0, -9.0],
                [9.0, 8.0, 7.0, 6.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        lengths = torch.tensor([3, 1, 0], dtype=torch.int32)
        out = TorchIndexerTopKBackend().select(score, 3, lengths=lengths, offset=10)
        self.assertEqual(out.dtype, torch.int32)
        self.assertTrue(
            torch.equal(
                out[0].sort().values,
                torch.tensor([10, 12, 11], dtype=torch.int32).sort().values,
            )
        )
        self.assertTrue(
            torch.equal(out[1], torch.tensor([10, -1, -1], dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(out[2], torch.tensor([-1, -1, -1], dtype=torch.int32))
        )

    def test_torch_backend_3d_shape(self):
        torch.manual_seed(1)
        score = torch.randn(2, 3, 8, dtype=torch.float32)
        lengths = torch.tensor([8, 7, 6, 5, 4, 3], dtype=torch.int32)
        out = TorchIndexerTopKBackend().select(score, 4, lengths=lengths)
        self.assertEqual(out.shape, (2, 3, 4))
        self.assertEqual(out.dtype, torch.int32)
        self.assertTrue(torch.all(out.reshape(-1, 4) < lengths.view(-1, 1)))

    def test_torch_backend_empty_3d_score(self):
        score = torch.empty(1, 2, 0, dtype=torch.float32)
        out = TorchIndexerTopKBackend().select(score, 3)
        self.assertEqual(out.shape, (1, 2, 3))
        self.assertEqual(out.dtype, torch.int32)
        self.assertTrue(torch.equal(out, torch.full_like(out, -1)))

        out_with_lengths = TorchIndexerTopKBackend().select(
            score, 3, lengths=torch.zeros(2, dtype=torch.int32)
        )
        self.assertTrue(torch.equal(out_with_lengths, torch.full_like(out, -1)))

        empty_topk = TorchIndexerTopKBackend().select(score, 0)
        self.assertEqual(empty_topk.shape, (1, 2, 0))

    def test_auto_backend_empty_3d_score_uses_safe_torch_path(self):
        score = torch.empty(1, 2, 0, dtype=torch.float32)
        out = select_indexer_topk(score, 512)
        self.assertEqual(out.shape, (1, 2, 512))
        self.assertTrue(torch.equal(out, torch.full_like(out, -1)))

    def test_auto_backend_masks_short_rows_on_torch_path(self):
        score = torch.tensor(
            [
                [0.1, 0.9, 0.5, 4.0, 3.0],
                [0.4, 0.2, 8.0, 7.0, 6.0],
            ],
            dtype=torch.float32,
        )
        lengths = torch.tensor([3, 2], dtype=torch.int32)

        out = AutoIndexerTopKBackend().select(score, 4, lengths=lengths)

        self.assertEqual(out.dtype, torch.int32)
        self.assertTrue(
            torch.equal(out[0], torch.tensor([1, 2, 0, -1], dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(out[1], torch.tensor([0, 1, -1, -1], dtype=torch.int32))
        )

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
    def test_auto_backend_topk_512_matches_torch_sets(self):
        torch.manual_seed(1)
        score = torch.randn(3, 2048, device="cuda", dtype=torch.float32)
        lengths = torch.tensor([2048, 777, 13], device="cuda", dtype=torch.int32)
        auto = AutoIndexerTopKBackend().select(score, 512, lengths=lengths)
        ref = TorchIndexerTopKBackend().select(score, 512, lengths=lengths)
        _assert_topk_sets(self, auto.cpu(), ref.cpu())

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
    def test_persistent_backend_topk_512_matches_torch_sets(self):
        torch.manual_seed(2)
        score = torch.randn(4, 16384, device="cuda", dtype=torch.float32)
        lengths = torch.tensor(
            [16384, 9000, 512, 17], device="cuda", dtype=torch.int32
        )
        persistent = PersistentIndexerTopKBackend().select(score, 512, lengths=lengths)
        ref = TorchIndexerTopKBackend().select(score, 512, lengths=lengths)
        _assert_topk_sets(self, persistent.cpu(), ref.cpu())
        _assert_valid_indices_compact(self, persistent, lengths)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
    def test_persistent_backend_topk_512_is_repeatable(self):
        torch.manual_seed(5)
        score = torch.randn(5, 16384, device="cuda", dtype=torch.float32)
        lengths = torch.tensor(
            [16384, 9000, 4096, 511, 17], device="cuda", dtype=torch.int32
        )
        first = PersistentIndexerTopKBackend().select(score, 512, lengths=lengths)
        for _ in range(5):
            got = PersistentIndexerTopKBackend().select(score, 512, lengths=lengths)
            _assert_topk_sets(self, got.cpu(), first.cpu())
            _assert_valid_indices_compact(self, got, lengths)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
    def test_persistent_backend_topk_1024_matches_torch_sets(self):
        torch.manual_seed(4)
        score = torch.randn(3, 4096, device="cuda", dtype=torch.float32)
        lengths = torch.tensor([4096, 2048, 33], device="cuda", dtype=torch.int32)
        persistent = PersistentIndexerTopKBackend().select(score, 1024, lengths=lengths)
        ref = TorchIndexerTopKBackend().select(score, 1024, lengths=lengths)
        _assert_topk_sets(self, persistent.cpu(), ref.cpu())
        _assert_valid_indices_compact(self, persistent, lengths)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
    def test_fast_backend_topk_2048_matches_torch_sets(self):
        torch.manual_seed(3)
        score = torch.randn(4, 4096, device="cuda", dtype=torch.float32)
        lengths = torch.tensor(
            [4096, 3000, 2048, 17], device="cuda", dtype=torch.int32
        )
        if is_sm120(score.device):
            with self.assertRaisesRegex(RuntimeError, "SM120 shared memory"):
                FastIndexerTopKBackend().select(score, 2048, lengths=lengths)
            return
        fast = FastIndexerTopKBackend().select(score, 2048, lengths=lengths)
        ref = TorchIndexerTopKBackend().select(score, 2048, lengths=lengths)
        _assert_topk_sets(self, fast.cpu(), ref.cpu())

    def test_fast_backend_rejects_unsupported_without_fallback(self):
        score = torch.randn(2, 16, dtype=torch.float32)
        with self.assertRaisesRegex(RuntimeError, "requires CUDA float32"):
            FastIndexerTopKBackend().select(score, 8)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
    def test_fast_backend_rejects_cuda_topk_512(self):
        score = torch.randn(2, 1024, device="cuda", dtype=torch.float32)
        lengths = torch.tensor([1024, 33], device="cuda", dtype=torch.int32)
        with self.assertRaisesRegex(RuntimeError, "topk=2048"):
            FastIndexerTopKBackend().select(score, 512, lengths=lengths)

    def test_persistent_backend_rejects_unsupported_without_fallback(self):
        score = torch.randn(2, 1024, dtype=torch.float32)
        lengths = torch.tensor([1024, 33], dtype=torch.int32)
        with self.assertRaisesRegex(RuntimeError, "requires CUDA float32"):
            PersistentIndexerTopKBackend().select(score, 512, lengths=lengths)


class TestIndexerTopKBackendSm120(unittest.TestCase):
    """SM120 provider selection and shared-memory fallback contract cases."""

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
    unittest.main()
