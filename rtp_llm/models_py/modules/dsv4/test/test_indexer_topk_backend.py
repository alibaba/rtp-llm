import os
import unittest
from contextlib import contextmanager

import torch

from rtp_llm.models_py.modules.dsv4.indexer_topk import (
    AutoIndexerTopKBackend,
    FastIndexerTopKBackend,
    PersistentIndexerTopKBackend,
    TorchIndexerTopKBackend,
    get_indexer_topk_backend,
)


@contextmanager
def _env(key: str, value: str):
    old = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old


def _assert_topk_sets(testcase, got, ref):
    testcase.assertEqual(got.shape, ref.shape)
    testcase.assertTrue(
        torch.equal(torch.sort(got, dim=-1).values, torch.sort(ref, dim=-1).values)
    )


def _assert_valid_indices_sorted(testcase, got, lengths):
    got_cpu = got.cpu()
    lengths_cpu = lengths.cpu().view(-1)
    for row, length in zip(got_cpu, lengths_cpu):
        valid_mask = row >= 0
        valid_count = int(valid_mask.sum().item())
        testcase.assertTrue(torch.all(row[:valid_count] >= 0))
        testcase.assertTrue(torch.all(row[valid_count:] == -1))
        testcase.assertTrue(torch.all(row[:valid_count] < int(length.item())))
        if valid_count > 1:
            testcase.assertTrue(
                torch.all(row[:valid_count][1:] >= row[:valid_count][:-1])
            )


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

    def test_dispatch_env(self):
        with _env("DSV4_INDEXER_TOPK_BACKEND", ""):
            os.environ.pop("DSV4_INDEXER_TOPK_BACKEND", None)
            self.assertIsInstance(get_indexer_topk_backend(), AutoIndexerTopKBackend)
        with _env("DSV4_INDEXER_TOPK_BACKEND", "torch"):
            self.assertIsInstance(get_indexer_topk_backend(), TorchIndexerTopKBackend)
        with _env("DSV4_INDEXER_TOPK_BACKEND", "fast"):
            self.assertIsInstance(get_indexer_topk_backend(), FastIndexerTopKBackend)
        with _env("DSV4_INDEXER_TOPK_BACKEND", "persistent"):
            self.assertIsInstance(get_indexer_topk_backend(), PersistentIndexerTopKBackend)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
    def test_auto_backend_topk_512_matches_torch_sets(self):
        torch.manual_seed(1)
        score = torch.randn(3, 2048, device="cuda", dtype=torch.float32)
        lengths = torch.tensor([2048, 777, 13], device="cuda", dtype=torch.int32)
        with _env("DSV4_INDEXER_TOPK_BACKEND", ""):
            os.environ.pop("DSV4_INDEXER_TOPK_BACKEND", None)
            auto = get_indexer_topk_backend().select(score, 512, lengths=lengths)
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
        _assert_valid_indices_sorted(self, persistent, lengths)

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
            self.assertTrue(torch.equal(got, first))
            _assert_valid_indices_sorted(self, got, lengths)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
    def test_persistent_backend_topk_1024_matches_torch_sets(self):
        torch.manual_seed(4)
        score = torch.randn(3, 4096, device="cuda", dtype=torch.float32)
        lengths = torch.tensor([4096, 2048, 33], device="cuda", dtype=torch.int32)
        persistent = PersistentIndexerTopKBackend().select(score, 1024, lengths=lengths)
        ref = TorchIndexerTopKBackend().select(score, 1024, lengths=lengths)
        _assert_topk_sets(self, persistent.cpu(), ref.cpu())
        _assert_valid_indices_sorted(self, persistent, lengths)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
    def test_fast_backend_topk_2048_matches_torch_sets(self):
        torch.manual_seed(3)
        score = torch.randn(4, 4096, device="cuda", dtype=torch.float32)
        lengths = torch.tensor(
            [4096, 3000, 2048, 17], device="cuda", dtype=torch.int32
        )
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


if __name__ == "__main__":
    unittest.main()
