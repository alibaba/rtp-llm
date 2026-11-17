import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

# Load the real helper without attention/__init__.py, which imports the
# compiled engine. These tensor-only regressions also run on CPU developer
# machines, whereas the existing operator tests exercise the full ROCm path.
_spec = importlib.util.spec_from_file_location(
    "rtp_paged_prefill_metadata",
    Path(__file__).resolve().parents[1] / "paged_prefill_metadata.py",
)
_metadata = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _metadata
_spec.loader.exec_module(_metadata)
prepare_paged_prefill_metadata = _metadata.prepare_paged_prefill_metadata
sanitize_and_pad_block_table = _metadata.sanitize_and_pad_block_table
prepare_host_paged_prefill_metadata = _metadata.prepare_host_paged_prefill_metadata
borrow_prefill_cumulative_lengths = _metadata.borrow_prefill_cumulative_lengths


class BorrowedPrefillLengthsTest(unittest.TestCase):
    def test_reuses_both_runner_tensors_without_arithmetic(self):
        q = torch.tensor([0, 14, 30], dtype=torch.int32)
        k = torch.tensor([0, 830, 1662], dtype=torch.int32)
        inputs = SimpleNamespace(
            is_cuda_graph=False,
            input_lengths=torch.tensor([14, 16]),
            cu_seqlens_device=q,
            cu_kv_seqlens_device=k,
        )
        with patch.object(torch, "cumsum", side_effect=AssertionError("GPU setup")):
            actual = borrow_prefill_cumulative_lengths(inputs, torch.device("cpu"))
        self.assertIs(actual[0], q)
        self.assertIs(actual[1], k)

    def test_graph_missing_wrong_shape_dtype_and_device_keep_fallback(self):
        base = dict(
            is_cuda_graph=False,
            input_lengths=torch.tensor([14]),
            cu_seqlens_device=torch.tensor([0, 14], dtype=torch.int32),
            cu_kv_seqlens_device=torch.tensor([0, 830], dtype=torch.int32),
        )
        for override in (
            {"is_cuda_graph": True},
            {"cu_seqlens_device": None},
            {"cu_kv_seqlens_device": torch.tensor([0], dtype=torch.int32)},
            {"cu_seqlens_device": torch.tensor([0, 14], dtype=torch.int64)},
        ):
            inputs = SimpleNamespace(**dict(base, **override))
            self.assertIsNone(
                borrow_prefill_cumulative_lengths(inputs, torch.device("cpu"))
            )
        self.assertIsNone(
            borrow_prefill_cumulative_lengths(
                SimpleNamespace(**base), torch.device("cuda:0")
            )
        )


class HostPagedPrefillMetadataTest(unittest.TestCase):
    def test_matches_original_for_boundary_shapes(self):
        generator = torch.Generator().manual_seed(20261117)
        for block_size in (8, 16, 32, 64, 128):
            for lengths in (
                [0, 1, block_size, block_size + 1],
                [817, 818, 819, 820],
                [830, 831, 832, 833],
            ):
                with self.subTest(block_size=block_size, lengths=lengths):
                    k = torch.tensor(lengths, dtype=torch.int32)
                    cols = (max(lengths) + block_size - 1) // block_size + 3
                    table = torch.randint(
                        -1,
                        1000,
                        (len(lengths), cols),
                        generator=generator,
                        dtype=torch.int32,
                    )
                    before = table.clone()
                    reference, _ = sanitize_and_pad_block_table(
                        table, k, block_size, max(lengths)
                    )
                    actual = prepare_host_paged_prefill_metadata(
                        table, k, block_size, max(lengths), torch.device("cpu")
                    )
                    self.assertTrue(torch.equal(actual.block_table, reference))
                    self.assertTrue(torch.equal(actual.seqlen_k, k))
                    self.assertEqual(
                        actual.kv_indptr.tolist(), [0] * (len(lengths) + 1)
                    )
                    self.assertEqual(actual.kv_page_indices.tolist(), [0])
                    self.assertTrue(torch.equal(table, before))

    def test_snapshot_owns_storage_and_preserves_invalid_valid_ids(self):
        table = torch.tensor([[2, -1, 7]], dtype=torch.int32)
        lengths = torch.tensor([48], dtype=torch.int32)
        first = prepare_host_paged_prefill_metadata(
            table, lengths, 16, 48, torch.device("cpu"), fp8=True
        )
        table.fill_(99)
        lengths.fill_(16)
        second = prepare_host_paged_prefill_metadata(
            table, lengths, 16, 16, torch.device("cpu")
        )
        self.assertEqual(first.block_table[0, :3].tolist(), [2, -1, 7])
        self.assertEqual(first.seqlen_k.tolist(), [48])
        self.assertNotEqual(
            first.host_staging.data_ptr(), second.host_staging.data_ptr()
        )
        self.assertEqual(first.descale.tolist(), [1.0])
        self.assertEqual(first.descale.dtype, torch.float32)

    def test_strided_host_table(self):
        table = torch.arange(20, dtype=torch.int32).reshape(2, 10)[:, ::2]
        lengths = torch.tensor([17, 65], dtype=torch.int32)
        actual = prepare_host_paged_prefill_metadata(
            table, lengths, 16, 65, torch.device("cpu")
        )
        expected, _ = sanitize_and_pad_block_table(table, lengths, 16, 65)
        self.assertTrue(torch.equal(actual.block_table, expected))

    def test_rejects_invalid_geometry_without_hiding_errors(self):
        for table, lengths, block_size, maximum in (
            (
                torch.ones(1, 1, dtype=torch.int32),
                torch.tensor([17], dtype=torch.int32),
                16,
                17,
            ),
            (
                torch.ones(1, 1, dtype=torch.int32),
                torch.tensor([-1], dtype=torch.int32),
                16,
                16,
            ),
            (
                torch.ones(1, 1, dtype=torch.int32),
                torch.tensor([1], dtype=torch.int32),
                0,
                16,
            ),
            (
                torch.ones(1, 0, dtype=torch.int32),
                torch.tensor([0], dtype=torch.int32),
                16,
                16,
            ),
            (
                torch.ones(1, 1, dtype=torch.int32),
                torch.tensor([1], dtype=torch.int32),
                16,
                0,
            ),
        ):
            with self.assertRaises(ValueError):
                prepare_host_paged_prefill_metadata(
                    table, lengths, block_size, maximum, torch.device("cpu")
                )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/ROCm")
    def test_async_transfer_retains_sources_across_batches(self):
        device = torch.device("cuda", torch.cuda.current_device())
        stream = torch.cuda.Stream()
        snapshots = []
        with torch.cuda.stream(stream):
            for block in range(1, 65):
                table = torch.tensor([[block, block + 1, -1]], dtype=torch.int32)
                lengths = torch.tensor([17], dtype=torch.int32)
                snapshots.append(
                    prepare_host_paged_prefill_metadata(table, lengths, 16, 17, device)
                )
                table.zero_()
                lengths.zero_()
        stream.synchronize()
        for block, result in enumerate(snapshots, 1):
            self.assertTrue(result.host_staging.is_pinned())
            self.assertEqual(
                result.block_table.cpu()[0, :3].tolist(), [block, block + 1, block + 1]
            )
            self.assertEqual(result.seqlen_k.cpu().tolist(), [17])


class PagedPrefillMetadataTest(unittest.TestCase):
    def test_padding_retains_valid_ids_and_covers_speculative_prefetch(self):
        for block_size in (8, 16, 32, 64):
            with self.subTest(block_size=block_size):
                lengths = torch.tensor(
                    [1, block_size + 1, 3 * block_size], dtype=torch.int32
                )
                table = torch.tensor(
                    [[9, -1, -1], [4, 7, -1], [3, 5, 8]], dtype=torch.int32
                )
                before = table.clone()
                actual, _ = sanitize_and_pad_block_table(
                    table, lengths, block_size, 3 * block_size
                )
                self.assertEqual(
                    actual.shape[1], 3 + (128 + block_size - 1) // block_size
                )
                self.assertEqual(actual[0].tolist(), [9] * actual.shape[1])
                self.assertEqual(actual[1].tolist(), [4] + [7] * (actual.shape[1] - 1))
                self.assertEqual(actual[2, :3].tolist(), [3, 5, 8])
                self.assertTrue(torch.equal(table, before))

    def test_valid_invalid_ids_are_not_hidden(self):
        table = torch.tensor([[2, -1, 7]], dtype=torch.int32)
        actual, _ = sanitize_and_pad_block_table(table, torch.tensor([48]), 16, 48)
        self.assertEqual(actual[0, :3].tolist(), [2, -1, 7])

    def test_uses_existing_lengths_and_prepares_dummy_inputs_once(self):
        lengths = torch.tensor([17, 33], dtype=torch.int32)
        cu = torch.tensor([0, 17, 50], dtype=torch.int32)
        table = torch.tensor([[2, 4, -1], [8, 10, 12]], dtype=torch.int32)
        params = prepare_paged_prefill_metadata(cu, table, 16, 33, seqlen_k=lengths)
        self.assertIs(params.seqlen_k, lengths)
        self.assertEqual(params.kv_indptr.tolist(), [0, 0, 0])
        self.assertEqual(params.kv_page_indices.tolist(), [0])
        self.assertIsNone(params.descale)

    def test_new_batches_and_cache_groups_do_not_reuse_sanitized_tables(self):
        table = torch.tensor([[3, -1, -1]], dtype=torch.int32)
        old = prepare_paged_prefill_metadata(torch.tensor([0, 1]), table, 16, 1)
        old_table = old.block_table.clone()
        table[0] = torch.tensor([6, 9, -1])
        new = prepare_paged_prefill_metadata(torch.tensor([0, 17]), table, 16, 17)
        self.assertTrue(torch.equal(old.block_table, old_table))
        self.assertNotEqual(old.block_table.data_ptr(), new.block_table.data_ptr())
        self.assertEqual(new.block_table[0, :3].tolist(), [6, 9, 9])

    def test_fp8_scale_is_batch_owned(self):
        params = prepare_paged_prefill_metadata(
            torch.tensor([0, 16]), torch.tensor([[1]]), 16, 16, fp8=True
        )
        self.assertEqual(params.descale.dtype, torch.float32)
        self.assertEqual(params.descale.tolist(), [1.0])


if __name__ == "__main__":
    unittest.main()
