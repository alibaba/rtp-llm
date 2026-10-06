import unittest
from types import SimpleNamespace

import torch

from rtp_llm.models_py.modules.hybrid.cp_host_metadata import cp_planning_host_mirrors


class TestCPHostMetadata(unittest.TestCase):
    def info(self, chunks, shuffle):
        return SimpleNamespace(
            prefill_cp_chunk_lengths=chunks.clone(),
            prefill_shuffle_indices=shuffle.clone(),
            prefill_cp_chunk_lengths_cpu=chunks,
            prefill_shuffle_indices_cpu=shuffle,
        )

    def test_original_owners_and_padding_are_preserved(self):
        for sizes in ([0], [2, 4, 8, 2], [8] * 20):
            chunks = torch.tensor(sizes, dtype=torch.int32)
            shuffle = torch.arange(sum(sizes), dtype=torch.int32)
            if shuffle.numel():
                shuffle[-1] = -1
            info = self.info(chunks, shuffle)
            actual = cp_planning_host_mirrors(info)
            self.assertIs(actual[0], chunks)
            self.assertIs(actual[1], shuffle)
            self.assertEqual(actual[1].tolist(), shuffle.tolist())

    def test_legacy_and_undefined_mirrors_use_fallback(self):
        self.assertIsNone(cp_planning_host_mirrors(SimpleNamespace()))
        self.assertIsNone(
            cp_planning_host_mirrors(
                SimpleNamespace(
                    prefill_cp_chunk_lengths_cpu=None, prefill_shuffle_indices_cpu=None
                )
            )
        )

    def test_invalid_or_partial_mirror_is_rejected(self):
        chunks = torch.tensor([2, 4], dtype=torch.int32)
        shuffle = torch.arange(6, dtype=torch.int32)
        for field, invalid in (
            ("prefill_cp_chunk_lengths_cpu", None),
            ("prefill_shuffle_indices_cpu", shuffle[:-1]),
            ("prefill_cp_chunk_lengths_cpu", chunks.float()),
            ("prefill_cp_chunk_lengths_cpu", chunks.reshape(1, 2)),
            ("prefill_shuffle_indices_cpu", torch.arange(12, dtype=torch.int32)[::2]),
        ):
            with self.subTest(field=field):
                info = self.info(chunks, shuffle)
                setattr(info, field, invalid)
                with self.assertRaises(ValueError):
                    cp_planning_host_mirrors(info)

    def test_forward_owners_are_not_global_or_reused(self):
        first = self.info(
            torch.tensor([2], dtype=torch.int32),
            torch.tensor([0, -1], dtype=torch.int32),
        )
        second = self.info(
            torch.tensor([4], dtype=torch.int32), torch.arange(4, dtype=torch.int32)
        )
        a, b = cp_planning_host_mirrors(first), cp_planning_host_mirrors(second)
        b[1].fill_(7)
        self.assertEqual(a[1].tolist(), [0, -1])


if __name__ == "__main__":
    unittest.main()
