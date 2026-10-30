"""Compare fused planning with the original Torch path at page boundaries."""

import unittest
from unittest.mock import patch

import torch
from rtp_llm.models_py import prefill_optimization
from rtp_llm.models_py.modules.dsv4.cp import (
    build_kv_allgather_restore_indices,
    cp_actual_owned_kv_lens,
    cp_padded_local_kv_lens,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_cp_impl import (
    _cp_sharded_slot_mapping,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class CPMetadataTest(unittest.TestCase):
    def compare(self, fn, *args, **kwargs):
        with patch.object(prefill_optimization, "ENABLED", False):
            expected = fn(*args, **kwargs)
        with patch.object(prefill_optimization, "ENABLED", True):
            actual = fn(*args, **kwargs)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(torch.equal(actual, expected))

    def test_lengths_and_restore(self):
        for cp, block in [(1, 1), (3, 17), (8, 64)]:
            for dtype in (torch.int32, torch.int64):
                lens = torch.tensor(
                    [
                        0,
                        1,
                        block - 1,
                        block,
                        block + 1,
                        cp * block - 1,
                        cp * block,
                        cp * block + 1,
                        200003,
                        0,
                    ],
                    device="cuda",
                    dtype=dtype,
                )
                self.compare(cp_padded_local_kv_lens, lens, cp, block)
                for rank in range(cp):
                    self.compare(cp_actual_owned_kv_lens, lens, cp, block, rank)
                for data in (lens, lens[:1], lens[:0], lens[::2], lens.repeat(4)):
                    self.compare(
                        build_kv_allgather_restore_indices, data, cp, block, data.device
                    )

    def test_known_total_avoids_scalar_readback(self):
        for values in ([], [0], [1, 17, 513, 200003]):
            lens = torch.tensor(values, device="cuda", dtype=torch.int64)
            with patch.object(prefill_optimization, "ENABLED", False):
                expected = build_kv_allgather_restore_indices(lens, 8, 64, lens.device)
            with patch.object(prefill_optimization, "ENABLED", True):
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU]
                ) as prof:
                    actual = build_kv_allgather_restore_indices(
                        lens, 8, 64, lens.device, known_total=sum(values)
                    )
                self.assertTrue(torch.equal(actual, expected))
                self.assertNotIn(
                    "aten::_local_scalar_dense", [x.key for x in prof.key_averages()]
                )

    def test_slots(self):
        for kernel_tpb, owner_tpb in [(1, 64), (16, 64), (17, 17)]:
            for dtype in (torch.int32, torch.int64):
                # Includes null pages, negative page sentinels and capacity tails.
                pages = torch.tensor(
                    [[3, 0, -1, 9], [2, 7, 1, 8]], device="cuda", dtype=dtype
                )
                positions = torch.arange(4097, device="cuda", dtype=dtype)
                requests = positions % 2
                for rank in range(3):
                    for table in (pages, pages[:, :3], pages[:, :0]):
                        self.compare(
                            _cp_sharded_slot_mapping,
                            positions,
                            table,
                            requests,
                            kernel_tpb,
                            3,
                            rank,
                            owner_tokens_per_block=owner_tpb,
                        )
                self.compare(
                    _cp_sharded_slot_mapping,
                    positions[:0],
                    pages,
                    requests[:0],
                    kernel_tpb,
                    3,
                    0,
                    owner_tokens_per_block=owner_tpb,
                )

    def test_cpu_fallback(self):
        lens = torch.tensor([0, 17, 513, 200003])
        self.compare(cp_actual_owned_kv_lens, lens, 8, 64, 7)
        self.compare(cp_padded_local_kv_lens, lens, 8, 64)
        self.compare(build_kv_allgather_restore_indices, lens, 8, 64, lens.device)


if __name__ == "__main__":
    unittest.main()
