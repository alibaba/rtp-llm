"""Sparse checkpoint destinations use the existing DSV4 pool layouts."""

import unittest

import torch

from rtp_llm.models_py.modules.dsv4.fp8.sparse_save import (
    SparseSaveTargets,
    iter_sparse_full_kv_slots,
    iter_sparse_save_slots,
)


class SparseSaveSlotsTest(unittest.TestCase):
    def test_f_last_and_n_keep_distinct_state_destinations_in_one_slot(self) -> None:
        positions = torch.arange(1300, 1340, dtype=torch.int64)
        targets = SparseSaveTargets(
            torch.tensor([[1340, 1320]], dtype=torch.int64),
            torch.tensor([[18, 17]], dtype=torch.int32),
        )
        mapped = list(
            iter_sparse_save_slots(
                targets,
                positions,
                torch.zeros_like(positions),
                entries_per_block=128,
                tokens_per_block=256,
            )
        )
        self.assertEqual([end for end, _ in mapped], [1340, 1320])
        n_slots, f_slots = mapped[0][1], mapped[1][1]
        self.assertEqual(int((n_slots >= 0).sum()), 40)
        self.assertEqual(int((f_slots >= 0).sum()), 20)
        self.assertEqual(int(n_slots[0]), 18 * 128 + 1300 % 128)
        self.assertEqual(int(f_slots[0]), 17 * 128 + 1300 % 128)
        self.assertTrue(torch.all(f_slots[20:] == -1))

    def test_state_short_suffix_keeps_runtime_independent(self) -> None:
        positions = torch.arange(1300, 1320, dtype=torch.int64)
        req = torch.zeros_like(positions)
        targets = SparseSaveTargets(
            torch.tensor([[1301, 1320]], dtype=torch.int64),
            torch.tensor([[17, 18]], dtype=torch.int32),
        )
        mapped = list(
            iter_sparse_save_slots(
                targets,
                positions,
                req,
                entries_per_block=128,
                tokens_per_block=256,
                runtime_block_table=torch.tensor(
                    [[0, 0, 0, 0, 0, 19]], dtype=torch.int32
                ),
            )
        )
        self.assertEqual([p for p, _ in mapped], [1301, 1320])
        self.assertEqual(mapped[0][1][0].item(), 17 * 128 + 1300 % 128)
        self.assertTrue(torch.all(mapped[0][1][1:] == -1))
        self.assertTrue(torch.all(mapped[1][1] >= 0))

    def test_cp2_state_owner_and_cross_slot_history(self) -> None:
        positions = torch.tensor([1024, 1025], dtype=torch.int64)
        req = torch.zeros_like(positions)
        targets = SparseSaveTargets(
            torch.tensor([[1026]], dtype=torch.int64),
            torch.tensor([[7]], dtype=torch.int32),
        )
        rank0 = list(
            iter_sparse_save_slots(
                targets,
                positions,
                req,
                entries_per_block=4,
                tokens_per_block=512,
                cp_size=2,
                cp_rank=0,
            )
        )[0][1]
        rank1 = list(
            iter_sparse_save_slots(
                targets,
                positions,
                req,
                entries_per_block=4,
                tokens_per_block=512,
                cp_size=2,
                cp_rank=1,
            )
        )[0][1]
        self.assertEqual(rank0.tolist(), [28, 29])
        self.assertEqual(rank1.tolist(), [-1, -1])

    def test_full_kv_expands_physical_destination_and_boundary(self) -> None:
        positions = torch.arange(1300, 1544, dtype=torch.int64)
        targets = SparseSaveTargets(
            torch.tensor([[1544]], dtype=torch.int64),
            torch.tensor([[9]], dtype=torch.int32),
        )
        slots = list(
            iter_sparse_full_kv_slots(
                targets,
                positions,
                torch.zeros_like(positions),
                physical_tokens_per_block=256,
                kernel_tokens_per_block=64,
                entries_per_kernel_block=16,
                ratio=4,
            )
        )[0][1]
        self.assertEqual(int((slots >= 0).sum()), 2)
        self.assertEqual(slots[-5].item(), 9 * 4 * 16)
        self.assertEqual(slots[-1].item(), 9 * 4 * 16 + 1)

    def test_missing_optional_destination_yields_no_work(self) -> None:
        targets = SparseSaveTargets(
            torch.tensor([[1301, -1]], dtype=torch.int64),
            torch.tensor([[-1, -1]], dtype=torch.int32),
        )
        self.assertEqual(
            list(
                iter_sparse_save_slots(
                    targets,
                    torch.tensor([1300]),
                    torch.tensor([0]),
                    entries_per_block=128,
                    tokens_per_block=256,
                )
            ),
            [],
        )

    def test_cuda_alias_decisions_are_capture_safe(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        device = torch.device("cuda")
        positions = torch.arange(1300, 1320, dtype=torch.int64, device=device)
        req = torch.zeros_like(positions)
        targets = SparseSaveTargets(
            torch.tensor([[1320]], dtype=torch.int64),
            torch.tensor([[3]], dtype=torch.int32),
        )
        runtime = torch.tensor([[0, 0, 0, 0, 0, 3]], dtype=torch.int32, device=device)
        full_positions = torch.arange(1536, 1544, dtype=torch.int64, device=device)
        full_req = torch.zeros_like(full_positions)
        full_targets = SparseSaveTargets(
            torch.tensor([[1544]], dtype=torch.int64),
            torch.tensor([[9]], dtype=torch.int32),
        )
        full_runtime = torch.zeros((1, 28), dtype=torch.int32, device=device)
        full_runtime[0, 24] = 9 * 4

        # CUDA scalar .item() is illegal during graph capture. Both alias
        # paths must produce all-invalid slots entirely on device.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            state_slots = list(
                iter_sparse_save_slots(
                    targets,
                    positions,
                    req,
                    entries_per_block=128,
                    tokens_per_block=256,
                    runtime_block_table=runtime,
                )
            )[0][1]
            full_slots = list(
                iter_sparse_full_kv_slots(
                    full_targets,
                    full_positions,
                    full_req,
                    physical_tokens_per_block=256,
                    kernel_tokens_per_block=64,
                    entries_per_kernel_block=16,
                    ratio=4,
                    runtime_block_table=full_runtime,
                )
            )[0][1]
        graph.replay()
        self.assertTrue(torch.all(state_slots == -1).item())
        self.assertTrue(torch.all(full_slots == -1).item())


if __name__ == "__main__":
    unittest.main()
