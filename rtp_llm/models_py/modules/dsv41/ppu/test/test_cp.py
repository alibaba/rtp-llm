"""CP zigzag placement and collective contracts."""

import threading
import unittest

import torch

from rtp_llm.models_py.modules.dsv41.ppu.cp import (
    CPPrefillContext,
    ThreadedCPGroup,
    build_zigzag_plan,
)

CP = 8
JOIN_TIMEOUT = 300

class _PlanCase:
    """A split/gather/restore scenario shared by several assertions."""

    def __init__(self, lengths, cp_size=CP, prefix_lengths=None, seed=11):
        generator = torch.Generator().manual_seed(seed)
        ids = [
            torch.randint(0, 50_000, (length,), generator=generator)
            for length in lengths
        ]
        full_ids = torch.cat(ids) if ids else torch.zeros(0, dtype=torch.int64)
        cu = torch.zeros(len(lengths) + 1, dtype=torch.int64)
        for index, length in enumerate(lengths):
            cu[index + 1] = cu[index] + length
        self.lengths = list(lengths)
        self.full_ids = full_ids
        self.full_cu = cu
        self.cp_size = cp_size
        self.prefix_lengths = prefix_lengths or [0] * len(lengths)
        self.plans = [
            build_zigzag_plan(
                full_ids,
                cu,
                cp_size=cp_size,
                cp_rank=rank,
                prefix_lengths=self.prefix_lengths,
            )
            for rank in range(cp_size)
        ]


class ZigzagSplitTest(unittest.TestCase):
    def _assert_roundtrip(self, case: _PlanCase):
        total = int(case.full_ids.numel())
        # Uniform chunk length (the all-gather shape contract).
        chunks = {plan.chunk_length for plan in case.plans}
        self.assertEqual(len(chunks), 1, f"chunk lengths differ: {chunks}")
        # Every real token is owned by exactly one rank.
        owned = sum(int(plan.local_is_real.sum()) for plan in case.plans)
        self.assertEqual(owned, total)
        restore = case.plans[0].unpad_restore
        self.assertEqual(int(restore.numel()), total)
        self.assertEqual(int(restore.unique().numel()), total, "duplicate owner")
        self.assertTrue(bool((restore >= 0).all()))
        self.assertTrue(
            bool((restore < case.cp_size * case.plans[0].chunk_length).all())
        )
        # gather (rank-major) then restore == identity, in logical order.
        gathered = torch.cat([plan.local_input_ids for plan in case.plans])
        restored = gathered[restore]
        self.assertTrue(torch.equal(restored, case.full_ids))
        # Positions round-trip through the request prefixes.
        prefixes = torch.tensor(case.prefix_lengths, dtype=torch.int64)
        expected_positions = torch.cat(
            [
                torch.arange(case.prefix_lengths[b], case.prefix_lengths[b] + length)
                for b, length in enumerate(case.lengths)
            ]
            or [torch.zeros(0, dtype=torch.int64)]
        )
        gathered_pos = torch.cat([plan.local_positions for plan in case.plans])
        self.assertTrue(
            torch.equal(gathered_pos[restore], expected_positions),
            "restored positions must be the original model order",
        )

    def test_roundtrip_single_request_lengths(self):
        for length in (1, 2, 7, 15, 16, 17, 31, 33, 64, 129):
            with self.subTest(length=length):
                self._assert_roundtrip(_PlanCase([length]))

    def test_roundtrip_multi_request(self):
        self._assert_roundtrip(_PlanCase([3, 16, 20]))
        self._assert_roundtrip(_PlanCase([1, 1, 1, 33]))
        self._assert_roundtrip(_PlanCase([17, 0, 5]))

    def test_roundtrip_with_prefixes(self):
        self._assert_roundtrip(_PlanCase([9, 17], prefix_lengths=[4, 128]))

    def test_padding_never_selected_and_empty_shards(self):
        # L=1 over CP8: only rank 0's first front row is real; every other
        # row on every rank is padding and must be excluded from the restore.
        case = _PlanCase([1])
        plan = case.plans[0]
        self.assertEqual(int(plan.local_is_real.sum()), 1)
        for rank in range(1, CP):
            self.assertEqual(int(case.plans[rank].local_is_real.sum()), 0)
        gathered = torch.cat([p.local_input_ids for p in case.plans])
        restored = gathered[plan.unpad_restore]
        self.assertTrue(torch.equal(restored, case.full_ids))
        # Padding rows clamp to a valid token (kernels stay in range) but are
        # dropped by the restore permutation.
        for plan in case.plans:
            padded_rows = (~plan.local_is_real).sum()
            if int(padded_rows):
                self.assertTrue(bool((plan.local_positions >= 0).all()))

    def test_local_layout_matches_reference_coordinates(self):
        # The rank-local order is [front block | back block] per request with
        # the reference's mirrored back-half coordinates (cp.py:549-560).
        length = 129
        case = _PlanCase([length])
        padded = ((length + 2 * CP - 1) // (2 * CP)) * 2 * CP
        pair = (padded // CP) // 2
        for rank in range(CP):
            rel = case.plans[rank].relative_positions.tolist()
            expected = [rank * pair + j for j in range(pair)] + [
                padded - (rank + 1) * pair + j for j in range(pair)
            ]
            self.assertEqual(rel, expected)


class ThreadedCollectiveTest(unittest.TestCase):
    def test_all_gather_restore_order(self):
        # Every rank contributes rows tagged with its rank; the restored
        # order must equal the logical (sorted) order.
        full_ids = torch.arange(33, dtype=torch.int64)
        cu = torch.tensor([0, 33], dtype=torch.int64)
        group = ThreadedCPGroup(CP)
        plans = [
            build_zigzag_plan(full_ids, cu, cp_size=CP, cp_rank=rank)
            for rank in range(CP)
        ]
        contexts = [
            CPPrefillContext(plan=plans[rank], group=group.rank_view(rank))
            for rank in range(CP)
        ]
        results = [None] * CP
        errors = []

        def run(rank):
            try:
                # Local rows carry their TOKEN ids, so the restored order is
                # directly comparable with the original sequence.
                local = plans[rank].local_input_ids.to(torch.float32)
                results[rank] = contexts[rank].all_gather_restore(local)
            except Exception as error:  # noqa: BLE001
                errors.append((rank, error))

        threads = [threading.Thread(target=run, args=(rank,)) for rank in range(CP)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(JOIN_TIMEOUT)
            self.assertFalse(thread.is_alive(), "collective rendezvous hung")
        self.assertEqual(errors, [])
        restored = results[0]
        self.assertEqual(tuple(restored.shape), (33,))
        self.assertTrue(torch.equal(restored, full_ids.to(torch.float32)))
        for rank in range(1, CP):
            self.assertTrue(torch.equal(results[rank], restored))

    def test_all_gather_shapes_and_chunk_contract(self):
        full_ids = torch.arange(17, dtype=torch.int64)
        cu = torch.tensor([0, 17], dtype=torch.int64)
        group = ThreadedCPGroup(CP)
        plans = [
            build_zigzag_plan(full_ids, cu, cp_size=CP, cp_rank=rank)
            for rank in range(CP)
        ]
        contexts = [
            CPPrefillContext(plan=plans[rank], group=group.rank_view(rank))
            for rank in range(CP)
        ]
        outputs = [None] * CP
        errors = []

        def run(rank):
            try:
                chunk = plans[rank].chunk_length
                # Row i of the local chunk carries the local index i in every
                # column: the restored rows map back to each token's owner's
                # local row (unpad_restore % chunk).
                local = torch.arange(chunk).unsqueeze(1).expand(chunk, 3).contiguous()
                gathered = contexts[rank].group.all_gather(local)
                outputs[rank] = (
                    gathered.shape,
                    contexts[rank].all_gather_restore(local),
                )
            except Exception as error:  # noqa: BLE001
                errors.append((rank, error))

        threads = [threading.Thread(target=run, args=(rank,)) for rank in range(CP)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(JOIN_TIMEOUT)
            self.assertFalse(thread.is_alive(), "collective rendezvous hung")
        self.assertEqual(errors, [])
        chunk = plans[0].chunk_length
        shape, restored = outputs[0]
        self.assertEqual(tuple(shape), (CP * chunk, 3))
        self.assertEqual(tuple(restored.shape), (17, 3))
        expected_local_row = plans[0].unpad_restore % chunk
        self.assertTrue(
            torch.equal(restored[:, 0], expected_local_row),
            "restored rows must map to each token's rank-local row",
        )

    def test_collective_order_mismatch_detected(self):
        # A rank skipping a collective makes the next rendezvous's op names
        # disagree; every rank must fail loudly (no silent stale results).
        group = ThreadedCPGroup(CP)
        errors = []

        def run(rank):
            view = group.rank_view(rank)
            try:
                if rank == 3:
                    view.all_reduce_sum(torch.zeros(4))
                else:
                    view.all_gather(torch.zeros(4))
            except Exception as error:  # noqa: BLE001
                errors.append((rank, error))

        threads = [threading.Thread(target=run, args=(rank,)) for rank in range(CP)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(JOIN_TIMEOUT)
            self.assertFalse(thread.is_alive(), "mismatch handling hung")
        self.assertEqual(len(errors), CP, f"expected every rank to raise: {errors}")


if __name__ == "__main__":
    unittest.main()
