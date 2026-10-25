"""CPU contract tests for batched compact-CP receiver validation.

Covers the fused unpack path: admission-time stacked expectations must accept
byte-identical packets and reject the same corruptions as the legacy
per-sender chain, with the tail-owner exemption preserved.
"""

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]

pkg = types.ModuleType("v41_compact_fused_test_pkg")
pkg.__path__ = [str(ROOT)]
sys.modules[pkg.__name__] = pkg


def load(name):
    spec = importlib.util.spec_from_file_location(
        f"{pkg.__name__}.{name}", ROOT / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


plan_mod = load("_v41_compact_cp_plan")
transport = load("_v41_compact_cp_transport")

CompactCPPlan = plan_mod.CompactCPPlan
HEADER_BYTES = plan_mod.HEADER_BYTES
RECORD_BYTES = plan_mod.RECORD_BYTES
TAIL_RECORD_BYTES = plan_mod.TAIL_RECORD_BYTES


def make_plans(layer, ratio, start=4096):
    return tuple(
        CompactCPPlan("packed", -9, 2, 7, 3, 11, start, rank, layer, ratio)
        for rank in range(4)
    )


def make_operands(plan):
    rows = plan.rows
    g = torch.Generator().manual_seed(1000 + plan.rank)
    main = torch.randint(0, 256, (rows, 288), dtype=torch.uint8, generator=g)
    index68 = torch.randint(0, 256, (rows, 68), dtype=torch.uint8, generator=g)
    tail = (
        torch.arange(2048, dtype=torch.float32).reshape(2, 1024) / 32
        if plan.tail_count
        else None
    )
    return main, index68, tail


def make_received(plans, metadata):
    packets = []
    for plan, meta in zip(plans, metadata):
        main, index68, tail = make_operands(plan)
        packet = transport.pack(
            plan,
            meta,
            main,
            index68[:, :64].contiguous().view(torch.int8),
            index68[:, 64:].contiguous().view(torch.int32).flatten(),
            tail,
        )
        packets.append(packet)
    return torch.stack(packets)


def expected_operands(plans):
    mains, index68s = [], []
    for plan in plans:
        main, index68, _ = make_operands(plan)
        mains.append(main)
        index68s.append(index68)
    return torch.cat(mains), torch.cat(index68s)


class FusedValidationTest(unittest.TestCase):
    def check_roundtrip(self, layer, ratio):
        plans = make_plans(layer, ratio)
        metadata = transport.prepare_metadata_batch(plans, torch.device("cpu"))
        (stack,) = transport.prepare_validation_stacks((plans,), torch.device("cpu"))
        self.assertTrue(
            torch.equal(
                stack.header,
                torch.stack([torch.tensor(p.header, dtype=torch.int64) for p in plans]),
            )
        )
        self.assertEqual(stack.positions.shape, (4, plans[0].rows))
        received = make_received(plans, metadata)
        main, index, state = transport.unpack(plans, metadata, received, stack)
        expected_main, expected_index = expected_operands(plans)
        self.assertTrue(torch.equal(main, expected_main))
        self.assertTrue(torch.equal(index, expected_index))
        if ratio == 2:
            self.assertIsNotNone(state)
            self.assertTrue(torch.equal(state, make_operands(plans[0])[2]))
        else:
            self.assertIsNone(state)
        # Legacy compatibility path (no precomputed stack) returns identical bytes.
        legacy = transport.unpack(plans, metadata, received)
        for a, b in zip((main, index, state), legacy):
            if a is None:
                self.assertIsNone(b)
            else:
                self.assertTrue(torch.equal(a, b))

    def test_ratio2_roundtrip(self):
        self.check_roundtrip(2, 2)

    def test_ratio1_roundtrip(self):
        self.check_roundtrip(20, 1)

    def check_rejected(self, layer, ratio, corrupt):
        plans = make_plans(layer, ratio)
        metadata = transport.prepare_metadata_batch(plans, torch.device("cpu"))
        (stack,) = transport.prepare_validation_stacks((plans,), torch.device("cpu"))
        received = make_received(plans, metadata)
        corrupt(received)
        with self.assertRaises(ValueError) as ctx:
            transport.unpack(plans, metadata, received, stack)
        self.assertIn("Invalid compact CP identity or row order", str(ctx.exception))

    def test_corrupt_header_rejected(self):
        self.check_rejected(2, 2, lambda r: received_flip(r, 2, 5))

    def test_corrupt_positions_rejected(self):
        offset = HEADER_BYTES + 3 * RECORD_BYTES
        self.check_rejected(2, 2, lambda r: received_flip(r, 2, offset))
        self.check_rejected(20, 1, lambda r: received_flip(r, 1, offset))

    def test_corrupt_tail_identity_rejected(self):
        end = HEADER_BYTES + 512 * RECORD_BYTES
        self.check_rejected(2, 2, lambda r: received_flip(r, 3, end + 3))

    def test_non_owner_state_nonzero_rejected(self):
        end = HEADER_BYTES + 512 * RECORD_BYTES
        self.check_rejected(2, 2, lambda r: received_flip(r, 1, end + 100))
        self.check_rejected(2, 2, lambda r: received_flip(r, 3, end + 8000))

    def test_owner_state_nonzero_accepted(self):
        # Rank 0 owns the CSA tail; its state rows are real (nonzero) data.
        plans = make_plans(2, 2)
        metadata = transport.prepare_metadata_batch(plans, torch.device("cpu"))
        (stack,) = transport.prepare_validation_stacks((plans,), torch.device("cpu"))
        received = make_received(plans, metadata)
        self.assertTrue(bool((received[0, -4096:] != 0).any()))
        transport.unpack(plans, metadata, received, stack)

    def test_mismatched_stack_rejected(self):
        plans = make_plans(2, 2)
        metadata = transport.prepare_metadata_batch(plans, torch.device("cpu"))
        (stack,) = transport.prepare_validation_stacks(
            (make_plans(2, 2, start=8192),), torch.device("cpu")
        )
        received = make_received(plans, metadata)
        with self.assertRaises(ValueError):
            transport.unpack(plans, metadata, received, stack)

    def test_stack_shape_contract_rejected(self):
        plans = make_plans(2, 2)
        metadata = transport.prepare_metadata_batch(plans, torch.device("cpu"))
        (stack,) = transport.prepare_validation_stacks((plans,), torch.device("cpu"))
        received = make_received(plans, metadata)
        bad = transport.ValidationStack(
            stack.header[:2],
            stack.positions,
            stack.tail_identity,
            stack.tail_state_exempt,
        )
        with self.assertRaises(ValueError) as ctx:
            transport.unpack(plans, metadata, received, bad)
        self.assertIn("validation stack", str(ctx.exception))

    def test_prepare_validation_stacks_requires_four_plans(self):
        with self.assertRaises(ValueError):
            transport.prepare_validation_stacks(
                (make_plans(2, 2)[:3],), torch.device("cpu")
            )

    def test_all_producer_groups_one_storage(self):
        groups = tuple(make_plans(layer, ratio) for layer, ratio in plan_mod.PRODUCERS)
        stacks = transport.prepare_validation_stacks(groups, torch.device("cpu"))
        self.assertEqual(len(stacks), 4)
        for plans, stack in zip(groups, stacks):
            rows = plans[0].rows
            self.assertEqual(stack.positions.shape, (4, rows))
            for rank, plan in enumerate(plans):
                self.assertTrue(
                    torch.equal(
                        stack.header[rank],
                        torch.tensor(plan.header, dtype=torch.int64),
                    )
                )
                self.assertTrue(
                    torch.equal(
                        stack.positions[rank],
                        torch.tensor(plan.boundary_positions, dtype=torch.int64),
                    )
                )
            if plans[0].ratio == 2:
                self.assertEqual(
                    stack.tail_state_exempt.tolist(), [True, False, False, False]
                )
            else:
                self.assertIsNone(stack.tail_identity)
                self.assertIsNone(stack.tail_state_exempt)


def received_flip(received, rank, offset):
    received[rank, offset] ^= 1
    return received


if __name__ == "__main__":
    unittest.main()
