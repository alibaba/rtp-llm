"""EP plan identity and CPU simulation of exact gather/return row ownership."""

import dataclasses
import sys
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models.test.dsv41_cache_codec_test import load_codec_source
from rtp_llm.models.test.dsv41_moe_test import (
    MODULES,
    dispatch,
    moe,
    routed,
    routed_weights,
)

with patch.dict(sys.modules, MODULES):
    collective = load_codec_source("_dsv41_collective_test", "v41_collective.py")


class V41CollectiveTest(unittest.TestCase):
    def plan(self, role="decode"):
        return collective.V41EPPlan(
            dispatch.V41DispatchLayout(128, 2, 8),
            role,
            0 if role == "decode" else 1,
            0 if role == "decode" else 21,
            tuple(range(8)) if role == "decode" else (4, 5, 6, 7),
            "worker-generation-1",
            2,
            intermediate_dim=128,
        )

    def test_fingerprint_binds_numerical_topology_and_schedule_identity(self):
        plan = self.plan()
        self.assertEqual(len(plan.fingerprint), 32)
        self.assertEqual(plan.fingerprint, self.plan().fingerprint)
        for field, value in (
            ("layer_id", 1),
            ("generation", "worker-generation-2"),
            ("rows_per_rank", 4),
            ("swiglu_limit", 0),
            ("intermediate_dim", 256),
            ("world_ranks", tuple(range(8, 16))),
            ("dispatch", dataclasses.replace(plan.dispatch, experts=16)),
        ):
            self.assertNotEqual(
                plan.fingerprint,
                dataclasses.replace(plan, **{field: value}).fingerprint,
            )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            plan.rows_per_rank = 4

    def test_target_roles_reject_invalid_stages_members_and_bounds(self):
        plan = self.plan()
        for field, value in (
            ("role", "tp8"),
            ("pp_rank", 1),
            ("layer_id", 40),
            ("world_ranks", tuple(range(4))),
            ("world_ranks", (0,) * 8),
            ("world_ranks", list(range(8))),
            ("generation", ""),
            ("rows_per_rank", 0),
            ("intermediate_dim", 129),
            ("swiglu_limit", float("nan")),
            ("dispatch", dataclasses.replace(plan.dispatch, experts=9)),
        ):
            with self.assertRaises(ValueError):
                dataclasses.replace(plan, **{field: value})
        with self.assertRaisesRegex(ValueError, "stage"):
            dataclasses.replace(self.plan("prefill"), layer_id=20)

    def test_unadmitted_or_rebound_collectives_fail_before_transport(self):
        local = moe.V41LocalMoE(
            routed.V41RoutedExperts(
                routed_weights(experts=1, dim=128, inter=128), backend="eager"
            )
        )
        op = collective.V41CollectiveMoE(local, self.plan(), object())
        with patch.object(torch.distributed, "all_gather_into_tensor") as gather:
            with self.assertRaisesRegex(RuntimeError, "admission"):
                op(torch.empty(0, 128), torch.empty(0, 2), torch.empty(0, 2))
            gather.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            op.admit()
        op._admitted = True
        op._identity = op._runtime_identity()
        local.expert_start_id += 1
        with self.assertRaisesRegex(RuntimeError, "admission"):
            op(torch.empty(0, 128), torch.empty(0, 2), torch.empty(0, 2))

    def test_both_transports_preserve_rank_rows_and_fp32_partial_returns(self):
        for role in ("prefill", "decode"):
            plan = self.plan(role)
            world, capacity = len(plan.world_ranks), plan.rows_per_rank
            tensors = routed_weights(experts=8, dim=128, inter=128)
            x = torch.randn(
                world * capacity, 128, generator=torch.Generator().manual_seed(411)
            ).bfloat16()
            weights = torch.tensor([[0.37, 0.7]]).repeat(world * capacity, 1)
            ids = torch.stack(
                (
                    torch.arange(world * capacity) % 8,
                    (torch.arange(world * capacity) + 3) % 8,
                ),
                -1,
            ).int()
            valid = torch.arange(world * capacity) % 3 != 1
            x[~valid] = 0
            weights[~valid] = 0
            ids[~valid] = -1
            locals_, partials = [], []
            per_rank = 8 // world
            for rank in range(world):
                local = moe.V41LocalMoE(
                    routed.V41RoutedExperts(
                        {
                            k: v[rank * per_rank : (rank + 1) * per_rank]
                            for k, v in tensors.items()
                        },
                        backend="eager",
                    ),
                    expert_start_id=rank * per_rank,
                    max_tokens=32,
                    masked_token_limit=32,
                )
                locals_.append(local)
                partials.append(
                    local(
                        x,
                        weights,
                        ids,
                        layout="masked" if role == "decode" else "contiguous",
                    )
                )
            expected = torch.zeros_like(partials[0])
            for part in partials:
                expected.add_(part)
            wire = dispatch.pack_dispatch(
                x, weights, ids, valid, plan.dispatch, backend="eager"
            )
            for rank, local in enumerate(locals_):
                begin, end = rank * capacity, (rank + 1) * capacity
                op = collective.V41CollectiveMoE(local, plan, object())
                gathers = []

                def gather(out, value, *, group):
                    self.assertIs(group, op.process_group)
                    source = {
                        torch.uint8: wire,
                        torch.bfloat16: x,
                        torch.float32: weights,
                        torch.int32: ids,
                    }[value.dtype]
                    torch.testing.assert_close(value, source[begin:end], rtol=0, atol=0)
                    out.copy_(source)
                    gathers.append(value.dtype)

                def exchange(out, value, *, group):
                    self.assertIs(group, op.process_group)
                    self.assertEqual(value.dtype, torch.float32)
                    torch.testing.assert_close(value, partials[rank], rtol=0, atol=0)
                    out.copy_(torch.cat([part[begin:end] for part in partials]))

                def pack(*args, **kwargs):
                    return dispatch.pack_dispatch(*args, **kwargs, backend="eager")

                with patch.dict(sys.modules, MODULES), patch.object(
                    torch.distributed, "all_gather_into_tensor", side_effect=gather
                ), patch.object(
                    torch.distributed, "all_to_all_single", side_effect=exchange
                ), patch.object(
                    collective, "pack_dispatch", side_effect=pack
                ):
                    actual = op._forward(
                        x[begin:end],
                        weights[begin:end],
                        ids[begin:end],
                        valid[begin:end],
                    )
                torch.testing.assert_close(actual, expected[begin:end], rtol=0, atol=0)
                self.assertEqual(
                    gathers,
                    (
                        [torch.uint8]
                        if role == "decode"
                        else [torch.bfloat16, torch.float32, torch.int32]
                    ),
                )


if __name__ == "__main__":
    unittest.main()
