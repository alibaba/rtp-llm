# SPDX-License-Identifier: Apache-2.0

import sys
import tempfile
import unittest
from contextlib import contextmanager
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from rtp_llm.models_py.distributed import collective_torch as collective


def _check_control_group_startup_rank(rank, rendezvous):
    dist = torch.distributed
    dist.init_process_group(
        "gloo",
        init_method="file://" + rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=15),
    )
    checks = unittest.TestCase()
    config = SimpleNamespace(
        world_rank=rank,
        world_size=2,
        local_rank=rank,
        tp_size=1,
        dp_size=2,
        ep_size=2,
    )
    try:
        for capabilities, enabled in (
            ([(12, 0), (12, 0)], True),
            ([(12, 0), (9, 0)], None),
            ([(9, 0), (12, 0)], None),
            ([(12, 0), (12, 1)], None),
            ([(9, 0), (10, 0)], False),
        ):
            with (
                patch.object(collective, "is_cuda", return_value=True),
                patch.object(collective, "get_sm", return_value=capabilities[rank]),
                patch.object(collective, "_group_map", {}),
                patch.object(dist, "new_group", wraps=dist.new_group) as new_group,
            ):
                if enabled is None:
                    with checks.assertRaisesRegex(RuntimeError, "all world ranks"):
                        collective._create_process_groups(
                            config, "gloo", timedelta(seconds=15)
                        )
                    new_group.assert_not_called()
                else:
                    collective._create_process_groups(
                        config, "gloo", timedelta(seconds=15)
                    )
                    checks.assertEqual(
                        collective.Group.DP_CONTROL in collective._group_map, enabled
                    )
                    if enabled:
                        group = collective._group_map[collective.Group.DP_CONTROL]
                        vote = torch.tensor([rank], dtype=torch.int32)
                        dist.all_reduce(vote, group=group)
                        checks.assertEqual(vote.item(), 1)
                        dist.destroy_process_group(group)
                    else:
                        new_group.assert_not_called()
                dist.barrier()
    finally:
        dist.destroy_process_group()


class CollectiveTorchCommOpsUnitTest(unittest.TestCase):
    def test_live_world_checks_architectures_before_subgroups(self):
        with tempfile.TemporaryDirectory() as directory:
            torch.multiprocessing.spawn(
                _check_control_group_startup_rank,
                args=(directory + "/world",),
                nprocs=2,
                join=True,
            )

    def _registered_callbacks(
        self,
        config=None,
        process_group=None,
        control_group=None,
        dp_group=None,
        capability=(12, 0),
    ):
        if process_group is None:
            process_group = MagicMock()
            process_group.size.return_value = 1
        if config is None:
            config = SimpleNamespace(
                tp_size=1,
                dp_size=1,
                world_size=1,
                local_world_size=1,
                tp_rank=0,
            )
        compute_ops = SimpleNamespace(register_comm_ops=MagicMock())
        groups = {collective.Group.DP_AND_TP: process_group}
        if control_group is not None:
            groups[collective.Group.DP_CONTROL] = control_group
        if dp_group is not None:
            groups[collective.Group.DP.name + "0"] = dp_group

        with (
            patch.dict(sys.modules, {"librtp_compute_ops": compute_ops}),
            patch.object(
                collective,
                "_group_map",
                groups,
            ),
            patch.object(collective, "_parallelism_config", config),
            patch.object(collective, "is_cuda", return_value=capability is not None),
            patch.object(collective, "get_sm", return_value=capability),
        ):
            collective._register_process_groups_to_cpp()

        compute_ops.register_comm_ops.assert_called_once()
        return compute_ops.register_comm_ops.call_args.args

    def _registered_allreduce(self):
        return self._registered_callbacks()[1]

    def test_cpu_dp_vote_never_accesses_cuda(self):
        for tp_size in (1, 2):
            with self.subTest(tp_size=tp_size):
                self._check_cpu_dp_vote(tp_size)

    def _check_cpu_dp_vote(self, tp_size):
        group = MagicMock()
        group.size.return_value = 2
        config = SimpleNamespace(
            tp_size=tp_size,
            dp_size=4 // tp_size,
            ep_size=4,
            world_size=4,
            local_world_size=4,
            local_rank=0,
        )
        allreduce = self._registered_callbacks(config, control_group=group)[1]
        for use_dest in (False, True):
            with self.subTest(use_dest=use_dest):
                value = torch.tensor([0], dtype=torch.int32)
                dest = torch.empty_like(value) if use_dest else None

                def vote(target, **kwargs):
                    self.assertEqual(target.device.type, "cpu")
                    self.assertIs(kwargs["group"], group)
                    self.assertEqual(kwargs["op"], torch.distributed.ReduceOp.MAX)
                    target.fill_(1)

                with (
                    patch.object(
                        torch.cuda,
                        "current_device",
                        side_effect=AssertionError("CUDA access"),
                    ),
                    patch.object(
                        torch.distributed, "all_reduce", side_effect=vote
                    ) as reduce,
                ):
                    result = allreduce(value, 2, collective._CPP_PARALLEL_MODE_DP, dest)
                self.assertIs(result, dest if use_dest else value)
                self.assertEqual(result.item(), 1)
                reduce.assert_called_once()
                if use_dest:
                    self.assertEqual(value.item(), 0)

    def test_cpu_dp_vote_rejects_missing_leader_group(self):
        for tp_size in (1, 2):
            with self.subTest(tp_size=tp_size):
                config = SimpleNamespace(
                    tp_size=tp_size,
                    dp_size=4 // tp_size,
                    ep_size=4,
                    world_size=4,
                    local_world_size=4,
                    local_rank=0,
                )
                allreduce = self._registered_callbacks(config)[1]
                with self.assertRaisesRegex(RuntimeError, "TP leader group"):
                    allreduce(
                        torch.tensor([0]), 2, collective._CPP_PARALLEL_MODE_DP, None
                    )

    def test_cpu_dp_without_ep_retains_nccl_promotion(self):
        self._check_nccl_promotion(ep_size=1, capability=(12, 0))

    def test_other_devices_cpu_dp_with_ep_retains_nccl_promotion(self):
        for capability in ((9, 0), (10, 0), (12, 1)):
            with self.subTest(capability=capability):
                self._check_nccl_promotion(ep_size=4, capability=capability)

    def _check_nccl_promotion(self, ep_size, capability):
        group = MagicMock()
        group.size.return_value = 2
        config = SimpleNamespace(
            tp_size=2,
            dp_size=2,
            ep_size=ep_size,
            world_size=4,
            local_world_size=4,
            local_rank=0,
        )
        with patch.object(torch.distributed, "get_rank", return_value=0):
            allreduce = self._registered_callbacks(
                config, dp_group=group, capability=capability
            )[1]
        value = torch.tensor([0], dtype=torch.int32)
        promoted = torch.tensor([0], dtype=torch.int32)

        def vote(target, **kwargs):
            self.assertIs(target, promoted)
            self.assertIs(kwargs["group"], group)
            target.fill_(1)

        with (
            patch.object(torch.cuda, "current_device", return_value=0),
            patch.object(torch.Tensor, "to", return_value=promoted) as promote,
            patch.object(torch.distributed, "all_reduce", side_effect=vote) as reduce,
        ):
            result = allreduce(value, 2, collective._CPP_PARALLEL_MODE_DP, None)
        promote.assert_called_once_with(torch.device("cuda", 0))
        reduce.assert_called_once()
        self.assertIs(result, value)
        self.assertEqual(value.item(), 1)

    def test_control_group_is_created_only_for_sm120_dp_expert_parallelism(self):
        for capability in (None, (9, 0), (10, 0), (12, 0), (12, 1)):
            with self.subTest(capability=capability):
                self._check_control_group_creation(capability)

    @contextmanager
    def _capture_group_creation(self, config, capabilities, world_backend="nccl"):
        capability = capabilities[config.world_rank]
        tensor = torch.tensor
        world = MagicMock()

        def capability_tensor(values, **kwargs):
            expected_device = (
                torch.device("cuda", config.local_rank)
                if world_backend == "nccl"
                else torch.device("cpu")
            )
            self.assertEqual(kwargs.pop("device"), expected_device)
            return tensor(values, **kwargs)

        def gather(output, local, *, group):
            self.assertIs(group, world)
            self.assertEqual(local.tolist(), list(capability or (-1, -1)))
            self.assertEqual(local.dtype, torch.int32)
            self.assertEqual(len(output), len(capabilities))
            new_group.assert_not_called()
            for target, value in zip(output, capabilities):
                target.copy_(tensor(value or (-1, -1), dtype=local.dtype))

        with (
            patch.object(collective, "_group_map", {}),
            patch.object(torch.distributed, "group", SimpleNamespace(WORLD=world)),
            patch.object(torch.distributed, "get_backend", return_value=world_backend),
            patch.object(
                torch.distributed, "all_gather", side_effect=gather
            ) as all_gather,
            patch.object(torch, "tensor", side_effect=capability_tensor),
            patch.object(
                torch.distributed, "new_group", return_value=MagicMock()
            ) as new_group,
            patch.object(torch.distributed, "barrier"),
            patch.object(collective, "_get_symm_mem", return_value=MagicMock()),
            patch.object(collective, "is_cuda", return_value=capability is not None),
            patch.object(collective, "get_sm", return_value=capability) as get_sm,
        ):
            yield SimpleNamespace(
                new_group=new_group, all_gather=all_gather, get_sm=get_sm
            )

    def _check_control_group_creation(self, capability):
        for tp_size, dp_size, ep_size in (
            (2, 2, 1),
            (2, 2, 2),
            (2, 2, 4),
            (1, 4, 4),
            (1, 4, 1),
            (4, 1, 4),
            (1, 1, 1),
        ):
            for rank in range(tp_size * dp_size):
                config = SimpleNamespace(
                    world_rank=rank,
                    world_size=tp_size * dp_size,
                    tp_size=tp_size,
                    dp_size=dp_size,
                    ep_size=ep_size,
                    local_rank=rank,
                )
                with (
                    self.subTest(tp=tp_size, dp=dp_size, ep=ep_size, rank=rank),
                    self._capture_group_creation(
                        config, [capability] * config.world_size
                    ) as calls,
                ):
                    collective._create_process_groups(config, "nccl", None)
                    groups = [call.kwargs for call in calls.new_group.call_args_list]
                    control_groups = [g for g in groups if g["backend"] == "gloo"]
                    dp_ep = capability == (12, 0) and dp_size > 1 and ep_size > 1
                    if capability is None:
                        calls.get_sm.assert_not_called()
                    elif dp_size > 1 and ep_size > 1:
                        calls.get_sm.assert_called_once_with(rank)
                    if dp_size > 1 and ep_size > 1:
                        calls.all_gather.assert_called_once()
                    else:
                        calls.all_gather.assert_not_called()
                    if dp_ep:
                        self.assertEqual(
                            control_groups,
                            [
                                {
                                    "ranks": list(range(0, tp_size * dp_size, tp_size)),
                                    "backend": "gloo",
                                    "timeout": None,
                                }
                            ],
                        )
                        self.assertEqual(groups[0], control_groups[0])
                    else:
                        self.assertEqual(control_groups, [])
                    self.assertEqual(
                        collective.Group.DP_CONTROL in collective._group_map,
                        dp_ep and rank % tp_size == 0,
                    )
                    if tp_size == 2 and dp_size == 2:
                        self.assertEqual(
                            [g["ranks"] for g in groups if g["backend"] == "nccl"],
                            [[0, 2], [1, 3], [0, 1], [2, 3]],
                        )

    def test_mixed_sm120_ranks_fail_before_any_subgroup(self):
        for capabilities in (
            [(12, 0), (9, 0), (12, 0), (9, 0)],
            [(9, 0), (12, 0), (9, 0), (12, 0)],
            [(12, 0), (12, 1), (12, 0), (12, 1)],
            [(12, 0), None, (12, 0), None],
        ):
            for tp_size in (1, 2):
                for rank in range(4):
                    config = SimpleNamespace(
                        world_rank=rank,
                        world_size=4,
                        local_rank=rank,
                        tp_size=tp_size,
                        dp_size=4 // tp_size,
                        ep_size=4,
                    )
                    with (
                        self.subTest(capabilities=capabilities, tp=tp_size, rank=rank),
                        self._capture_group_creation(config, capabilities) as calls,
                    ):
                        with self.assertRaisesRegex(
                            RuntimeError, "all world ranks"
                        ) as error:
                            collective._create_process_groups(config, "nccl", None)
                        self.assertIn(
                            str([list(value or (-1, -1)) for value in capabilities]),
                            str(error.exception),
                        )
                        calls.all_gather.assert_called_once()
                        calls.new_group.assert_not_called()
                        self.assertEqual(collective._group_map, {})

    def test_mixed_non_sm120_ranks_keep_existing_groups(self):
        capabilities = [(9, 0), (10, 0), (12, 1), (9, 0)]
        for rank in range(4):
            config = SimpleNamespace(
                world_rank=rank,
                world_size=4,
                local_rank=rank,
                tp_size=2,
                dp_size=2,
                ep_size=4,
            )
            with (
                self.subTest(rank=rank),
                self._capture_group_creation(config, capabilities) as calls,
            ):
                collective._create_process_groups(config, "nccl", None)
                calls.all_gather.assert_called_once()
                groups = [call.kwargs for call in calls.new_group.call_args_list]
                self.assertEqual([g["backend"] for g in groups], ["nccl"] * 4)
                self.assertEqual(
                    [g["ranks"] for g in groups], [[0, 2], [1, 3], [0, 1], [2, 3]]
                )

    def test_capability_exchange_uses_existing_world_backend(self):
        config = SimpleNamespace(
            world_rank=0,
            world_size=2,
            local_rank=0,
            tp_size=1,
            dp_size=2,
            ep_size=2,
        )
        with self._capture_group_creation(config, [(12, 0)] * 2, "gloo") as calls:
            collective._create_process_groups(config, "nccl", None)
            calls.all_gather.assert_called_once()
            calls.new_group.assert_called_once_with(
                ranks=[0, 1], backend="gloo", timeout=None
            )

    def test_single_rank_allreduce_returns_input_without_dest(self):
        allreduce = self._registered_allreduce()
        tensor = torch.tensor([1.0, 2.0])

        result = allreduce(
            tensor,
            0,
            collective._CPP_PARALLEL_MODE_DP_AND_TP,
            None,
        )

        self.assertIs(result, tensor)

    def test_single_rank_allreduce_copies_into_dest(self):
        allreduce = self._registered_allreduce()
        tensor = torch.tensor([1.0, 2.0])
        dest = torch.full_like(tensor, -1)

        result = allreduce(
            tensor,
            0,
            collective._CPP_PARALLEL_MODE_DP_AND_TP,
            dest,
        )

        self.assertIs(result, dest)
        torch.testing.assert_close(dest, tensor)

    def test_single_rank_allgather_copies_explicit_send_buffer(self):
        allgather = self._registered_callbacks()[2]
        send = torch.tensor([1.0, 2.0])
        recv = torch.full((1, 2), -1.0)

        allgather(
            [recv],
            collective._CPP_PARALLEL_MODE_DP_AND_TP,
            [send],
            False,
        )

        torch.testing.assert_close(recv, send.reshape_as(recv))

    def test_pure_dp_registers_world_group_for_cpp_callback(self):
        process_group = MagicMock()
        process_group.size.return_value = 2
        config = SimpleNamespace(
            tp_size=1,
            dp_size=2,
            ep_size=1,
            world_size=2,
            local_world_size=2,
            tp_rank=0,
        )

        broadcast = self._registered_callbacks(config, process_group)[0]
        with (
            patch.object(
                torch.distributed, "get_global_rank", return_value=0
            ) as get_global_rank,
            patch.object(torch.cuda, "current_device", return_value=0),
        ):
            broadcast([], 0, collective._CPP_PARALLEL_MODE_DP)

        get_global_rank.assert_called_once_with(process_group, 0)

    def test_regular_allreduce_supports_explicit_out_of_place(self):
        tensor = torch.tensor([1.0, 2.0])

        def reduce_in_place(target, **_kwargs):
            target.add_(10)

        with (
            patch.object(collective, "_get_rocm_rccl", return_value=None),
            patch.object(collective, "_get_flashinfer_allreduce") as flashinfer,
            patch.object(collective, "_get_symm_mem") as symm_mem,
            patch.object(collective, "_get_group", return_value=object()),
            patch.object(torch.distributed, "all_reduce", side_effect=reduce_in_place),
        ):
            flashinfer.return_value.get_flashinfer_allreduce.return_value = None
            symm_mem.return_value.get_symm_mem_communicator.return_value = None
            result = collective.all_reduce(tensor, collective.Group.TP, inplace=False)

        self.assertIsNot(result, tensor)
        torch.testing.assert_close(tensor, torch.tensor([1.0, 2.0]))
        torch.testing.assert_close(result, torch.tensor([11.0, 12.0]))

    def test_capture_allreduce_supports_explicit_out_of_place(self):
        tensor = torch.tensor([1.0, 2.0])
        capture = MagicMock()
        capture.ensure_capture_comm_ready.return_value = None
        capture.should_use_capture_collectives.return_value = True
        capture.capture_all_reduce.side_effect = lambda target, _group: target.add_(10)

        with (
            patch.object(collective, "_get_rocm_rccl", return_value=capture),
            patch.object(collective, "_get_group", return_value=object()),
        ):
            result = collective.all_reduce(tensor, collective.Group.TP, inplace=False)

        self.assertIsNot(result, tensor)
        torch.testing.assert_close(tensor, torch.tensor([1.0, 2.0]))
        torch.testing.assert_close(result, torch.tensor([11.0, 12.0]))


if __name__ == "__main__":
    unittest.main()
