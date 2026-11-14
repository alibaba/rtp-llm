# SPDX-License-Identifier: Apache-2.0

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from rtp_llm.config.collective_config import CollectiveConfig
from rtp_llm.models_py.distributed import collective_torch as collective


class CollectiveTorchCommOpsUnitTest(unittest.TestCase):
    def _registered_callbacks(self, config=None, process_group=None, stage_group=None):
        if process_group is None:
            process_group = MagicMock()
            process_group.size.return_value = 1
        if stage_group is None:
            stage_group = process_group
        if config is None:
            config = SimpleNamespace(
                tp_size=1,
                dp_size=1,
                pp_size=1,
                world_size=1,
                local_world_size=1,
                tp_rank=0,
            )
        compute_ops = SimpleNamespace(register_comm_ops=MagicMock())

        with (
            patch.dict(sys.modules, {"librtp_compute_ops": compute_ops}),
            patch.object(
                collective,
                "_group_map",
                {
                    collective.Group.WORLD: process_group,
                    collective.Group.STAGE: stage_group,
                },
            ),
            patch.object(collective, "_parallelism_config", config),
            patch.object(collective, "_get_group", return_value=process_group),
            patch.object(torch.distributed, "get_rank", return_value=0),
        ):
            collective._register_process_groups_to_cpp()

        compute_ops.register_comm_ops.assert_called_once()
        return compute_ops.register_comm_ops.call_args.args

    def _registered_allreduce(self):
        return self._registered_callbacks()[1]

    def test_single_rank_allreduce_returns_input_without_dest(self):
        allreduce = self._registered_allreduce()
        tensor = torch.tensor([1.0, 2.0])

        for mode in (
            collective._CPP_PARALLEL_MODE_STAGE,
            collective._CPP_PARALLEL_MODE_WORLD,
        ):
            with self.subTest(mode=mode):
                result = allreduce(tensor, 0, mode, None)
                self.assertIs(result, tensor)

    def test_single_rank_allreduce_copies_into_dest(self):
        allreduce = self._registered_allreduce()
        tensor = torch.tensor([1.0, 2.0])
        for mode in (
            collective._CPP_PARALLEL_MODE_STAGE,
            collective._CPP_PARALLEL_MODE_WORLD,
        ):
            with self.subTest(mode=mode):
                dest = torch.full_like(tensor, -1)
                result = allreduce(tensor, 0, mode, dest)
                self.assertIs(result, dest)
                torch.testing.assert_close(dest, tensor)

    def test_single_rank_allgather_copies_explicit_send_buffer(self):
        allgather = self._registered_callbacks()[2]
        send = torch.tensor([1.0, 2.0])
        for mode in (
            collective._CPP_PARALLEL_MODE_STAGE,
            collective._CPP_PARALLEL_MODE_WORLD,
        ):
            with self.subTest(mode=mode):
                recv = torch.full((1, 2), -1.0)
                allgather([recv], mode, [send], False)
                torch.testing.assert_close(recv, send.reshape_as(recv))

    def test_allreduce_uses_registered_stage_and_world_groups(self):
        world_group = MagicMock()
        world_group.size.return_value = 4
        stage_group = MagicMock()
        stage_group.size.return_value = 2
        config = SimpleNamespace(
            tp_size=1,
            dp_size=2,
            pp_size=2,
            world_size=4,
            local_world_size=4,
            tp_rank=0,
            dp_rank=0,
        )
        allreduce = self._registered_callbacks(config, world_group, stage_group)[1]
        tensor = MagicMock(spec=torch.Tensor)
        tensor.is_cuda = True

        for mode, process_group in (
            (collective._CPP_PARALLEL_MODE_STAGE, stage_group),
            (collective._CPP_PARALLEL_MODE_WORLD, world_group),
        ):
            with (
                self.subTest(mode=mode),
                patch.object(torch.cuda, "current_device", return_value=0),
                patch.object(torch.distributed, "all_reduce") as all_reduce,
            ):
                result = allreduce(tensor, 0, mode, None)
                self.assertIs(result, tensor)
                all_reduce.assert_called_once_with(
                    tensor,
                    op=torch.distributed.ReduceOp.SUM,
                    group=process_group,
                )

    def test_pure_dp_registers_world_group_for_cpp_callback(self):
        process_group = MagicMock()
        process_group.size.return_value = 2
        config = SimpleNamespace(
            tp_size=1,
            dp_size=2,
            pp_size=1,
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
            patch.object(collective, "_get_all_reduce_module") as flashinfer,
            patch.object(collective, "_get_symm_mem") as symm_mem,
            patch.object(collective, "_get_group", return_value=object()),
            patch.object(torch.distributed, "all_reduce", side_effect=reduce_in_place),
        ):
            flashinfer.return_value.get_all_reduce.return_value = None
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

    def test_tp_locality_controls_uds_and_collective_policy(self):
        # Include TP3 and TP2 groups that cross node boundaries inside PP stages.
        # The middle PP stages cross nodes even though TP <= local world size.
        for world_size, pp_size, tp_size, local_size, local_ranks in (
            (12, 4, 3, 4, {0, 1, 2, 9, 10, 11}),
            (6, 3, 2, 3, {0, 1, 4, 5}),
            (8, 1, 8, 4, set()),
            (4, 1, 1, 4, set()),  # DP4 keeps the TP1 shortcut.
        ):
            for rank in range(world_size):
                with self.subTest(world=world_size, pp=pp_size, tp=tp_size, rank=rank):
                    config = SimpleNamespace(
                        world_size=world_size,
                        world_rank=rank,
                        pp_size=pp_size,
                        dp_size=world_size // (pp_size * tp_size),
                        tp_size=tp_size,
                        dp_rank=(rank % (world_size // pp_size)) // tp_size,
                        tp_rank=rank % tp_size,
                        local_world_size=local_size,
                        local_rank=rank % local_size,
                    )
                    compute_ops = SimpleNamespace(
                        register_comm_ops=MagicMock(),
                        init_cpu_tp_broadcaster=MagicMock(),
                    )
                    module = MagicMock()
                    module.get_canonical_p2p.return_value = None
                    policy = CollectiveConfig("nccl")
                    group = object()
                    with (
                        patch.dict(sys.modules, {"librtp_compute_ops": compute_ops}),
                        patch.object(collective, "_parallelism_config", config),
                        patch.object(collective, "_group_map", {}),
                        patch.object(
                            collective, "_cpu_tp_broadcaster_base_path", "/tmp/tp_test"
                        ),
                        patch.object(collective, "_get_group", return_value=group),
                        patch.object(torch.distributed, "get_rank", return_value=rank),
                        patch.object(
                            collective,
                            "_get_all_reduce_module",
                            return_value=module,
                        ),
                    ):
                        collective._register_process_groups_to_cpp()
                        collective._init_all_reduce(config, policy)

                    compute_ops.register_comm_ops.assert_called_once()
                    if rank in local_ranks:
                        compute_ops.init_cpu_tp_broadcaster.assert_called_once_with(
                            config.tp_rank, tp_size, "/tmp/tp_test"
                        )
                    else:
                        compute_ops.init_cpu_tp_broadcaster.assert_not_called()

                    if tp_size > 1:
                        module.init_all_reduce.assert_called_once_with(
                            group,
                            torch.device("cuda", config.local_rank),
                            config,
                            policy,
                        )
                    else:
                        module.init_all_reduce.assert_not_called()


if __name__ == "__main__":
    unittest.main()
