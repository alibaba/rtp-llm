# SPDX-License-Identifier: Apache-2.0

import sys
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from rtp_llm.models_py.distributed import collective_torch as collective


def _cpu_tp_broadcast_worker(rank, world_size, rendezvous_dir):
    torch.set_num_threads(1)
    for cycle, tp_size in enumerate((world_size, world_size // 2)):
        torch.distributed.init_process_group(
            "gloo",
            init_method=f"file://{rendezvous_dir}/init_{cycle}",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=30),
        )
        config = SimpleNamespace(
            tp_size=tp_size,
            dp_size=world_size // tp_size,
            world_size=world_size,
            world_rank=rank,
            local_world_size=1,
            tp_rank=rank % tp_size,
        )
        compute_ops = SimpleNamespace(
            register_comm_ops=MagicMock(), clear_comm_ops=MagicMock()
        )
        try:
            collective._group_map = {
                collective.Group.DP_AND_TP: torch.distributed.group.WORLD
            }
            collective._parallelism_config = config
            collective._initialized = True
            with patch.object(collective, "_get_symm_mem"):
                collective._create_process_groups(config, "gloo", timedelta(seconds=30))
            with patch.dict(sys.modules, {"librtp_compute_ops": compute_ops}):
                collective._register_process_groups_to_cpp()
            broadcast = compute_ops.register_comm_ops.call_args.args[3]
            assert broadcast is not None
            root = (rank // tp_size) * tp_size
            with (
                patch.object(
                    torch.cuda,
                    "current_device",
                    side_effect=AssertionError("CUDA access"),
                ),
                patch.object(
                    torch.cuda, "synchronize", side_effect=AssertionError("CUDA sync")
                ),
            ):
                for step, size in enumerate((4, 1, 0, 9)):
                    value = root * 10 + step + 1
                    local_value = value if rank == root else -1
                    hints = torch.full((size,), local_value, dtype=torch.int64)
                    packed = torch.full(
                        (size * 3,), local_value % 256, dtype=torch.uint8
                    )
                    noncontiguous = torch.full(
                        (size, 3), complex(local_value, 2), dtype=torch.complex64
                    ).transpose(0, 1)
                    scalar = torch.tensor(local_value, dtype=torch.bfloat16)
                    broadcast(
                        [hints, packed, noncontiguous, scalar],
                        0,
                        collective._CPP_PARALLEL_MODE_TP,
                    )
                    torch.testing.assert_close(hints, torch.full_like(hints, value))
                    torch.testing.assert_close(
                        packed, torch.full_like(packed, value % 256)
                    )
                    torch.testing.assert_close(
                        noncontiguous,
                        torch.full_like(noncontiguous, complex(value, 2)),
                    )
                    torch.testing.assert_close(scalar, torch.full_like(scalar, value))
        finally:
            with (
                patch.dict(
                    sys.modules,
                    {
                        "librtp_compute_ops": compute_ops,
                        "rtp_llm.models_py.utils.arch": SimpleNamespace(
                            is_cuda=lambda: False
                        ),
                        "rtp_llm.models_py.distributed.tp_collective_workspace": SimpleNamespace(
                            clear_tp_collective_workspaces=lambda: None
                        ),
                    },
                ),
                patch.object(collective, "_get_rocm_rccl", return_value=None),
            ):
                collective.destroy_distributed_environment()
            compute_ops.clear_comm_ops.assert_called_once()
            assert collective._cpu_tp_group is None


class CollectiveTorchCommOpsUnitTest(unittest.TestCase):
    def _registered_callbacks(self, config=None, process_group=None, cpu_group=None):
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

        with (
            patch.dict(sys.modules, {"librtp_compute_ops": compute_ops}),
            patch.object(
                collective,
                "_group_map",
                {collective.Group.DP_AND_TP: process_group},
            ),
            patch.object(collective, "_parallelism_config", config),
            patch.object(collective, "_cpu_tp_group", cpu_group),
        ):
            collective._register_process_groups_to_cpp()

        compute_ops.register_comm_ops.assert_called_once()
        return compute_ops.register_comm_ops.call_args.args

    def test_intra_node_does_not_create_gloo_group(self):
        config = SimpleNamespace(tp_size=4, local_world_size=4)
        with (
            patch.object(collective, "_cpu_tp_group", None),
            patch.object(torch.distributed, "new_group") as new_group,
            patch.object(torch.distributed, "is_gloo_available", return_value=False),
        ):
            collective._create_cpu_tp_group(config, timedelta(seconds=30))
            self.assertIsNone(collective._cpu_tp_group)
            new_group.assert_not_called()

    def test_single_rank_does_not_require_gloo_with_unknown_local_size(self):
        config = SimpleNamespace(tp_size=1, local_world_size=0)
        with (
            patch.object(collective, "_cpu_tp_group", None),
            patch.object(torch.distributed, "new_group") as new_group,
            patch.object(torch.distributed, "is_gloo_available", return_value=False),
        ):
            collective._create_cpu_tp_group(config, timedelta(seconds=30))
            self.assertIsNone(collective._cpu_tp_group)
            new_group.assert_not_called()

    def test_cross_node_requires_gloo_support(self):
        config = SimpleNamespace(tp_size=4, local_world_size=2)
        with (
            patch.object(collective, "_cpu_tp_group", None),
            patch.object(torch.distributed, "is_gloo_available", return_value=False),
        ):
            with self.assertRaisesRegex(RuntimeError, "requires PyTorch Gloo"):
                collective._create_cpu_tp_group(config, timedelta(seconds=30))

    def test_cpu_broadcast_rejects_wrong_mode_and_cuda_input(self):
        callback = self._registered_callbacks(cpu_group=MagicMock())[3]
        with self.assertRaisesRegex(ValueError, "CPU TP process group"):
            callback([], 0, collective._CPP_PARALLEL_MODE_DP)
        tensor = SimpleNamespace(device=torch.device("cuda"))
        with patch.object(torch.distributed, "get_global_rank", return_value=0):
            with self.assertRaisesRegex(ValueError, "requires CPU tensors"):
                callback([tensor], 0, collective._CPP_PARALLEL_MODE_TP)

    def test_cpu_group_is_not_published_after_partial_initialization_failure(self):
        config = SimpleNamespace(tp_size=2, local_world_size=1, dp_size=2, world_rank=0)
        with (
            patch.object(collective, "_cpu_tp_group", None),
            patch.object(torch.distributed, "is_gloo_available", return_value=True),
            patch.object(
                torch.distributed,
                "new_group",
                side_effect=[object(), RuntimeError("group creation failed")],
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "group creation failed"):
                collective._create_cpu_tp_group(config, timedelta(seconds=30))
            self.assertIsNone(collective._cpu_tp_group)

    @unittest.skipUnless(torch.distributed.is_gloo_available(), "Gloo unavailable")
    def test_cpu_broadcast_multiple_tp_groups_and_reinitialization(self):
        with tempfile.TemporaryDirectory() as rendezvous_dir:
            torch.multiprocessing.spawn(
                _cpu_tp_broadcast_worker,
                args=(4, str(Path(rendezvous_dir).resolve())),
                nprocs=4,
                join=True,
            )

    def _registered_allreduce(self):
        return self._registered_callbacks()[1]

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
            world_size=2,
            local_world_size=2,
            tp_rank=0,
        )

        broadcast = self._registered_callbacks(config, process_group)[0]
        with patch.object(
            torch.distributed, "get_global_rank", return_value=0
        ) as get_global_rank, patch.object(torch.cuda, "current_device", return_value=0):
            broadcast([], 0, collective._CPP_PARALLEL_MODE_DP)

        get_global_rank.assert_called_once_with(process_group, 0)

    def test_regular_allreduce_supports_explicit_out_of_place(self):
        tensor = torch.tensor([1.0, 2.0])

        def reduce_in_place(target, **_kwargs):
            target.add_(10)

        with (
            patch.object(collective, "_get_rocm_rccl", return_value=None),
            patch.object(
                collective, "_get_flashinfer_allreduce"
            ) as flashinfer,
            patch.object(collective, "_get_symm_mem") as symm_mem,
            patch.object(collective, "_get_group", return_value=object()),
            patch.object(
                torch.distributed, "all_reduce", side_effect=reduce_in_place
            ),
        ):
            flashinfer.return_value.get_flashinfer_allreduce.return_value = None
            symm_mem.return_value.get_symm_mem_communicator.return_value = None
            result = collective.all_reduce(
                tensor, collective.Group.TP, inplace=False
            )

        self.assertIsNot(result, tensor)
        torch.testing.assert_close(tensor, torch.tensor([1.0, 2.0]))
        torch.testing.assert_close(result, torch.tensor([11.0, 12.0]))

    def test_capture_allreduce_supports_explicit_out_of_place(self):
        tensor = torch.tensor([1.0, 2.0])
        capture = MagicMock()
        capture.ensure_capture_comm_ready.return_value = None
        capture.should_use_capture_collectives.return_value = True
        capture.capture_all_reduce.side_effect = lambda target, _group: target.add_(
            10
        )

        with (
            patch.object(collective, "_get_rocm_rccl", return_value=capture),
            patch.object(collective, "_get_group", return_value=object()),
        ):
            result = collective.all_reduce(
                tensor, collective.Group.TP, inplace=False
            )

        self.assertIsNot(result, tensor)
        torch.testing.assert_close(tensor, torch.tensor([1.0, 2.0]))
        torch.testing.assert_close(result, torch.tensor([11.0, 12.0]))


if __name__ == "__main__":
    unittest.main()
