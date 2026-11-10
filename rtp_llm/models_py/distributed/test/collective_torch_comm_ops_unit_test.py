# SPDX-License-Identifier: Apache-2.0

import os
import socket
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from rtp_llm.models_py.distributed import collective_torch as collective


class CpuTpBroadcasterPathTest(unittest.TestCase):
    def test_long_tmpdir_produces_bindable_isolated_paths(self):
        with (
            tempfile.TemporaryDirectory(dir="/tmp") as root,
            patch.dict(
                os.environ,
                {
                    "TMPDIR": os.path.join(root, "worker-" + "x" * 180),
                    "RTP_LLM_CPU_TP_BROADCASTER_DIR": "",
                    "RTP_LLM_CPU_TP_BROADCASTER_ID": "path-test",
                },
            ),
        ):
            config = SimpleNamespace(dp_rank=0)
            make_path = collective._make_cpu_tp_broadcaster_base_path
            path = make_path(config, 12345)
            self.assertEqual(path, make_path(config, 12345))
            self.assertNotEqual(path, make_path(SimpleNamespace(dp_rank=1), 12345))
            with patch.dict(os.environ, {"TMPDIR": os.environ["TMPDIR"] + "-other"}):
                self.assertNotEqual(path, make_path(config, 12345))
            socket_path = path + "_0.sock"
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
                listener.bind(socket_path)
                self.addCleanup(os.unlink, socket_path)

    def test_explicit_directory_is_preserved(self):
        with (
            tempfile.TemporaryDirectory(dir="/tmp") as root,
            patch.dict(
                os.environ,
                {
                    "TMPDIR": "/unused/" + "x" * 180,
                    "RTP_LLM_CPU_TP_BROADCASTER_DIR": root,
                    "RTP_LLM_CPU_TP_BROADCASTER_ID": "path-test",
                },
            ),
        ):
            path = collective._make_cpu_tp_broadcaster_base_path(
                SimpleNamespace(dp_rank=0), 12345
            )
            self.assertEqual(os.path.dirname(path), root)

    def test_explicit_overlong_directory_still_raises(self):
        with (
            patch.dict(
                os.environ,
                {"RTP_LLM_CPU_TP_BROADCASTER_DIR": "/tmp/" + "x" * 180},
            ),
            patch.object(collective.os, "makedirs") as makedirs,
        ):
            with self.assertRaisesRegex(ValueError, "UDS path too long"):
                collective._make_cpu_tp_broadcaster_base_path(
                    SimpleNamespace(dp_rank=0), 12345
                )
            makedirs.assert_not_called()


class CollectiveTorchCommOpsUnitTest(unittest.TestCase):
    def _registered_callbacks(self, config=None, process_group=None):
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
        ):
            collective._register_process_groups_to_cpp()

        compute_ops.register_comm_ops.assert_called_once()
        return compute_ops.register_comm_ops.call_args.args

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
