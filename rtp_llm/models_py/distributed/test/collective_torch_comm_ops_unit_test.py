# SPDX-License-Identifier: Apache-2.0

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from rtp_llm.models_py.distributed import collective_torch as collective


class CollectiveTorchCommOpsUnitTest(unittest.TestCase):
    def test_cuda_only_chunk_validation_leaves_rocm_initialization_unchanged(self):
        with (
            patch.object(torch.version, "hip", "7.0"),
            patch.object(torch.distributed, "all_gather_object") as gather,
        ):
            collective._validate_tp_moe_chunking_config(
                SimpleNamespace(tp_size=2, dp_size=1)
            )
        gather.assert_not_called()

    def test_tp_chunk_mismatch_rejected_on_enabled_and_disabled_ranks(self):
        configs = [("0", "overlap", "4096"), ("4", "overlap", "4096")]
        pc = SimpleNamespace(tp_size=2, dp_size=1)

        def gather(output, local, group):
            output[:] = configs

        for local in configs:
            with (
                patch.dict(os.environ, {"MOE_TP_CHUNKS": local[0]}, clear=True),
                patch.object(collective, "_get_group", return_value=object()),
                patch.object(
                    torch.distributed, "all_gather_object", side_effect=gather
                ),
            ):
                with self.assertRaisesRegex(ValueError, "differs across TP ranks"):
                    collective._validate_tp_moe_chunking_config(pc)

    def test_tp_prefill_signature_mismatch_is_rejected_before_collectives(self):
        pc = SimpleNamespace(tp_size=2, dp_size=1)
        local = {
            "MOE_TP_CHUNKS": "2",
            "MOE_TP_CHUNK_MODE": "overlap",
            "MOE_TP_CHUNK_MIN_TOKENS": "1",
            "MOE_TP_PREFILL_BACKEND": "flashinfer_sm12x",
            "MOE_TP_DIRECT_OUTPUT": "1",
            "MOE_TP_FUSION_MIN_TOKENS": "1",
            "DSV4_FP8_QUANT_KERNEL": "auto",
        }
        expected = ("2", "overlap", "1", "flashinfer_sm12x", "1", "1", "auto")
        for changed_index, changed_value in ((3, "default"), (4, "0")):
            with self.subTest(changed_index=changed_index):

                def gather(output, received, group):
                    self.assertEqual(received, expected)
                    remote = list(received)
                    remote[changed_index] = changed_value
                    output[:] = [received, tuple(remote)]

                with (
                    patch.dict(os.environ, local, clear=True),
                    patch.object(collective, "_get_group", return_value=object()),
                    patch.object(
                        torch.distributed, "all_gather_object", side_effect=gather
                    ),
                ):
                    with self.assertRaisesRegex(ValueError, "differs across TP ranks"):
                        collective._validate_tp_moe_chunking_config(pc)

    def test_matching_disabled_tp_chunk_config_still_participates(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(collective, "_get_group", return_value=object()),
            patch.object(
                torch.distributed,
                "all_gather_object",
                side_effect=lambda output, local, **_: output.__setitem__(
                    slice(None), [local, local]
                ),
            ) as gather,
        ):
            collective._validate_tp_moe_chunking_config(
                SimpleNamespace(tp_size=2, dp_size=1)
            )
            gather.assert_called_once()

    def test_async_allreduce_owns_storage_and_defers_consumer_wait(self):
        tensor = MagicMock(spec=torch.Tensor)
        tensor.is_cuda = True
        tensor.device = torch.device("cuda", 0)
        process_group, work = object(), MagicMock()
        with (
            patch.object(torch.version, "hip", None),
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
            patch.object(collective, "_get_group", return_value=process_group),
            patch.object(torch.distributed, "get_backend", return_value="nccl"),
            patch.object(torch.distributed, "all_reduce", return_value=work) as reduce,
            patch.object(collective, "_get_flashinfer_allreduce") as custom,
            patch.object(
                torch.cuda, "current_stream", side_effect=["consumer0", "consumer1"]
            ),
        ):
            pending = collective.all_reduce_async(tensor, collective.Group.TP)
            self.assertIs(pending.tensor, tensor)
            work.wait.assert_not_called()
            reduce.assert_called_once_with(
                tensor,
                op=torch.distributed.ReduceOp.SUM,
                group=process_group,
                async_op=True,
            )
            self.assertIs(pending.wait(), tensor)
            self.assertIs(pending.wait(), tensor)
            self.assertEqual(work.wait.call_count, 2)
            self.assertEqual(
                [call.args[0] for call in tensor.record_stream.call_args_list],
                ["consumer0", "consumer1"],
            )
            custom.assert_not_called()

    def test_async_allreduce_rejects_cpu_before_launch(self):
        with patch.object(torch.distributed, "all_reduce") as reduce:
            with self.assertRaisesRegex(ValueError, "CUDA tensor"):
                collective.all_reduce_async(torch.ones(2), collective.Group.TP)
            reduce.assert_not_called()

    def test_async_allreduce_out_of_place_preserves_input_ownership(self):
        tensor = MagicMock(spec=torch.Tensor)
        tensor.is_cuda = True
        with (
            patch.object(torch.version, "hip", None),
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
            patch.object(collective, "_get_group", return_value=object()),
            patch.object(torch.distributed, "get_backend", return_value="nccl"),
            patch.object(torch.distributed, "all_reduce") as reduce,
        ):
            pending = collective.all_reduce_async(
                tensor, collective.Group.TP, inplace=False
            )
            tensor.clone.assert_called_once()
            self.assertIs(pending.tensor, tensor.clone.return_value)
            self.assertIs(reduce.call_args.args[0], tensor.clone.return_value)

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
