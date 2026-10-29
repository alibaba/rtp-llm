# SPDX-License-Identifier: Apache-2.0

import sys
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from rtp_llm.models_py.distributed import collective_torch as collective


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


class CollectiveTorchCrossHostWorkspaceUnitTest(unittest.TestCase):
    """Exercise workspace admission without initializing CUDA or native kernels."""

    def _workspace(
        self,
        kind,
        *,
        world_size=8,
        same_host=False,
        peer_fabric="matching",
        bad_allocation=None,
        bad_allocation_dtype=None,
        peer_allocation_failure=False,
        missing_multicast=False,
        mapping_failure=False,
    ):
        group = MagicMock()
        group.size.return_value = world_size
        group.rank.return_value = 0
        fabric = (b"shared-nvlink-cluster", 7)
        props = SimpleNamespace(
            major=10,
            minor=3,
            uuid="rank-0",
            multi_processor_count=148,
        )
        symm = MagicMock()
        symm.get_backend.return_value = "CUDA"
        allocations, mappings, votes = [], [], []

        def allocate(size, **kwargs):
            tensor = MagicMock()
            elements = size if isinstance(size, int) else size[0] * size[1]
            tensor.numel.return_value = elements
            tensor.dtype = kwargs["dtype"]
            tensor.element_size.return_value = {
                torch.uint8: 1,
                torch.bfloat16: 2,
                torch.int32: 4,
            }[tensor.dtype]
            allocations.append(tensor)
            return tensor

        def rendezvous(tensor, **_kwargs):
            mappings.append(tensor)
            if mapping_failure:
                raise RuntimeError("peer mapping failed")
            handle = MagicMock()
            handle.multicast_ptr = 0 if missing_multicast else 4096
            return handle

        def gather_metadata(output, local, **_kwargs):
            for rank in range(world_size):
                item = list(local)
                item[0] = "host-a" if same_host or rank < world_size // 2 else "host-b"
                item[1] = f"rank-{rank}"
                if rank == world_size - 1 and peer_fabric != "matching":
                    item[-1] = (
                        None if peer_fabric == "missing" else (b"other-cluster", 7)
                    )
                output[rank] = tuple(item)

        def agree(local_ok, *_args):
            votes.append(local_ok)
            return local_ok and not (peer_allocation_failure and len(votes) == 3)

        probe_count = 0

        def is_fabric(tensor):
            nonlocal probe_count
            index = probe_count
            probe_count += 1
            return index != bad_allocation and tensor.dtype != bad_allocation_dtype

        symm.empty.side_effect = allocate
        symm.rendezvous.side_effect = rendezvous
        nvml = MagicMock()
        nvml.NVML_P2P_STATUS_OK = 0
        nvml.nvmlDeviceGetP2PStatus.return_value = 0
        with ExitStack() as stack:
            stack.enter_context(
                patch.dict(
                    sys.modules,
                    {
                        "torch.distributed._symmetric_memory": symm,
                        "pynvml": nvml,
                    },
                )
            )
            stack.enter_context(
                patch.object(
                    torch.distributed,
                    "_symmetric_memory",
                    symm,
                    create=True,
                )
            )
            for target, name, value in [
                (torch.cuda, "is_current_stream_capturing", False),
                (torch.cuda, "get_device_properties", props),
                (torch.cuda, "synchronize", None),
                (torch, "zeros", MagicMock()),
                (torch.distributed, "barrier", None),
                (collective, "_nvlink_fabric", fabric),
                (collective, "_load_push_rs_kernel", MagicMock()),
                (
                    collective,
                    "_load_custom_ag_kernels",
                    (MagicMock(), MagicMock()),
                ),
            ]:
                stack.enter_context(patch.object(target, name, return_value=value))
            stack.enter_context(
                patch.object(
                    torch.distributed,
                    "all_gather_object",
                    side_effect=gather_metadata,
                )
            )
            stack.enter_context(
                patch.object(collective, "_all_ready", side_effect=agree)
            )
            probe = stack.enter_context(
                patch.object(
                    collective,
                    "_is_fabric_allocation",
                    side_effect=is_fabric,
                )
            )
            if kind == "rs":
                state = collective.create_push_reduce_scatter(
                    group,
                    "cuda:0",
                    max_m=64,
                    n=7168,
                )
            else:
                state = collective.create_custom_all_gather(
                    group,
                    "cuda:0",
                    max_m=64,
                    k=7168,
                    fp8=kind == "fp8_ag",
                )
        return SimpleNamespace(
            state=state,
            symm=symm,
            probe=probe,
            allocations=allocations,
            mappings=mappings,
            votes=votes,
        )

    def test_cross_host_nvlink_initializes_all_collectives(self):
        for kind in ("bf16_ag", "fp8_ag", "rs"):
            for world_size in (4, 8):
                with self.subTest(kind=kind, world_size=world_size):
                    result = self._workspace(kind, world_size=world_size)
                    self.assertIsNotNone(result.state)
                    expected = {"fp8_ag": 4, "bf16_ag": 3, "rs": 1}[kind]
                    self.assertEqual(result.probe.call_count, expected)
                    self.assertEqual(len(result.mappings), expected)
                    self.assertTrue(all(result.votes))

    def test_same_host_does_not_require_fabric_allocations(self):
        for kind in ("bf16_ag", "fp8_ag", "rs"):
            with self.subTest(kind=kind):
                result = self._workspace(
                    kind,
                    world_size=4,
                    same_host=True,
                    bad_allocation=0,
                )
                self.assertIsNotNone(result.state)
                result.probe.assert_not_called()

    def test_cross_host_rejects_missing_or_different_nvlink_domain(self):
        for kind in ("bf16_ag", "fp8_ag", "rs"):
            for fabric in ("missing", "different"):
                with self.subTest(kind=kind, fabric=fabric):
                    result = self._workspace(kind, peer_fabric=fabric)
                    self.assertIsNone(result.state)
                    result.symm.empty.assert_not_called()
                    result.symm.rendezvous.assert_not_called()

    def test_cross_host_rejects_non_fabric_allocation_before_rendezvous(self):
        for kind in ("bf16_ag", "fp8_ag", "rs"):
            with self.subTest(kind=kind):
                result = self._workspace(kind, bad_allocation=0)
                self.assertIsNone(result.state)
                result.probe.assert_called_once()
                result.symm.rendezvous.assert_not_called()
                self.assertFalse(result.votes[-1])

    def test_cross_host_fp8_checks_scale_allocation(self):
        result = self._workspace("fp8_ag", bad_allocation_dtype=torch.int32)
        self.assertIsNone(result.state)
        self.assertEqual(result.probe.call_count, 4)
        self.assertEqual(result.probe.call_args.args[0].dtype, torch.int32)
        result.symm.rendezvous.assert_not_called()

    def test_peer_allocation_failure_prevents_local_rendezvous(self):
        for kind in ("bf16_ag", "fp8_ag", "rs"):
            with self.subTest(kind=kind):
                result = self._workspace(kind, peer_allocation_failure=True)
                self.assertIsNone(result.state)
                result.symm.rendezvous.assert_not_called()

    def test_cross_host_allgather_requires_multicast_mapping(self):
        for kind in ("bf16_ag", "fp8_ag"):
            with self.subTest(kind=kind):
                result = self._workspace(kind, missing_multicast=True)
                self.assertIsNone(result.state)
                self.assertFalse(result.votes[-1])

    def test_cross_host_mapping_failure_does_not_enable_collective(self):
        for kind in ("bf16_ag", "fp8_ag", "rs"):
            with self.subTest(kind=kind):
                result = self._workspace(kind, mapping_failure=True)
                self.assertIsNone(result.state)
                self.assertFalse(result.votes[-1])


if __name__ == "__main__":
    unittest.main()
