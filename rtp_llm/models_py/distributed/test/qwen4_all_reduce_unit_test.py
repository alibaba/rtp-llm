import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from rtp_llm.config.qwen4_collective_config import Qwen4CollectiveOptions
from rtp_llm.config.server_config_setup import set_parallelism_config
from rtp_llm.models_py.distributed import collective_torch as collective
from rtp_llm.models_py.distributed import qwen4_all_reduce as qwen4_ar
from rtp_llm.models_py.distributed.flashinfer_ipc_all_reduce import (
    FlashInferIpcAllReduce,
    IpcInitializationError,
)
from rtp_llm.ops import ParallelismConfig


def topology(size=2):
    pc = ParallelismConfig()
    pc.tp_size = pc.ep_size = pc.world_size = pc.local_world_size = size
    set_parallelism_config(pc)
    return pc


class Qwen4AllReduceUnitTest(unittest.TestCase):
    def setUp(self):
        self.group = object()
        self.device = torch.device("cuda", 0)
        self.options = Qwen4CollectiveOptions("sm120_ipc")
        self.patches = [
            patch.object(qwen4_ar, "_configured_options", Qwen4CollectiveOptions()),
            patch.object(qwen4_ar.dist, "get_world_size", return_value=2),
            patch.object(qwen4_ar.dist, "get_rank", return_value=0),
            patch.object(qwen4_ar, "exchange_group_state", side_effect=self.agree),
        ]
        for item in self.patches:
            item.start()
        self.addCleanup(self.cleanup)

    @staticmethod
    def agree(group, value, tag):
        return [value, value]

    def cleanup(self):
        qwen4_ar.destroy_qwen4_allreduce()
        for item in reversed(self.patches):
            item.stop()

    def init(self, options=None):
        return qwen4_ar.init_qwen4_allreduce(
            self.group, self.device, topology(), options
        )

    def test_auto_selects_supported_ipc_without_a_model_name(self):
        with (
            patch.object(qwen4_ar, "_capability_error", return_value=None),
            patch.object(qwen4_ar, "Qwen4IpcAllReduce") as factory,
        ):
            self.assertIs(self.init(), factory.return_value)
            self.assertEqual(factory.call_args.args[2].backend, "sm120_ipc")

    def test_explicit_nccl_never_probes_or_allocates(self):
        with (
            patch.object(qwen4_ar, "_capability_error") as probe,
            patch.object(qwen4_ar, "Qwen4IpcAllReduce") as factory,
        ):
            self.assertIsNone(self.init(Qwen4CollectiveOptions("nccl")))
        probe.assert_not_called()
        factory.assert_not_called()

    def test_auto_unsupported_topology_falls_back(self):
        with (
            patch.object(qwen4_ar, "_capability_error") as probe,
            patch.object(qwen4_ar, "Qwen4IpcAllReduce") as factory,
        ):
            self.assertIsNone(
                qwen4_ar.init_qwen4_allreduce(self.group, self.device, topology(8))
            )
        probe.assert_not_called()
        factory.assert_not_called()

    def test_options_reject_invalid_combinations(self):
        for backend, large in (("invalid", False), ("nccl", True), ("sm120_p2p", True)):
            with self.subTest(backend=backend), self.assertRaises(ValueError):
                Qwen4CollectiveOptions(backend, large)

    def test_explicit_wrong_topology_fails_before_probe(self):
        with patch.object(qwen4_ar, "_capability_error") as probe:
            with self.assertRaisesRegex(ValueError, "TP8"):
                self.init(Qwen4CollectiveOptions("sm120_p2p"))
        probe.assert_not_called()

    def test_rank_policy_mismatch_fails_before_optional_initialization(self):
        def disagree(group, state, tag):
            return [state, ["nccl", None, None, False]]

        with (
            patch.object(qwen4_ar, "exchange_group_state", side_effect=disagree),
            patch.object(qwen4_ar, "_capability_error") as probe,
            self.assertRaisesRegex(ValueError, "differs across TP ranks"),
        ):
            self.init(self.options)
        probe.assert_not_called()

    def test_one_rank_probe_failure_falls_back_before_allocation(self):
        def peer_failed(group, value, tag):
            if tag == "capabilities":
                return [value, [value[0], value[1], "missing peer API"]]
            return self.agree(group, value, tag)

        with (
            patch.object(qwen4_ar, "exchange_group_state", side_effect=peer_failed),
            patch.object(qwen4_ar, "_capability_error", return_value=None),
            patch.object(qwen4_ar, "Qwen4IpcAllReduce") as factory,
        ):
            self.assertIsNone(self.init(self.options))
        factory.assert_not_called()

    def test_auto_rank_topology_disagreement_falls_back_together(self):
        def disagree(group, state, tag):
            if tag == "capabilities":
                return [state, ["nccl", state[1], None]]
            return self.agree(group, state, tag)

        with (
            patch.object(qwen4_ar, "exchange_group_state", side_effect=disagree),
            patch.object(qwen4_ar, "_capability_error", return_value=None),
            patch.object(qwen4_ar, "Qwen4IpcAllReduce") as factory,
        ):
            self.assertIsNone(self.init())
        factory.assert_not_called()

    def test_agreed_local_setup_failure_can_fall_back(self):
        with (
            patch.object(qwen4_ar, "_capability_error", return_value=None),
            patch.object(
                qwen4_ar,
                "Qwen4IpcAllReduce",
                side_effect=IpcInitializationError("allocation failed"),
            ),
        ):
            self.assertIsNone(self.init(self.options))

    def test_collective_failure_propagates(self):
        with (
            patch.object(qwen4_ar, "_capability_error", return_value=None),
            patch.object(
                qwen4_ar,
                "Qwen4IpcAllReduce",
                side_effect=RuntimeError("collective failed"),
            ),
            self.assertRaisesRegex(RuntimeError, "collective failed"),
        ):
            self.init(self.options)

    def test_initialization_is_idempotent_and_rebuilds_after_destroy(self):
        with (
            patch.object(qwen4_ar, "_capability_error", return_value=None),
            patch.object(qwen4_ar, "Qwen4IpcAllReduce") as factory,
        ):
            first = self.init(self.options)
            self.assertIs(self.init(self.options), first)
            factory.assert_called_once()
            qwen4_ar.destroy_qwen4_allreduce()
            first.destroy.assert_called_once()
            self.init(self.options)
            self.assertEqual(factory.call_count, 2)

    def test_policy_change_requires_teardown(self):
        self.init(Qwen4CollectiveOptions("nccl"))
        with self.assertRaisesRegex(RuntimeError, "before changing"):
            self.init(self.options)

    def test_public_initializer_forwards_one_policy(self):
        module = MagicMock()
        module.get_canonical_p2p.return_value = None
        pc = topology()
        with (
            patch.object(collective, "_get_qwen4_allreduce", return_value=module),
            patch.object(collective, "_get_group", return_value=self.group),
        ):
            collective._init_qwen4_allreduce(pc, self.options)
        module.init_qwen4_allreduce.assert_called_once_with(
            self.group, self.device, pc, self.options
        )

    def make_ipc(self, large=False):
        def workspace(group, device, max_bytes):
            result = FlashInferIpcAllReduce.__new__(FlashInferIpcAllReduce)
            result.max_bytes = max_bytes
            result.registered_graph_inputs = True
            return result

        with (
            patch(
                "rtp_llm.models_py.distributed.flashinfer_ipc_all_reduce.FlashInferIpcAllReduce",
                side_effect=workspace,
            ) as factory,
            patch.object(
                torch.cuda, "mem_get_info", return_value=(10 * 1024**3, 72 * 1024**3)
            ),
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
        ):
            context = qwen4_ar.Qwen4IpcAllReduce(
                self.group, self.device, Qwen4CollectiveOptions("sm120_ipc", large)
            )
        self.assertEqual(factory.call_args_list[0].kwargs["max_bytes"], 32 * 2560 * 2)
        if large:
            self.assertIsNot(context._workspace, context._prefill_workspace)
        return context

    def test_shape_capacity_capture_and_large_prefill_domain(self):
        tensor = SimpleNamespace(
            is_cuda=True,
            device=self.device,
            dtype=torch.bfloat16,
            ndim=2,
            shape=(4, 2560),
            is_contiguous=lambda: True,
            data_ptr=lambda: 16,
        )
        for large in (False, True):
            context = self.make_ipc(large)
            with patch("torch.cuda.is_current_stream_capturing", return_value=False):
                for rows in (1, 4, 16, 32):
                    tensor.shape = (rows, 2560)
                    self.assertTrue(context.should_use_cpp_small(tensor))
                for rows in (4093, 4094, 4095, 4096, 8192, 12288, 16384):
                    tensor.shape = (rows, 2560)
                    self.assertEqual(context.should_use(tensor), large)
                for shape in ((0, 2560), (33, 2560), (4, 5120)):
                    tensor.shape = shape
                    self.assertFalse(context.should_use(tensor))
            tensor.shape = (4, 2560)
            with patch("torch.cuda.is_current_stream_capturing", return_value=True):
                self.assertFalse(context.should_use(tensor))
                context.allow_ipc_capture = True
                self.assertTrue(context.should_use_registered_graph_input(tensor))
                tensor.shape = (4096, 2560)
                self.assertFalse(context.should_use(tensor))

    def test_prefill_budget_failure_preserves_decode_workspace(self):
        context = self.make_ipc()
        context._prefill_policy = None
        small = context._workspace
        with (
            patch.object(
                torch.cuda, "mem_get_info", return_value=(1024**2, 72 * 1024**3)
            ),
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
            patch(
                "rtp_llm.models_py.distributed.flashinfer_ipc_all_reduce.FlashInferIpcAllReduce"
            ) as factory,
        ):
            context.prepare_large_prefill()
        factory.assert_not_called()
        self.assertFalse(context.large_prefill)
        self.assertIs(context._workspace, small)

    def test_freeze_prevents_request_time_prefill_allocation(self):
        context = self.make_ipc()
        context._prefill_policy = None
        qwen4_ar.freeze_workspaces()
        with patch.object(torch.cuda, "mem_get_info") as probe:
            context.prepare_large_prefill()
        probe.assert_not_called()

    def test_peer_budget_rejection_does_not_allocate(self):
        context = self.make_ipc()
        context._prefill_policy = None
        with (
            patch.object(
                torch.cuda, "mem_get_info", return_value=(10 * 1024**3, 72 * 1024**3)
            ),
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
            patch.object(qwen4_ar, "exchange_group_state", return_value=[True, False]),
            patch(
                "rtp_llm.models_py.distributed.flashinfer_ipc_all_reduce.FlashInferIpcAllReduce"
            ) as factory,
        ):
            context.prepare_large_prefill()
        factory.assert_not_called()

    def test_capture_registers_and_restores_admission(self):
        communicator = MagicMock()
        communicator.allow_ipc_capture = False
        with patch.object(qwen4_ar, "_communicator", communicator):
            qwen4_ar.begin_graph_capture()
            self.assertTrue(communicator.allow_ipc_capture)
            qwen4_ar.end_graph_capture()
            self.assertFalse(communicator.allow_ipc_capture)
        communicator.finish_graph_capture.assert_called_once()

    def test_capture_failure_restores_without_registration(self):
        communicator = MagicMock()
        with patch.object(qwen4_ar, "_communicator", communicator):
            qwen4_ar.begin_graph_capture()
            qwen4_ar.abort_graph_capture()
            self.assertFalse(communicator.allow_ipc_capture)
        communicator.finish_graph_capture.assert_not_called()

    def test_registration_failure_propagates_after_restoring_admission(self):
        communicator = MagicMock()
        communicator.finish_graph_capture.side_effect = RuntimeError(
            "registration failed"
        )
        with patch.object(qwen4_ar, "_communicator", communicator):
            qwen4_ar.begin_graph_capture()
            with self.assertRaisesRegex(RuntimeError, "registration failed"):
                qwen4_ar.end_graph_capture()
            self.assertFalse(communicator.allow_ipc_capture)
            self.assertFalse(qwen4_ar._capture_active)

    def test_configure_installs_cli_override_without_environment_mutation(self):
        import os

        config = SimpleNamespace(
            qwen4_all_reduce_backend="nccl",
            qwen4_all_reduce_large_prefill=False,
            parallelism_config=topology(),
        )
        environment = dict(os.environ)
        qwen4_ar.configure_collectives(config)
        self.assertEqual(dict(os.environ), environment)
        with patch.object(qwen4_ar, "_capability_error") as probe:
            self.assertIsNone(self.init())
        probe.assert_not_called()

    def test_unaligned_views_keep_backend_and_output_contract(self):
        for capturing in (False, True):
            for inplace in (False, True):
                with self.subTest(capturing=capturing, inplace=inplace):
                    source = torch.arange(4 * 2560 + 1, dtype=torch.bfloat16)[1:].view(
                        4, 2560
                    )
                    original = source.clone()
                    target = source if inplace else torch.empty_like(source)
                    workspace = FlashInferIpcAllReduce.__new__(FlashInferIpcAllReduce)
                    workspace.handle = 73
                    workspace.rank = 0
                    workspace.buffer = [912]
                    workspace.max_bytes = 32 * 2560 * 2
                    workspace.registered_graph_inputs = True
                    workspace.capture_calls = 0
                    workspace.ops = MagicMock()

                    def kernel(handle, value, output, staging, capacity, sms):
                        self.assertEqual(value.data_ptr() % 16, 0)
                        self.assertEqual(output.data_ptr() % 16, 0)
                        self.assertEqual(staging, 912)
                        output.copy_(value * 2)

                    workspace.ops.all_reduce.side_effect = kernel
                    with patch.object(
                        torch.cuda,
                        "is_current_stream_capturing",
                        return_value=capturing,
                    ):
                        self.assertIs(workspace.all_reduce(source, target), target)
                    torch.testing.assert_close(target, original * 2)
                    if not inplace:
                        torch.testing.assert_close(source, original)

    def test_collective_inplace_and_out_of_place_contract(self):
        tensor = torch.ones((2, 4))
        fast_path = MagicMock()
        fast_path.should_use.return_value = True
        fast_path.all_reduce_into.side_effect = lambda source, target: target.fill_(2)
        fast_path.all_reduce.return_value = torch.full_like(tensor, 3)
        module = SimpleNamespace(get_qwen4_allreduce=lambda: fast_path)
        with (
            patch.object(collective, "_get_qwen4_allreduce", return_value=module),
            patch.object(collective, "_canonical_p2p", None),
        ):
            self.assertIs(collective.all_reduce(tensor, collective.Group.TP), tensor)
            original = tensor.clone()
            result = collective.all_reduce(tensor, collective.Group.TP, inplace=False)
        torch.testing.assert_close(tensor, original)
        torch.testing.assert_close(result, torch.full_like(tensor, 3))


class CollectiveAgreementTest(unittest.TestCase):
    def test_store_rounds_do_not_reuse_stale_decisions(self):
        from rtp_llm.models_py.distributed import collective_utils

        store = torch.distributed.HashStore()
        group = SimpleNamespace(group_name="agreement_unit_test")
        with (
            patch.object(
                collective_utils.dist.distributed_c10d,
                "_get_default_store",
                return_value=store,
            ),
            patch.object(collective_utils.dist, "get_world_size", return_value=1),
            patch.object(collective_utils.dist, "get_rank", return_value=0),
        ):
            self.assertEqual(
                collective_utils.exchange_group_state(group, False, "probe"), [False]
            )
            self.assertEqual(
                collective_utils.exchange_group_state(group, True, "probe"), [True]
            )

    def test_missing_rank_timeout_propagates_without_gpu_collective(self):
        from rtp_llm.models_py.distributed import collective_utils

        store = MagicMock()
        store.wait.side_effect = RuntimeError("missing rank")
        with (
            patch.object(
                collective_utils.dist.distributed_c10d,
                "_get_default_store",
                return_value=store,
            ),
            patch.object(collective_utils.dist, "get_world_size", return_value=2),
            patch.object(collective_utils.dist, "get_rank", return_value=0),
            patch.object(collective_utils.dist, "all_reduce") as gpu_collective,
            self.assertRaisesRegex(RuntimeError, "missing rank"),
        ):
            collective_utils.exchange_group_state(
                SimpleNamespace(group_name="missing"), True, "probe"
            )
        self.assertLessEqual(store.wait.call_args.args[1].total_seconds(), 120)
        gpu_collective.assert_not_called()


if __name__ == "__main__":
    unittest.main()
