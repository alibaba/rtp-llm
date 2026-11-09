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
        self.options = Qwen4CollectiveOptions("sm120_ipc", model_type="qwen4_exp")
        self.patches = [
            patch.object(qwen4_ar.dist, "get_world_size", return_value=2),
            patch.object(qwen4_ar.dist, "get_rank", return_value=0),
            patch.object(qwen4_ar.dist, "all_gather_object", side_effect=self.agree),
        ]
        for item in self.patches:
            item.start()
        self.addCleanup(self.cleanup)

    @staticmethod
    def agree(peers, value, group):
        peers[:] = [value] * len(peers)

    def cleanup(self):
        qwen4_ar.destroy_qwen4_allreduce()
        for item in reversed(self.patches):
            item.stop()

    def init(self, options=None):
        return qwen4_ar.init_qwen4_allreduce(
            self.group, self.device, topology(), options
        )

    def test_default_never_probes_or_allocates_optional_backends(self):
        with (
            patch.object(qwen4_ar, "_capability_error") as probe,
            patch.object(qwen4_ar, "Qwen4IpcAllReduce") as factory,
        ):
            self.assertIsNone(self.init())
        probe.assert_not_called()
        factory.assert_not_called()
        self.assertIsNone(qwen4_ar.get_canonical_p2p())

    def test_options_reject_invalid_combinations(self):
        for backend, large in (("auto", False), ("nccl", True), ("sm120_p2p", True)):
            with self.subTest(backend=backend), self.assertRaises(ValueError):
                Qwen4CollectiveOptions(backend, large)

    def test_nonqwen_owner_and_wrong_topology_fail_before_probe(self):
        with patch.object(qwen4_ar, "_capability_error") as probe:
            with self.assertRaisesRegex(ValueError, "owned by Qwen4"):
                self.init(Qwen4CollectiveOptions("sm120_ipc", model_type="qwen_3"))
            with self.assertRaisesRegex(ValueError, "TP8"):
                self.init(Qwen4CollectiveOptions("sm120_p2p", model_type="qwen4_exp"))
        probe.assert_not_called()

    def test_rank_policy_mismatch_fails_before_optional_initialization(self):
        def disagree(peers, state, group):
            peers[:] = [state, (Qwen4CollectiveOptions(), None, False)]

        with (
            patch.object(qwen4_ar.dist, "all_gather_object", side_effect=disagree),
            patch.object(qwen4_ar, "_capability_error") as probe,
            self.assertRaisesRegex(ValueError, "differs across TP ranks"),
        ):
            self.init(self.options)
        probe.assert_not_called()

    def test_one_rank_probe_failure_falls_back_before_allocation(self):
        def peer_failed(peers, value, group):
            if isinstance(value, tuple):
                self.agree(peers, value, group)
            else:
                peers[:] = [None, "missing peer API"]

        with (
            patch.object(qwen4_ar.dist, "all_gather_object", side_effect=peer_failed),
            patch.object(qwen4_ar, "_capability_error", return_value=None),
            patch.object(qwen4_ar, "Qwen4IpcAllReduce") as factory,
        ):
            self.assertIsNone(self.init(self.options))
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
        self.init()
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
        workspace = FlashInferIpcAllReduce.__new__(FlashInferIpcAllReduce)
        workspace.max_bytes = 128 * 1024 * 1024 if large else 32 * 2560 * 2
        workspace.registered_graph_inputs = True
        with patch(
            "rtp_llm.models_py.distributed.flashinfer_ipc_all_reduce.FlashInferIpcAllReduce",
            return_value=workspace,
        ) as factory:
            context = qwen4_ar.Qwen4IpcAllReduce(
                self.group,
                self.device,
                Qwen4CollectiveOptions("sm120_ipc", large, "qwen4_exp"),
            )
        factory.assert_called_once_with(
            self.group, self.device, max_bytes=workspace.max_bytes
        )
        return context

    def test_shape_capacity_capture_and_large_prefill_domain(self):
        tensor = SimpleNamespace(
            is_cuda=True,
            device=self.device,
            dtype=torch.bfloat16,
            ndim=2,
            shape=(4, 2560),
            is_contiguous=lambda: True,
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

    def test_model_prefill_workspace_obeys_same_option(self):
        from rtp_llm.models_py.modules.qwen4_exp import tp_reduce

        context = SimpleNamespace(
            disabled=False, cpp_small_enabled=True, large_prefill=False
        )
        pc = SimpleNamespace(
            role_type="PREFILL", tp_size=2, dp_size=1, local_world_size=2
        )
        with (
            patch.object(tp_reduce, "get_qwen4_allreduce", return_value=context),
            patch.object(tp_reduce, "Qwen4PrefillReduce") as factory,
        ):
            self.assertIsNone(
                tp_reduce.prepare_prefill_reduce(SimpleNamespace(is_mtp=False), pc)
            )
            factory.assert_not_called()
            context.large_prefill = True
            self.assertIs(
                tp_reduce.prepare_prefill_reduce(SimpleNamespace(is_mtp=False), pc),
                factory.return_value,
            )
            factory.assert_called_once_with(context)

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


if __name__ == "__main__":
    unittest.main()
