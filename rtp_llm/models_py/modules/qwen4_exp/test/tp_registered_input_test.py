import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from rtp_llm.models_py.distributed import flashinfer_all_reduce as parent
from rtp_llm.models_py.distributed.flashinfer_ipc_all_reduce import (
    FlashInferIpcAllReduce,
)
from rtp_llm.models_py.modules.qwen4_exp import tp_reduce


class TpRegisteredInputTest(unittest.TestCase):
    def workspace(self, enabled):
        workspace = FlashInferIpcAllReduce.__new__(FlashInferIpcAllReduce)
        workspace.registered_graph_inputs = enabled
        workspace.capture_calls = 0
        workspace.handle = 73
        workspace.buffer = [912]
        workspace.rank = 0
        workspace.max_bytes = 163840
        workspace.ops = MagicMock()
        return workspace

    def test_large_prefill_requires_opt_in_exact_domain_and_eager_execution(self):
        workspace = self.workspace(False)
        workspace.max_bytes = 128 * 1024 * 1024
        for enabled, capturing, rows, expected in (
            (False, False, 4096, False),
            (True, False, 4096, True),
            (True, True, 4096, False),
            (True, False, 4097, False),
            (True, False, 12288, True),
            (True, False, 32, True),
        ):
            workspace.large_prefill = enabled
            with patch(
                "torch.cuda.is_current_stream_capturing", return_value=capturing
            ):
                self.assertEqual(
                    workspace.is_buffer_size_sufficient(2, rows, 2560, torch.bfloat16),
                    expected,
                )
                self.assertFalse(
                    workspace.is_buffer_size_sufficient(4, rows, 2560, torch.bfloat16)
                )
                self.assertFalse(
                    workspace.is_buffer_size_sufficient(2, rows, 2560, torch.float32)
                )

    def test_direct_registration_requires_capture_opt_in_and_disjoint_storage(self):
        base = torch.empty(200)
        source = base[:100]
        separate = torch.empty_like(source)
        partial = base[50:150]
        for enabled, capturing, output, expected in (
            (False, True, separate, 912),
            (True, False, separate, 912),
            (True, True, source, 912),
            (True, True, partial, 912),
            (True, True, separate, 0),
        ):
            workspace = self.workspace(enabled)
            with patch(
                "torch.cuda.is_current_stream_capturing", return_value=capturing
            ):
                actual = workspace.all_reduce(source, output)
            self.assertIs(actual, output)
            workspace.ops.all_reduce.assert_called_once_with(
                73, source, output, expected, 163840, 36
            )
            self.assertEqual(workspace.capture_calls, int(capturing))

    def test_independent_storage_wrappers_with_overlapping_addresses_stage(self):
        import numpy as np

        buffer = np.empty(200, dtype=np.float32)
        source = torch.from_numpy(buffer[:100])
        output = torch.from_numpy(buffer[50:150])
        self.assertNotEqual(
            source.untyped_storage().data_ptr(), output.untyped_storage().data_ptr()
        )
        workspace = self.workspace(True)
        with patch("torch.cuda.is_current_stream_capturing", return_value=True):
            workspace.all_reduce(source, output)
        workspace.ops.all_reduce.assert_called_once_with(
            73, source, output, 912, 163840, 36
        )

    def test_draft_capture_disabled_and_eager_never_request_direct_input(self):
        communicator = parent.FlashInferAllReduce.__new__(parent.FlashInferAllReduce)
        communicator._workspace = self.workspace(True)
        communicator.should_use = MagicMock(return_value=True)
        tensor = torch.empty(4, 2560)
        for allowed, capturing, enabled, expected in (
            (False, True, True, False),
            (True, False, True, False),
            (True, True, False, False),
            (True, True, True, True),
        ):
            communicator.allow_ipc_capture = allowed
            communicator._workspace.registered_graph_inputs = enabled
            with patch(
                "torch.cuda.is_current_stream_capturing", return_value=capturing
            ):
                self.assertEqual(
                    communicator.should_use_registered_graph_input(tensor), expected
                )

    def test_explicit_qwen4_dispatch_uses_out_of_place_only_when_admitted(self):
        tensor = SimpleNamespace(is_cuda=True)
        group = tp_reduce.collective_torch.Group.TP
        communicator = SimpleNamespace(
            should_use_registered_graph_input=MagicMock(return_value=True)
        )
        with patch.object(
            tp_reduce, "get_flashinfer_allreduce", return_value=communicator
        ), patch.object(tp_reduce.collective_torch, "all_reduce") as reduce:
            tp_reduce.all_reduce_target_graph(tensor, group)
            reduce.assert_called_once_with(tensor, group, inplace=False)
        communicator.should_use_registered_graph_input.return_value = False
        with patch.object(
            tp_reduce, "get_flashinfer_allreduce", return_value=communicator
        ), patch.object(tp_reduce.collective_torch, "all_reduce") as reduce:
            tp_reduce.all_reduce_target_graph(tensor, group)
            reduce.assert_called_once_with(tensor, group)
        with patch.object(
            tp_reduce, "get_flashinfer_allreduce", return_value=None
        ), patch.object(tp_reduce.collective_torch, "all_reduce") as reduce:
            tp_reduce.all_reduce_target_graph(tensor, group)
            reduce.assert_called_once_with(tensor, group)

    def test_installer_reaches_actual_routed_finalizer_and_keeps_other_routers(self):
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.pure_tp_router import (
            PureTpRouterNvfp4Cutlass,
        )

        router = PureTpRouterNvfp4Cutlass.__new__(PureTpRouterNvfp4Cutlass)
        attention = SimpleNamespace()
        mlp = SimpleNamespace(fused_moe=SimpleNamespace(router=router))
        tp_reduce.install_target_graph_all_reduce(attention, mlp)
        self.assertIs(attention.tp_all_reduce, tp_reduce.all_reduce_target_graph)
        self.assertIs(mlp.tp_all_reduce, tp_reduce.all_reduce_target_graph)
        self.assertIs(router.tp_all_reduce, tp_reduce.all_reduce_target_graph)
        other = SimpleNamespace()
        mlp.fused_moe.router = other
        tp_reduce.install_target_graph_all_reduce(attention, mlp)
        self.assertFalse(hasattr(other, "tp_all_reduce"))

    def test_routed_finalizer_uses_scoped_dispatch_but_preserves_skip_and_tp1(self):
        from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
            CombineForwardPayload,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.pure_tp_router import (
            PureTpRouterNvfp4Cutlass,
        )

        router = PureTpRouterNvfp4Cutlass.__new__(PureTpRouterNvfp4Cutlass)
        source = torch.randn(4, 2560)
        reduced = torch.randn_like(source)
        router.tp_all_reduce = MagicMock(return_value=reduced)
        for tp_size, skip, calls in ((2, False, 1), (2, True, 0), (1, False, 0)):
            router.tp_size = tp_size
            router.tp_all_reduce.reset_mock()
            result = router.finalize(
                CombineForwardPayload(fused_expert_output=source),
                torch.empty(4, 10),
                torch.empty(4, 10, dtype=torch.int32),
                False,
                {"skip_tp_allreduce": skip},
            )
            self.assertIs(result, reduced if calls else source)
            self.assertEqual(router.tp_all_reduce.call_count, calls)
            if calls:
                router.tp_all_reduce.assert_called_once_with(
                    source, group=tp_reduce.collective_torch.Group.TP
                )


if __name__ == "__main__":
    unittest.main()
