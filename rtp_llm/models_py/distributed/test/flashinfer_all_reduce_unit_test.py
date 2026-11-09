# SPDX-License-Identifier: Apache-2.0

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from rtp_llm.models_py.distributed import collective_torch as collective
from rtp_llm.models_py.distributed import flashinfer_all_reduce as flashinfer_ar


class _FakeTensor:
    is_cuda = True
    dtype = torch.bfloat16
    shape = (8, 5120)

    def dim(self):
        return 2

    def is_contiguous(self):
        return True

    def element_size(self):
        return 2


def _make_communicator(workspace=None):
    communicator = flashinfer_ar.FlashInferAllReduce.__new__(
        flashinfer_ar.FlashInferAllReduce
    )
    communicator.group = object()
    communicator.device = torch.device("cuda", 0)
    communicator.world_size = 2
    communicator.rank = 0
    communicator.disabled = False
    communicator._workspace = workspace
    communicator._hidden_dim = 5120
    communicator._dtype = torch.bfloat16
    communicator._max_num_tokens = 6553
    communicator._flashinfer_comm = MagicMock()
    return communicator


class FlashInferAllReduceUnitTest(unittest.TestCase):
    def tearDown(self):
        flashinfer_ar.destroy_flashinfer_allreduce()

    def test_existing_custom_ar_switch_is_opt_in(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(flashinfer_ar.enabled_by_env())
        for value in ("0", "false", "off", "no"):
            with (
                self.subTest(value=value),
                patch.dict(os.environ, {"FT_DISABLE_CUSTOM_AR": value}, clear=True),
            ):
                self.assertTrue(flashinfer_ar.enabled_by_env())
        with patch.dict(os.environ, {"FT_DISABLE_CUSTOM_AR": "1"}, clear=True):
            self.assertFalse(flashinfer_ar.enabled_by_env())

    def test_explicit_custom_ar_config_overrides_environment(self):
        with patch.dict(os.environ, {"FT_DISABLE_CUSTOM_AR": "0"}, clear=True):
            self.assertFalse(flashinfer_ar.enabled_by_config(True))
        with patch.dict(os.environ, {"FT_DISABLE_CUSTOM_AR": "1"}, clear=True):
            self.assertTrue(flashinfer_ar.enabled_by_config(False))

    def test_collective_forwards_explicit_custom_ar_config(self):
        communicator_module = MagicMock()
        group = object()
        config = SimpleNamespace(tp_size=2, local_rank=0, local_world_size=2)

        with (
            patch.object(
                collective,
                "_get_flashinfer_allreduce",
                return_value=communicator_module,
            ),
            patch.object(collective, "_get_group", return_value=group),
        ):
            collective._init_flashinfer_allreduce(
                config,
                disable_custom_all_reduce=True,
                ipc_all_reduce_large_prefill=True,
                enable_sm120_collectives=True,
            )

        communicator_module.init_flashinfer_allreduce.assert_called_once_with(
            group,
            torch.device("cuda", 0),
            single_node=True,
            disable_custom_all_reduce=True,
            large_prefill=True,
            enable_sm120_collectives=True,
        )

    def test_default_initialization_does_not_add_collectives(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(flashinfer_ar.dist, "get_world_size", return_value=2),
            patch.object(flashinfer_ar.dist, "get_rank", return_value=0),
            patch.object(flashinfer_ar.dist, "all_gather_object") as gather,
        ):
            communicator = flashinfer_ar.FlashInferAllReduce(
                object(), torch.device("cuda", 0), single_node=True
            )
        self.assertTrue(communicator.disabled)
        gather.assert_not_called()

    def test_existing_custom_ar_keeps_legacy_backend_on_sm120(self):
        comm = MagicMock()

        def agree(peers, value, group):
            peers[:] = [value, value]

        with (
            patch.dict(
                sys.modules,
                {"flashinfer": MagicMock(comm=comm), "flashinfer.comm": comm},
            ),
            patch.object(flashinfer_ar.dist, "get_world_size", return_value=2),
            patch.object(flashinfer_ar.dist, "get_rank", return_value=0),
            patch.object(
                flashinfer_ar.dist, "all_gather_object", side_effect=agree
            ) as gather,
            patch.object(torch.cuda, "get_device_capability", return_value=(12, 0)),
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
            patch.object(flashinfer_ar.dist, "all_reduce"),
            patch.object(torch, "tensor", return_value=MagicMock()),
            patch.dict(sys.modules, {"flashinfer.comm.mnnvl": MagicMock()}),
        ):
            communicator = flashinfer_ar.FlashInferAllReduce(
                object(),
                torch.device("cuda", 0),
                single_node=True,
                disable_custom_all_reduce=False,
            )
            tensor = _FakeTensor()
            tensor.shape = (4, 2560)
            self.assertFalse(communicator.cpp_small_enabled)
            self.assertTrue(communicator._create_workspace(tensor))
        comm.create_allreduce_fusion_workspace.assert_called_once()
        self.assertEqual(
            comm.create_allreduce_fusion_workspace.call_args.kwargs["backend"], "trtllm"
        )
        gather.assert_called_once_with(
            [(True, False, False)] * 2,
            (True, False, False),
            group=communicator.group,
        )

    def test_legacy_custom_ar_disables_when_a_peer_probe_failed(self):
        comm = MagicMock()

        def peer_failed(peers, value, group):
            self.assertEqual(value, (True, False, False))
            peers[:] = [value, (False, False, False)]

        with (
            patch.dict(
                sys.modules,
                {"flashinfer": MagicMock(comm=comm), "flashinfer.comm": comm},
            ),
            patch.object(flashinfer_ar.dist, "get_world_size", return_value=2),
            patch.object(flashinfer_ar.dist, "get_rank", return_value=0),
            patch.object(
                flashinfer_ar.dist, "all_gather_object", side_effect=peer_failed
            ),
        ):
            communicator = flashinfer_ar.FlashInferAllReduce(
                object(),
                torch.device("cuda", 0),
                single_node=True,
                disable_custom_all_reduce=False,
            )
        self.assertTrue(communicator.disabled)
        self.assertFalse(communicator.should_use(_FakeTensor()))
        comm.create_allreduce_fusion_workspace.assert_not_called()

    def test_default_tp8_skips_canonical_initialization(self):
        module = MagicMock()
        canonical = MagicMock()
        config = SimpleNamespace(
            tp_size=8, world_size=8, local_rank=0, local_world_size=8
        )
        with (
            patch.object(collective, "_get_flashinfer_allreduce", return_value=module),
            patch.object(collective, "_get_group", return_value=object()),
            patch.dict(
                sys.modules, {"rtp_llm.models_py.distributed.canonical_p2p": canonical}
            ),
        ):
            collective._init_flashinfer_allreduce(config)
        canonical.init_canonical_p2p.assert_not_called()

    def test_large_prefill_workspace_capacity_and_shape_policy(self):
        from rtp_llm.models_py.distributed.flashinfer_ipc_all_reduce import (
            FlashInferIpcAllReduce,
        )

        for enabled in (False, True):
            communicator = _make_communicator()
            communicator.cpp_small_enabled = True
            communicator._large_prefill = enabled
            workspace = MagicMock()
            tensor = _FakeTensor()
            tensor.shape = (4, 2560)
            with (
                patch("torch.cuda.is_current_stream_capturing", return_value=False),
                patch(
                    "rtp_llm.models_py.distributed.flashinfer_ipc_all_reduce.FlashInferIpcAllReduce",
                    return_value=workspace,
                ) as factory,
                patch("torch.tensor", return_value=MagicMock()),
                patch("torch.distributed.all_reduce"),
            ):
                self.assertTrue(communicator._create_workspace(tensor))
            capacity = 128 * 1024 * 1024 if enabled else 32 * 2560 * 2
            factory.assert_called_once_with(
                communicator.group, communicator.device, max_bytes=capacity
            )
            self.assertIs(workspace.large_prefill, enabled)
            policy = FlashInferIpcAllReduce.__new__(FlashInferIpcAllReduce)
            policy.max_bytes = capacity
            policy.large_prefill = enabled
            with patch("torch.cuda.is_current_stream_capturing", return_value=False):
                self.assertTrue(
                    policy.is_buffer_size_sufficient(2, 32, 2560, torch.bfloat16)
                )
                for rows in (4093, 4094, 4095, 4096, 8192, 12288, 16384):
                    self.assertEqual(
                        policy.is_buffer_size_sufficient(2, rows, 2560, torch.bfloat16),
                        enabled,
                    )
                self.assertFalse(
                    policy.is_buffer_size_sufficient(2, 33, 2560, torch.bfloat16)
                )
            with patch("torch.cuda.is_current_stream_capturing", return_value=True):
                self.assertFalse(
                    policy.is_buffer_size_sufficient(2, 4096, 2560, torch.bfloat16)
                )

    def test_target_reduce_default_and_invalid_peer_do_not_allocate(self):
        from rtp_llm.models_py.modules.qwen4_exp import tp_reduce

        communicator = SimpleNamespace(
            disabled=False, cpp_small_enabled=True, group=object()
        )
        config = SimpleNamespace(is_mtp=False)
        parallelism = SimpleNamespace(
            role_type="DECODE", tp_size=2, dp_size=1, local_world_size=2
        )
        for modes in (("legacy", "legacy"), ("ipc256", "invalid")):

            def gather(peers, value, group):
                peers[:] = modes

            with (
                self.subTest(modes=modes),
                patch.dict(os.environ, {}, clear=True),
                patch.object(
                    tp_reduce, "get_flashinfer_allreduce", return_value=communicator
                ),
                patch.object(tp_reduce.dist, "all_gather_object", side_effect=gather),
                patch.object(tp_reduce, "Qwen4TargetReduce") as factory,
            ):
                if "invalid" in modes:
                    with self.assertRaisesRegex(ValueError, "on every rank"):
                        tp_reduce.prepare_target_reduce(config, parallelism)
                else:
                    self.assertIsNone(
                        tp_reduce.prepare_target_reduce(config, parallelism)
                    )
                factory.assert_not_called()

    def test_shape_and_dtype_eligibility(self):
        workspace = MagicMock()
        workspace.is_buffer_size_sufficient.return_value = True
        communicator = _make_communicator(workspace)
        tensor = _FakeTensor()

        self.assertTrue(communicator.should_use(tensor))
        workspace.is_buffer_size_sufficient.assert_called_once_with(
            2, 8, 5120, torch.bfloat16
        )

        tensor.dtype = torch.float32
        self.assertFalse(communicator.should_use(tensor))

    def test_cpp_small_domain_keeps_prefill_and_other_widths_on_nccl(self):
        workspace = MagicMock()
        workspace.is_buffer_size_sufficient.return_value = True
        communicator = _make_communicator(workspace)
        communicator.cpp_small_enabled = True
        communicator._hidden_dim = 2560
        tensor = _FakeTensor()
        tensor.ndim = 2
        for rows in (1, 4, 16, 32):
            tensor.shape = (rows, 2560)
            self.assertTrue(communicator.should_use_cpp_small(tensor))
        for shape in ((0, 2560), (33, 2560), (4096, 2560), (4, 1280)):
            tensor.shape = shape
            self.assertFalse(communicator.should_use_cpp_small(tensor))
        tensor.shape = (4, 2560)
        communicator.cpp_small_enabled = False
        self.assertFalse(communicator.should_use_cpp_small(tensor))

    def test_oversized_prefill_falls_back_before_flashinfer_validation(self):
        workspace = MagicMock()
        communicator = _make_communicator(workspace)
        tensor = _FakeTensor()
        tensor.shape = (communicator._max_num_tokens + 1, 5120)

        self.assertFalse(communicator.should_use(tensor))
        workspace.is_buffer_size_sufficient.assert_not_called()

    def test_capture_never_silently_selects_nccl_before_warmup(self):
        communicator = _make_communicator()
        with patch("torch.cuda.is_current_stream_capturing", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "before CUDA graph capture"):
                communicator.should_use(_FakeTensor())

    def test_all_reduce_uses_safe_flashinfer_replay_contract(self):
        workspace = object()
        communicator = _make_communicator(workspace)
        communicator._flashinfer_comm.AllReduceFusionPattern.kAllReduce = 17
        output = object()
        communicator._flashinfer_comm.allreduce_fusion.return_value = output
        tensor = object()

        self.assertIs(communicator.all_reduce(tensor), output)
        communicator._flashinfer_comm.allreduce_fusion.assert_called_once_with(
            input=tensor,
            workspace=workspace,
            pattern=17,
            launch_with_pdl=True,
            trigger_completion_at_end=True,
        )

    def test_collective_default_preserves_inplace_contract(self):
        tensor = torch.ones((2, 4))
        fast_path = MagicMock()
        fast_path.should_use.return_value = True
        reduced = torch.full_like(tensor, 2)
        fast_path.all_reduce_into.side_effect = lambda source, target: target.copy_(
            reduced
        )
        module = SimpleNamespace(get_flashinfer_allreduce=lambda: fast_path)

        with patch.object(collective, "_get_flashinfer_allreduce", return_value=module):
            result = collective.all_reduce(tensor, collective.Group.TP)
        self.assertIs(result, tensor)
        torch.testing.assert_close(tensor, reduced)
        fast_path.should_use.assert_called_once_with(tensor)
        fast_path.all_reduce_into.assert_called_once_with(tensor, tensor)
        fast_path.all_reduce.assert_not_called()

    def test_collective_supports_explicit_out_of_place(self):
        tensor = torch.ones((2, 4))
        original = tensor.clone()
        fast_path = MagicMock()
        fast_path.should_use.return_value = True
        reduced = torch.full_like(tensor, 2)
        fast_path.all_reduce.return_value = reduced
        module = SimpleNamespace(get_flashinfer_allreduce=lambda: fast_path)

        with patch.object(collective, "_get_flashinfer_allreduce", return_value=module):
            result = collective.all_reduce(tensor, collective.Group.TP, inplace=False)
        self.assertIs(result, reduced)
        torch.testing.assert_close(tensor, original)

    def test_collective_preserves_inplace_contract(self):
        tensor = torch.ones((2, 4))
        fast_path = MagicMock()
        fast_path.should_use.return_value = True
        fast_path.all_reduce_into.side_effect = lambda source, target: target.fill_(2)
        module = SimpleNamespace(get_flashinfer_allreduce=lambda: fast_path)

        with patch.object(collective, "_get_flashinfer_allreduce", return_value=module):
            result = collective.all_reduce(tensor, collective.Group.TP, inplace=True)
        self.assertIs(result, tensor)
        torch.testing.assert_close(tensor, torch.full_like(tensor, 2))
        fast_path.all_reduce_into.assert_called_once_with(tensor, tensor)
        fast_path.all_reduce.assert_not_called()

    def test_destroy_is_idempotent(self):
        workspace = MagicMock()
        communicator = _make_communicator(workspace)
        communicator.destroy()
        communicator.destroy()
        workspace.destroy.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
