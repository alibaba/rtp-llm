# SPDX-License-Identifier: Apache-2.0

import os
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
            with self.subTest(value=value), patch.dict(
                os.environ, {"FT_DISABLE_CUSTOM_AR": value}, clear=True
            ):
                self.assertTrue(flashinfer_ar.enabled_by_env())
        with patch.dict(os.environ, {"FT_DISABLE_CUSTOM_AR": "1"}, clear=True):
            self.assertFalse(flashinfer_ar.enabled_by_env())

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

    def test_collective_dispatches_only_tp_to_flashinfer(self):
        tensor = torch.ones((2, 4))
        fast_path = MagicMock()
        fast_path.should_use.return_value = True
        reduced = torch.full_like(tensor, 2)
        fast_path.all_reduce.return_value = reduced
        module = SimpleNamespace(get_flashinfer_allreduce=lambda: fast_path)

        with patch.object(collective, "_get_flashinfer_allreduce", return_value=module):
            self.assertIs(collective.all_reduce(tensor, collective.Group.TP), reduced)
        fast_path.should_use.assert_called_once_with(tensor)
        fast_path.all_reduce.assert_called_once_with(tensor)

    def test_collective_preserves_inplace_contract(self):
        tensor = torch.ones((2, 4))
        fast_path = MagicMock()
        fast_path.should_use.return_value = True
        fast_path.all_reduce.return_value = torch.full_like(tensor, 2)
        module = SimpleNamespace(get_flashinfer_allreduce=lambda: fast_path)

        with patch.object(collective, "_get_flashinfer_allreduce", return_value=module):
            result = collective.all_reduce(tensor, collective.Group.TP, inplace=True)
        self.assertIs(result, tensor)
        torch.testing.assert_close(tensor, torch.full_like(tensor, 2))

    def test_destroy_is_idempotent(self):
        workspace = MagicMock()
        communicator = _make_communicator(workspace)
        communicator.destroy()
        communicator.destroy()
        workspace.destroy.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
