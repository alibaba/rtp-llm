import ctypes
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.distributed import collective_torch as ct
from rtp_llm.models_py.distributed import rocm_rccl as hr


class TestCollectiveTorchHipGraphUnit(unittest.TestCase):
    def setUp(self):
        self._orig_is_rocm_runtime = hr._is_rocm_runtime
        self._orig_rccl_comm = hr._rccl_comm
        self._orig_rccl_world_size = hr._rccl_world_size
        self._orig_rccl_lib = hr._rccl_lib
        self._orig_cache = dict(hr._hipgraph_allgather_outputs)
        hr._hipgraph_allgather_outputs.clear()

    def tearDown(self):
        hr._is_rocm_runtime = self._orig_is_rocm_runtime
        hr._rccl_comm = self._orig_rccl_comm
        hr._rccl_world_size = self._orig_rccl_world_size
        hr._rccl_lib = self._orig_rccl_lib
        hr._hipgraph_allgather_outputs.clear()
        hr._hipgraph_allgather_outputs.update(self._orig_cache)

    def test_should_use_hipgraph_capture_rccl(self):
        hr._is_rocm_runtime = True
        hr._rccl_comm = ctypes.c_void_p(123)
        with patch.object(hr, "_is_hipgraph_capture_active", return_value=True):
            self.assertTrue(hr.should_use_hipgraph_capture_rccl(True))
            self.assertFalse(hr.should_use_hipgraph_capture_rccl(False))

    def test_should_not_use_hipgraph_capture_rccl_without_comm(self):
        hr._is_rocm_runtime = True
        hr._rccl_comm = None
        with patch.object(hr, "_is_hipgraph_capture_active", return_value=True):
            self.assertFalse(hr.should_use_hipgraph_capture_rccl(True))

    def test_get_nccl_dtype_map_and_error(self):
        fp16_tensor = torch.zeros(1, dtype=torch.float16)
        self.assertEqual(
            hr._get_nccl_dtype(fp16_tensor), hr._NCCL_DTYPE_MAP[torch.float16]
        )

        with self.assertRaises(TypeError):
            hr._get_nccl_dtype(torch.zeros(1, dtype=torch.bool))

    def test_get_or_create_allgather_output_cache_reuse(self):
        hr._rccl_world_size = 3
        src = torch.zeros((2, 4), dtype=torch.float16)

        with patch.object(hr, "_is_hipgraph_capture_active", return_value=True):
            out1 = hr._get_or_create_allgather_output(src)
            out2 = hr._get_or_create_allgather_output(src)

        self.assertIs(out1, out2)
        self.assertEqual(tuple(out1.shape), (6, 4))
        self.assertEqual(out1.dtype, src.dtype)
        self.assertEqual(out1.device, src.device)
        self.assertEqual(len(hr._hipgraph_allgather_outputs), 1)

    def test_get_or_create_allgather_output_rejects_inactive_capture(self):
        hr._rccl_world_size = 2
        src = torch.zeros((2, 4), dtype=torch.float16)

        with patch.object(hr, "_is_hipgraph_capture_active", return_value=False):
            with self.assertRaises(RuntimeError):
                hr._get_or_create_allgather_output(src)

        self.assertEqual(len(hr._hipgraph_allgather_outputs), 0)

    def test_set_graph_capture_nccl_comm_clears_cache(self):
        hr._is_rocm_runtime = True
        hr._hipgraph_allgather_outputs[(tuple([2, 4]), torch.float16, "cpu", -1)] = (
            torch.zeros((2, 4), dtype=torch.float16)
        )
        fake_lib = object()

        with patch.object(hr, "_load_rccl", return_value=fake_lib), patch.object(
            hr, "_setup_rccl_api"
        ) as setup_api:
            hr.set_graph_capture_nccl_comm(456, 4, 0)

        setup_api.assert_called_once_with(fake_lib)
        self.assertIsInstance(hr._rccl_comm, ctypes.c_void_p)
        self.assertEqual(hr._rccl_comm.value, 456)
        self.assertEqual(hr._rccl_world_size, 4)
        self.assertEqual(len(hr._hipgraph_allgather_outputs), 0)

    def test_enter_mode_without_handle_keeps_existing_comm(self):
        hr._is_rocm_runtime = True
        hr._rccl_comm = ctypes.c_void_p(123)
        hr._rccl_world_size = 4

        hr.enter_graph_capture_mode(0, 0, 0)

        self.assertIsNotNone(hr._rccl_comm)
        self.assertEqual(hr._rccl_comm.value, 123)
        self.assertEqual(hr._rccl_world_size, 4)

    def test_set_graph_capture_nccl_comm_zero_handle_clears_state(self):
        hr._is_rocm_runtime = True
        hr._rccl_comm = ctypes.c_void_p(123)
        hr._rccl_world_size = 4
        hr._hipgraph_allgather_outputs[(tuple([2, 4]), torch.float16, "cpu", -1)] = (
            torch.zeros((2, 4), dtype=torch.float16)
        )

        hr.set_graph_capture_nccl_comm(0, 0, 0)

        self.assertIsNone(hr._rccl_comm)
        self.assertEqual(hr._rccl_world_size, 1)
        self.assertEqual(len(hr._hipgraph_allgather_outputs), 0)

    def test_prepare_hipgraph_capture_rccl_comm_bootstraps_on_tp(self):
        hr._is_rocm_runtime = True
        parallelism_config = SimpleNamespace(tp_size=2, world_size=2)
        tp_group = object()
        with patch.object(
            hr, "bootstrap_hipgraph_capture_rccl_comm_from_tp_group"
        ) as bootstrap:
            hr.prepare_hipgraph_capture_rccl_comm_if_needed(
                parallelism_config, tp_group
            )
        bootstrap.assert_called_once_with(tp_group)

    def test_all_gather_fail_fast_when_capture_has_no_rccl_comm(self):
        hr._is_rocm_runtime = True
        hr._rccl_comm = None
        tensor = torch.zeros((1, 2), dtype=torch.float16)
        with patch.object(hr, "_is_hipgraph_capture_active", return_value=True):
            with self.assertRaises(RuntimeError):
                ct.all_gather(tensor, ct.Group.TP)

    # ------------------------------------------------------------------
    # TP>1 capture/replay path: all_reduce and all_gather via mocked RCCL
    # ------------------------------------------------------------------

    def _make_fake_lib(self, all_reduce_ret=0, all_gather_ret=0):
        """Build a mock RCCL ctypes library that records calls."""
        import unittest.mock as mock

        fake_lib = mock.MagicMock()
        fake_lib.ncclAllReduce.return_value = all_reduce_ret
        fake_lib.ncclAllGather.return_value = all_gather_ret
        return fake_lib

    def test_capture_all_reduce_calls_rccl_and_succeeds(self):
        """hipgraph_capture_all_reduce calls ncclAllReduce with correct args."""
        hr._is_rocm_runtime = True
        hr._rccl_comm = ctypes.c_void_p(999)
        hr._rccl_world_size = 2
        fake_lib = self._make_fake_lib(all_reduce_ret=0)
        hr._rccl_lib = fake_lib

        tensor = torch.zeros((4,), dtype=torch.bfloat16)
        with patch("torch.cuda.current_stream") as mock_stream:
            mock_stream.return_value.cuda_stream = 0xDEAD
            hr.hipgraph_capture_all_reduce(tensor)

        fake_lib.ncclAllReduce.assert_called_once()
        args = fake_lib.ncclAllReduce.call_args[0]
        # sendbuff == recvbuff (in-place), count == tensor.numel()
        self.assertEqual(args[0], tensor.data_ptr())
        self.assertEqual(args[1], tensor.data_ptr())
        self.assertEqual(args[2], tensor.numel())
        self.assertEqual(args[3], hr._NCCL_DTYPE_MAP[torch.bfloat16])  # ncclBfloat16
        self.assertEqual(args[4], hr._NCCL_SUM)

    def test_capture_all_reduce_raises_on_rccl_error(self):
        """hipgraph_capture_all_reduce raises RuntimeError on non-zero RCCL return."""
        hr._is_rocm_runtime = True
        hr._rccl_comm = ctypes.c_void_p(999)
        hr._rccl_world_size = 2
        hr._rccl_lib = self._make_fake_lib(all_reduce_ret=5)

        tensor = torch.zeros((4,), dtype=torch.float16)
        with patch("torch.cuda.current_stream") as mock_stream:
            mock_stream.return_value.cuda_stream = 0
            with self.assertRaises(RuntimeError):
                hr.hipgraph_capture_all_reduce(tensor)

    def test_capture_all_gather_returns_correct_shape_and_calls_rccl(self):
        """hipgraph_capture_all_gather allocates output [world_size * N, ...] and calls ncclAllGather."""
        hr._is_rocm_runtime = True
        hr._rccl_comm = ctypes.c_void_p(999)
        hr._rccl_world_size = 4
        fake_lib = self._make_fake_lib(all_gather_ret=0)
        hr._rccl_lib = fake_lib

        tensor = torch.zeros((3, 8), dtype=torch.float16)
        with patch.object(hr, "_is_hipgraph_capture_active", return_value=True), patch(
            "torch.cuda.current_stream"
        ) as mock_stream:
            mock_stream.return_value.cuda_stream = 0
            out = hr.hipgraph_capture_all_gather(tensor)

        self.assertEqual(tuple(out.shape), (12, 8))  # world_size=4 * 3
        self.assertEqual(out.dtype, tensor.dtype)
        fake_lib.ncclAllGather.assert_called_once()

    def test_capture_all_gather_raises_on_rccl_error(self):
        """hipgraph_capture_all_gather raises RuntimeError on non-zero RCCL return."""
        hr._is_rocm_runtime = True
        hr._rccl_comm = ctypes.c_void_p(999)
        hr._rccl_world_size = 2
        hr._rccl_lib = self._make_fake_lib(all_gather_ret=7)

        tensor = torch.zeros((2,), dtype=torch.float16)
        with patch.object(hr, "_is_hipgraph_capture_active", return_value=True), patch(
            "torch.cuda.current_stream"
        ) as mock_stream:
            mock_stream.return_value.cuda_stream = 0
            with self.assertRaises(RuntimeError):
                hr.hipgraph_capture_all_gather(tensor)

    def test_all_reduce_dispatches_to_rccl_during_capture(self):
        """collective_torch.all_reduce routes through RCCL when capture is active."""
        hr._is_rocm_runtime = True
        hr._rccl_comm = ctypes.c_void_p(123)
        hr._rccl_world_size = 2
        fake_lib = self._make_fake_lib(all_reduce_ret=0)
        hr._rccl_lib = fake_lib

        tensor = torch.zeros((4,), dtype=torch.float16)
        with patch.object(hr, "_is_hipgraph_capture_active", return_value=True), patch(
            "torch.cuda.current_stream"
        ) as mock_stream:
            mock_stream.return_value.cuda_stream = 0
            result = ct.all_reduce(tensor, ct.Group.TP)

        self.assertIs(result, tensor)  # in-place: same object returned
        fake_lib.ncclAllReduce.assert_called_once()

    def test_all_gather_dispatches_to_rccl_during_capture(self):
        """collective_torch.all_gather routes through RCCL when capture is active."""
        hr._is_rocm_runtime = True
        hr._rccl_comm = ctypes.c_void_p(123)
        hr._rccl_world_size = 3
        fake_lib = self._make_fake_lib(all_gather_ret=0)
        hr._rccl_lib = fake_lib

        tensor = torch.zeros((2, 4), dtype=torch.bfloat16)
        with patch.object(hr, "_is_hipgraph_capture_active", return_value=True), patch(
            "torch.cuda.current_stream"
        ) as mock_stream:
            mock_stream.return_value.cuda_stream = 0
            out = ct.all_gather(tensor, ct.Group.TP)

        self.assertEqual(tuple(out.shape), (6, 4))
        fake_lib.ncclAllGather.assert_called_once()

    def test_enter_capture_mode_with_valid_handle_updates_comm(self):
        """enter_graph_capture_mode with a valid handle registers the new comm."""
        hr._is_rocm_runtime = True
        hr._rccl_comm = None
        hr._rccl_world_size = 1
        fake_lib = self._make_fake_lib()

        with patch.object(hr, "_load_rccl", return_value=fake_lib), patch.object(
            hr, "_setup_rccl_api"
        ):
            hr.enter_graph_capture_mode(nccl_comm_handle=777, world_size=2, rank=0)

        self.assertIsNotNone(hr._rccl_comm)
        self.assertEqual(hr._rccl_comm.value, 777)
        self.assertEqual(hr._rccl_world_size, 2)

    def test_exit_capture_mode_preserves_rccl_comm(self):
        """exit_graph_capture_mode must NOT clear _rccl_comm (comm reused across captures)."""
        hr._is_rocm_runtime = True
        hr._rccl_comm = ctypes.c_void_p(456)
        hr._rccl_world_size = 2

        hr.exit_graph_capture_mode()

        # Comm must be preserved for the next capture round.
        self.assertIsNotNone(hr._rccl_comm)
        self.assertEqual(hr._rccl_comm.value, 456)
        self.assertEqual(hr._rccl_world_size, 2)

    def test_non_tp_group_does_not_use_rccl_during_capture(self):
        """all_reduce on a non-TP group must NOT route through RCCL even during capture."""
        hr._is_rocm_runtime = True
        hr._rccl_comm = ctypes.c_void_p(123)
        hr._rccl_world_size = 2
        fake_lib = self._make_fake_lib(all_reduce_ret=0)
        hr._rccl_lib = fake_lib

        tensor = torch.zeros((4,), dtype=torch.float16)
        with patch.object(hr, "_is_hipgraph_capture_active", return_value=True), patch(
            "rtp_llm.models_py.distributed.collective_torch._group_map", {}
        ):
            # DP group: should_use_capture_collectives(is_tp_group=False) -> False
            # all_reduce will try to use the regular torch.distributed path and
            # raise because torch.distributed is not initialised here.
            try:
                ct.all_reduce(tensor, ct.Group.DP)
            except Exception:
                pass  # expected — torch.distributed not initialised

        # RCCL ncclAllReduce must NOT have been called
        fake_lib.ncclAllReduce.assert_not_called()

    def test_destroy_capture_comm_clears_state_for_reinit(self):
        """destroy_capture_comm clears _rccl_comm so bootstrap can re-create on re-init."""
        hr._is_rocm_runtime = True
        hr._rccl_comm = ctypes.c_void_p(999)
        hr._rccl_world_size = 4
        hr._hipgraph_allgather_outputs[((6, 4), torch.float16, "cpu", -1)] = (
            torch.zeros((6, 4), dtype=torch.float16)
        )

        hr.destroy_capture_comm()

        self.assertIsNone(hr._rccl_comm)
        self.assertEqual(hr._rccl_world_size, 1)
        self.assertEqual(len(hr._hipgraph_allgather_outputs), 0)

    def test_bootstrap_creates_new_comm_after_destroy(self):
        """After destroy_capture_comm, bootstrap should attempt to create a new comm."""
        hr._is_rocm_runtime = True
        hr._rccl_comm = None  # cleared by prior destroy
        hr._rccl_world_size = 1

        with patch.object(
            hr, "bootstrap_hipgraph_capture_rccl_comm_from_tp_group"
        ) as bootstrap:
            parallelism_config = SimpleNamespace(tp_size=2, world_size=2)
            tp_group = object()
            hr.prepare_hipgraph_capture_rccl_comm_if_needed(
                parallelism_config, tp_group
            )

        # bootstrap must be called (not skipped) because _rccl_comm is None
        bootstrap.assert_called_once_with(tp_group)


if __name__ == "__main__":
    unittest.main()
