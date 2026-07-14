import ctypes
import sys
import unittest
import unittest.mock
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
        self._orig_ar_config = getattr(hr, "_rocm_ar_config", None)
        self._orig_metadata_group_map = getattr(ct, "_metadata_group_map", {}).copy()
        hr._hipgraph_allgather_outputs.clear()

    def tearDown(self):
        hr._is_rocm_runtime = self._orig_is_rocm_runtime
        hr._rccl_comm = self._orig_rccl_comm
        hr._rccl_world_size = self._orig_rccl_world_size
        hr._rccl_lib = self._orig_rccl_lib
        hr._hipgraph_allgather_outputs.clear()
        hr._hipgraph_allgather_outputs.update(self._orig_cache)
        if self._orig_ar_config is not None:
            hr._rocm_ar_config = self._orig_ar_config
        if hasattr(ct, "_metadata_group_map"):
            ct._metadata_group_map.clear()
            ct._metadata_group_map.update(self._orig_metadata_group_map)

    def test_should_use_hipgraph_capture_rccl(self):
        hr._is_rocm_runtime = True
        hr._rccl_comm = ctypes.c_void_p(123)
        with patch.object(hr, "_is_hipgraph_capture_active", return_value=True):
            self.assertTrue(hr.should_use_hipgraph_capture_rccl(True))
            self.assertFalse(hr.should_use_hipgraph_capture_rccl(False))

    def test_should_use_hipgraph_capture_rccl_even_without_comm(self):
        """Comm check is deferred to _get_rccl_runtime; should_use only checks runtime+capture+tp."""
        hr._is_rocm_runtime = True
        hr._rccl_comm = None
        with patch.object(hr, "_is_hipgraph_capture_active", return_value=True):
            self.assertTrue(hr.should_use_hipgraph_capture_rccl(True))

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

    def test_get_or_create_allgather_output_returns_temp_when_inactive(self):
        """Non-capture mode returns a fresh temporary tensor without caching."""
        hr._rccl_world_size = 2
        src = torch.zeros((2, 4), dtype=torch.float16)

        with patch.object(hr, "_is_hipgraph_capture_active", return_value=False):
            out = hr._get_or_create_allgather_output(src)

        self.assertEqual(tuple(out.shape), (4, 4))
        self.assertEqual(out.dtype, src.dtype)
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

    def test_configure_rocm_custom_ar_from_hw_config(self):
        cfg = SimpleNamespace(
            enable_rocm_vllm_custom_ar=True,
            enable_rocm_quick_reduce=True,
            rocm_quick_reduce_quantization="int6",
        )

        hr.configure_custom_ar_from_hw_config(cfg)

        self.assertTrue(hr._rocm_ar_config.enable_vllm_custom_ar)
        self.assertTrue(hr._rocm_ar_config.enable_quick_reduce)
        self.assertEqual(hr._rocm_ar_config.quick_reduce_quantization, "INT6")

    def test_prepare_comm_receives_metadata_group(self):
        hr._is_rocm_runtime = True
        parallelism_config = SimpleNamespace(tp_size=2, world_size=2)
        tp_group = object()
        tp_metadata_group = object()

        with patch.object(
            hr, "bootstrap_hipgraph_capture_rccl_comm_from_tp_group"
        ) as bootstrap, patch.object(hr, "_init_optional_ar_backends") as init_optional:
            hr.prepare_hipgraph_capture_rccl_comm_if_needed(
                parallelism_config,
                tp_group,
                tp_metadata_group=tp_metadata_group,
            )

        bootstrap.assert_called_once_with(tp_group)
        init_optional.assert_called_once_with(parallelism_config, tp_metadata_group)

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

    def test_capture_all_reduce_uses_quick_reduce_before_custom_and_trt(self):
        hr._is_rocm_runtime = True
        hr._rccl_comm = ctypes.c_void_p(999)
        hr._rccl_world_size = 2
        hr._rccl_lib = self._make_fake_lib(all_reduce_ret=0)
        hr._rocm_ar_config = hr.RocmAllReduceConfig(
            enable_vllm_custom_ar=True,
            enable_quick_reduce=True,
            quick_reduce_quantization="INT8",
        )
        tensor = torch.zeros((8,), dtype=torch.float16)
        qr_out = torch.ones_like(tensor)

        with patch.object(
            hr, "_try_quick_reduce_all_reduce", return_value=qr_out
        ) as qr, patch.object(
            hr, "_try_vllm_custom_all_reduce", return_value=None
        ) as custom, patch.object(
            hr, "_try_trt_all_reduce", return_value=None
        ) as trt:
            result = hr.hipgraph_capture_all_reduce(tensor, process_group=object())

        self.assertIs(result, qr_out)
        qr.assert_called_once()
        custom.assert_not_called()
        trt.assert_not_called()

    def test_non_capture_all_reduce_uses_quick_reduce_when_enabled(self):
        hr._is_rocm_runtime = True
        hr._rocm_ar_config = hr.RocmAllReduceConfig(
            enable_vllm_custom_ar=False,
            enable_quick_reduce=True,
            quick_reduce_quantization="INT8",
        )
        tensor = torch.zeros((8,), dtype=torch.float16)
        process_group = object()
        qr_out = torch.ones_like(tensor)

        with patch.object(
            hr, "_is_hipgraph_capture_active", return_value=False
        ), patch.object(hr, "_try_quick_reduce_all_reduce", return_value=qr_out) as qr:
            result = hr.try_non_capture_all_reduce(tensor, process_group)

        self.assertIs(result, qr_out)
        qr.assert_called_once_with(tensor, process_group)

    def test_capture_all_reduce_uses_custom_when_quick_reduce_disabled(self):
        hr._is_rocm_runtime = True
        hr._rccl_comm = ctypes.c_void_p(999)
        hr._rccl_world_size = 2
        hr._rccl_lib = self._make_fake_lib(all_reduce_ret=0)
        hr._rocm_ar_config = hr.RocmAllReduceConfig(
            enable_vllm_custom_ar=True,
            enable_quick_reduce=False,
            quick_reduce_quantization="FP",
        )
        tensor = torch.zeros((8,), dtype=torch.float16)
        custom_out = torch.ones_like(tensor)

        with patch.object(
            hr, "_try_quick_reduce_all_reduce", return_value=None
        ) as qr, patch.object(
            hr, "_try_vllm_custom_all_reduce", return_value=custom_out
        ) as custom, patch.object(
            hr, "_try_trt_all_reduce", return_value=None
        ) as trt:
            result = hr.hipgraph_capture_all_reduce(tensor, process_group=object())

        self.assertIs(result, custom_out)
        qr.assert_not_called()
        custom.assert_called_once()
        trt.assert_not_called()

    def test_capture_all_reduce_falls_back_to_trt_then_rccl(self):
        hr._is_rocm_runtime = True
        hr._rccl_comm = ctypes.c_void_p(999)
        hr._rccl_world_size = 2
        fake_lib = self._make_fake_lib(all_reduce_ret=0)
        hr._rccl_lib = fake_lib
        hr._rocm_ar_config = hr.RocmAllReduceConfig(
            enable_vllm_custom_ar=False,
            enable_quick_reduce=False,
            quick_reduce_quantization="FP",
        )
        tensor = torch.zeros((8,), dtype=torch.float16)

        with patch.object(
            hr, "_try_quick_reduce_all_reduce", return_value=None
        ) as qr, patch.object(
            hr, "_try_vllm_custom_all_reduce", return_value=None
        ) as custom, patch.object(
            hr, "_try_trt_all_reduce", return_value=None
        ) as trt, patch(
            "torch.cuda.current_stream"
        ) as mock_stream:
            mock_stream.return_value.cuda_stream = 0
            result = hr.hipgraph_capture_all_reduce(tensor, process_group=object())

        self.assertIs(result, tensor)
        qr.assert_not_called()
        custom.assert_not_called()
        trt.assert_called_once()
        fake_lib.ncclAllReduce.assert_called_once()

    def test_trt_dist_env_retains_workspace_capacity_for_size_guard(self):
        from rtp_llm.models_py.modules.base.rocm import trt_allreduce as trt_ar

        with patch.object(trt_ar.dist, "get_rank", return_value=0), patch.object(
            trt_ar.dist, "get_world_size", return_value=1
        ), patch.object(torch.cuda, "set_device"):
            dist_env = trt_ar.TrtllmDistEnv(
                group=object(), device_id=0, max_size_in_bytes=12345
            )

        self.assertEqual(12345, dist_env.max_size_in_bytes)

    def test_try_trt_all_reduce_rejects_tensor_larger_than_workspace(self):
        from rtp_llm.models_py.modules.base.rocm import trt_allreduce as trt_ar

        tensor = torch.zeros((8, 2048), dtype=torch.bfloat16)
        process_group = object()
        dist_env = SimpleNamespace(max_size_in_bytes=1024)
        with patch.object(
            hr, "_is_hidden_size_supported_for_trtllm", return_value=True
        ), patch.object(
            hr, "_is_trtllm_allreduce_ready", return_value=True
        ), patch.object(
            trt_ar._trtllm_comm_manager, "dist_env", dist_env
        ), patch.object(
            trt_ar, "allreduce"
        ) as trt_allreduce:
            result = hr._try_trt_all_reduce(tensor, process_group)

        self.assertIsNone(result)
        trt_allreduce.assert_not_called()

    def test_try_trt_all_reduce_uses_backend_within_workspace(self):
        from rtp_llm.models_py.modules.base.rocm import trt_allreduce as trt_ar

        tensor = torch.zeros((8, 2048), dtype=torch.bfloat16)
        expected = torch.ones_like(tensor)
        process_group = object()
        dist_env = SimpleNamespace(
            max_size_in_bytes=tensor.numel() * tensor.element_size()
        )
        with patch.object(
            hr, "_is_hidden_size_supported_for_trtllm", return_value=True
        ), patch.object(
            hr, "_is_trtllm_allreduce_ready", return_value=True
        ), patch.object(
            trt_ar._trtllm_comm_manager, "dist_env", dist_env
        ), patch.object(
            trt_ar, "allreduce", return_value=expected
        ) as trt_allreduce, patch.object(
            torch.cuda, "current_device", return_value=3
        ):
            result = hr._try_trt_all_reduce(tensor, process_group)

        self.assertIs(result, expected)
        trt_allreduce.assert_called_once_with(
            allreduce_in=tensor, group=process_group, device_id=3
        )

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
        mock_group = object()
        with patch.object(hr, "_is_hipgraph_capture_active", return_value=True), patch(
            "torch.cuda.current_stream"
        ) as mock_stream, patch(
            "rtp_llm.models_py.distributed.collective_torch._get_group",
            return_value=mock_group,
        ):
            mock_stream.return_value.cuda_stream = 0
            result = ct.all_reduce(tensor, ct.Group.TP)

        self.assertIs(result, tensor)  # in-place: same object returned
        fake_lib.ncclAllReduce.assert_called_once()

    def test_all_reduce_dispatches_to_quick_reduce_without_capture(self):
        """TP all_reduce should use ROCm QuickReduce on the non-capture path."""
        hr._is_rocm_runtime = True
        hr._rocm_ar_config = hr.RocmAllReduceConfig(
            enable_vllm_custom_ar=False,
            enable_quick_reduce=True,
            quick_reduce_quantization="INT8",
        )

        tensor = torch.zeros((4,), dtype=torch.float16)
        quick_reduce_out = torch.ones_like(tensor)
        mock_group = object()
        with patch.object(
            hr, "try_non_capture_all_reduce", return_value=quick_reduce_out
        ) as quick_reduce, patch(
            "rtp_llm.models_py.distributed.collective_torch._get_group",
            return_value=mock_group,
        ), patch(
            "rtp_llm.models_py.distributed.collective_torch.get_symm_mem_communicator"
        ) as symm_mem, patch(
            "torch.distributed.all_reduce"
        ) as torch_all_reduce:
            result = ct.all_reduce(tensor, ct.Group.TP)

        self.assertIs(result, quick_reduce_out)
        quick_reduce.assert_called_once_with(tensor, mock_group)
        symm_mem.assert_not_called()
        torch_all_reduce.assert_not_called()

    def test_all_gather_dispatches_to_rccl_during_capture(self):
        """collective_torch.all_gather routes through RCCL when capture is active."""
        hr._is_rocm_runtime = True
        hr._rccl_comm = ctypes.c_void_p(123)
        hr._rccl_world_size = 3
        fake_lib = self._make_fake_lib(all_gather_ret=0)
        hr._rccl_lib = fake_lib

        tensor = torch.zeros((2, 4), dtype=torch.bfloat16)
        mock_group = object()
        with patch.object(hr, "_is_hipgraph_capture_active", return_value=True), patch(
            "torch.cuda.current_stream"
        ) as mock_stream, patch(
            "rtp_llm.models_py.distributed.collective_torch._get_group",
            return_value=mock_group,
        ):
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

    def test_finish_session_calls_consume_when_pending(self):
        """consume_capture() is called when has_pending_capture() returns True."""
        mock_consume = unittest.mock.MagicMock()
        mock_has_pending = unittest.mock.MagicMock(return_value=True)
        fake_module = SimpleNamespace(
            consume_capture=mock_consume,
            has_pending_capture=mock_has_pending,
        )
        with patch.dict(
            "sys.modules",
            {"rtp_llm.models_py.modules.base.rocm.trt_allreduce": fake_module},
        ):
            hr.finish_hipgraph_capture_session()

        mock_has_pending.assert_called_once()
        mock_consume.assert_called_once()

    def test_finish_session_skips_when_not_pending(self):
        """consume_capture() is NOT called when has_pending_capture() returns False."""
        mock_consume = unittest.mock.MagicMock()
        mock_has_pending = unittest.mock.MagicMock(return_value=False)
        fake_module = SimpleNamespace(
            consume_capture=mock_consume,
            has_pending_capture=mock_has_pending,
        )
        with patch.dict(
            "sys.modules",
            {"rtp_llm.models_py.modules.base.rocm.trt_allreduce": fake_module},
        ):
            hr.finish_hipgraph_capture_session()

        mock_has_pending.assert_called_once()
        mock_consume.assert_not_called()

    def test_finish_session_propagates_consume_error(self):
        """Runtime errors from consume_capture() propagate (not silenced)."""
        mock_consume = unittest.mock.MagicMock(
            side_effect=RuntimeError("barrier timeout")
        )
        mock_has_pending = unittest.mock.MagicMock(return_value=True)
        fake_module = SimpleNamespace(
            consume_capture=mock_consume,
            has_pending_capture=mock_has_pending,
        )
        with patch.dict(
            "sys.modules",
            {"rtp_llm.models_py.modules.base.rocm.trt_allreduce": fake_module},
        ):
            with self.assertRaises(RuntimeError):
                hr.finish_hipgraph_capture_session()

    def test_finish_session_tolerates_import_error(self):
        """ImportError (trt_allreduce unavailable) is silently handled."""
        with patch.dict(
            "sys.modules", {"rtp_llm.models_py.modules.base.rocm.trt_allreduce": None}
        ):
            hr.finish_hipgraph_capture_session()


if __name__ == "__main__":
    unittest.main()
