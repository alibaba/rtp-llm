import os
import unittest
from types import SimpleNamespace
from unittest import mock


class TestCudaGraphSleepReclaim(unittest.TestCase):

    def test_sleep_preserves_tp_communicator_and_disabled_topology(self):
        from rtp_llm.models_py.distributed import symm_mem
        from rtp_llm.models_py.modules.dsv4.moe import mega_buf
        from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_deepgemm_linear import (
            CudaFp8DeepGEMMLinear,
        )
        from rtp_llm.utils import sleep_gpu_reclaim as reclaim

        buffer = SimpleNamespace(numel=lambda: 128, element_size=lambda: 2)
        comm = SimpleNamespace(buffer=buffer)
        for original_comm in (None, comm):
            for graph_baked in (False, True):
                with self.subTest(
                    had_comm=original_comm is not None, graph_baked=graph_baked
                ), mock.patch.dict(
                    os.environ, {"RTP_LLM_SLEEP_FREE_RUNTIME_CACHES": "1"}
                ), mock.patch.object(
                    symm_mem, "_symm_mem_comm", original_comm
                ), mock.patch.object(
                    symm_mem, "init_symm_mem_communicator"
                ) as init, mock.patch.object(
                    reclaim, "_cuda_graph_baked", return_value=graph_baked
                ), mock.patch.object(
                    reclaim.torch._C, "_cuda_clearCublasWorkspaces", create=True
                ), mock.patch.object(
                    CudaFp8DeepGEMMLinear,
                    "release_runtime_caches_for_sleep",
                    return_value=0,
                ), mock.patch.object(
                    mega_buf, "mega_buffers_graph_baked", return_value=graph_baked
                ), mock.patch.object(
                    mega_buf, "release_mega_symm_buffers", return_value=0
                ):
                    for _ in range(2):
                        notes = reclaim._clear_module_device_caches()
                        self.assertIs(symm_mem._symm_mem_comm, original_comm)
                        self.assertIs(comm.buffer, buffer)
                        self.assertIn(
                            "TP symmetric-memory communicator KEPT (communication topology)",
                            notes,
                        )
                    init.assert_not_called()

    def test_destructive_release_failure_reaches_sleep_hook(self):
        from rtp_llm.utils import sleep_gpu_reclaim as reclaim

        with mock.patch.dict(
            os.environ, {"RTP_LLM_SLEEP_FREE_RUNTIME_CACHES": "1"}, clear=True
        ), mock.patch.object(
            reclaim,
            "_clear_module_device_caches",
            side_effect=reclaim.RuntimeCacheReleaseError(
                "Mega symmetric-memory release failed"
            ),
        ), mock.patch.object(
            reclaim.torch.cuda, "device"
        ), mock.patch.object(
            reclaim.torch.cuda, "mem_get_info", return_value=(100, 200)
        ), mock.patch.object(
            reclaim, "_snapshot_summary", return_value=""
        ), mock.patch.object(
            reclaim.torch.cuda, "empty_cache"
        ) as trim:
            with self.assertRaisesRegex(
                reclaim.RuntimeCacheReleaseError, "Mega symmetric-memory"
            ):
                reclaim.release_and_trim("cuda:0")
            trim.assert_not_called()

    def test_init_memory_probe_is_opt_in(self):
        from rtp_llm.utils import gpu_mem_probe

        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(
            gpu_mem_probe.torch, "cuda"
        ) as cuda:
            gpu_mem_probe.log_gpu_mem("start/baseline")
            self.assertEqual(cuda.mock_calls, [])

        with mock.patch.dict(
            os.environ, {"RTP_LLM_RECORD_MEM_HISTORY": "1"}, clear=True
        ), mock.patch.object(gpu_mem_probe.torch, "cuda") as cuda, mock.patch.object(
            gpu_mem_probe, "_maybe_enable_mem_history"
        ) as history:
            cuda.is_available.return_value = True
            cuda.mem_get_info.return_value = (100, 200)
            cuda.memory_reserved.return_value = 10
            cuda.memory_allocated.return_value = 5
            gpu_mem_probe.log_gpu_mem("start/baseline", device=1)
            history.assert_called_once_with()
            cuda.mem_get_info.assert_called_once_with(1)

    def setUp(self):
        from rtp_llm.models_py.utils import cuda_graph_state

        self.state = cuda_graph_state
        self.saved_graph_state = cuda_graph_state._GRAPH_BAKED
        cuda_graph_state._GRAPH_BAKED = False

    def tearDown(self):
        self.state._GRAPH_BAKED = self.saved_graph_state

    def test_graph_state_is_sticky(self):
        self.state.mark_cuda_graph_baked(False)
        self.assertFalse(self.state.cuda_graph_baked())
        self.state.mark_cuda_graph_baked(True)
        self.assertTrue(self.state.cuda_graph_baked())
        # A non-graph model loaded later must not disable protection for the
        # already captured graph.
        self.state.mark_cuda_graph_baked(False)
        self.assertTrue(self.state.cuda_graph_baked())

    def test_optional_release_requires_env_and_no_graph(self):
        from rtp_llm.utils.sleep_gpu_reclaim import _optional_release_allowed

        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(_optional_release_allowed(False))
        with mock.patch.dict(
            os.environ, {"RTP_LLM_SLEEP_FREE_RUNTIME_CACHES": "1"}, clear=True
        ):
            self.assertTrue(_optional_release_allowed(False))
            self.assertFalse(_optional_release_allowed(True))

    def test_legacy_mega_switch_is_accepted_as_compatibility_alias(self):
        from rtp_llm.models_py.utils.cuda_graph_state import (
            runtime_cache_release_enabled,
        )

        with mock.patch.dict(
            os.environ, {"RTP_LLM_SLEEP_FREE_MEGA_SYMM": "1"}, clear=True
        ):
            self.assertTrue(runtime_cache_release_enabled())

    def test_graph_role_still_trims_free_allocator_segments(self):
        from rtp_llm.utils import sleep_gpu_reclaim

        self.state.mark_cuda_graph_baked(True)
        with mock.patch.object(sleep_gpu_reclaim.torch.cuda, "empty_cache") as empty:
            # Avoid CUDA device setup and snapshot work; graph-baked pointers
            # stay live, but ordinary fully-free segments must still be handed
            # back to the driver without recapture.
            with mock.patch.object(sleep_gpu_reclaim.torch.cuda, "device"):
                with mock.patch.object(
                    sleep_gpu_reclaim.torch.cuda,
                    "mem_get_info",
                    side_effect=[(100, 200), (150, 200)],
                ):
                    with mock.patch.object(
                        sleep_gpu_reclaim, "_snapshot_summary", return_value=""
                    ), mock.patch.object(
                        sleep_gpu_reclaim,
                        "_clear_module_device_caches",
                        return_value=[],
                    ), mock.patch.object(
                        sleep_gpu_reclaim.torch.cuda, "synchronize"
                    ):
                        sleep_gpu_reclaim.release_and_trim("cuda:0")
            empty.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
