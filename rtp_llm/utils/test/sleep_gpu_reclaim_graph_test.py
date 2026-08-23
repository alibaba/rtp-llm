import os
import unittest
from unittest import mock


class TestCudaGraphSleepReclaim(unittest.TestCase):
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

    def test_graph_guard_keeps_allocator_empty_cache_from_python(self):
        from rtp_llm.utils import sleep_gpu_reclaim

        self.state.mark_cuda_graph_baked(True)
        with mock.patch.object(sleep_gpu_reclaim.torch.cuda, "empty_cache") as empty:
            # Avoid CUDA device setup and snapshot work; this verifies the
            # graph decision at the only destructive Python allocator call.
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
            empty.assert_not_called()


if __name__ == "__main__":
    unittest.main()
