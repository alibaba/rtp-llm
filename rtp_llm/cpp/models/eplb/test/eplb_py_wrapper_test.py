import unittest
from unittest.mock import patch

import torch
from eplb_runtime_test_utils import FakeModel, FakeRuntime, fake_runtime_module

from rtp_llm.cpp.models.eplb.test.libth_eplb_py_wrapper_test import EplbPyWrapperOP


class TestEplbPyWrapper(unittest.TestCase):

    def setUp(self):
        runtime_module = fake_runtime_module()
        runtime_module.start()
        self.addCleanup(runtime_module.stop)
        self.model = FakeModel()
        self.wrapper = EplbPyWrapperOP()
        self.addCleanup(self.wrapper.destroy)
        # All bridge entry points release the GIL in the test DSO.
        self.wrapper.initialize(
            self.model,
            ep_rank=1,
            ep_size=2,
            model_id=1,
            is_normal_engine=True,
            sync_prompt_phase=True,
        )

    def test_constructor_passes_explicit_layers_options_and_identity(self):
        self.assertEqual(self.model.state.calls, ["get_layers"])
        self.assertEqual(self.model.state.identity, (1, 2, 1))
        self.assertTrue(self.model.state.is_normal_engine)
        self.assertTrue(self.model.state.sync_prompt_phase)
        self.assertEqual(
            self.model.state.options,
            {
                "eplb_mode": 3,
                "eplb_update_time": 7,
                "eplb_stats_window_size": 3,
                "eplb_rearrange_after_prefill": True,
                "eplb_rearrange_p2p_groups": 2,
            },
        )

    def test_tensor_is_forwarded_with_original_storage_and_stride(self):
        counts = torch.arange(12, dtype=torch.int32).reshape(3, 4).t()
        self.wrapper.step(counts, is_prompt=True)
        self.assertEqual(self.model.state.prompts, [True])
        runtime = self.model.state.runtime()
        self.assertEqual(runtime.counts.data_ptr(), counts.data_ptr())
        self.assertEqual(runtime.counts.stride(), counts.stride())
        torch.testing.assert_close(runtime.counts, counts)

    def test_balance_result_and_config_update_are_forwarded(self):
        self.model.state.apply_on_calls = {2, 3}
        counts = torch.tensor([[2, 3]], dtype=torch.int32)
        self.assertFalse(self.wrapper.step(counts))
        self.assertTrue(self.wrapper.step(counts))
        self.wrapper.update_config(mode=0, update_time=13)
        self.assertFalse(self.wrapper.step(counts))
        self.assertIn(("update", 0, 13), self.model.state.calls)
        self.assertEqual(self.model.state.applied, 1)

    def test_python_failure_propagates_to_caller(self):
        with patch.object(FakeRuntime, "step", side_effect=RuntimeError("step failed")):
            with self.assertRaisesRegex(RuntimeError, "step failed"):
                self.wrapper.step(torch.zeros((1, 2), dtype=torch.int32))

    def test_runtime_is_released_with_gil_when_destroyed(self):
        reference = self.model.state.runtime
        self.assertIsNotNone(reference())
        self.wrapper.destroy()
        self.assertIsNone(reference())
        self.assertTrue(self.model.state.destroyed)


if __name__ == "__main__":
    unittest.main()
