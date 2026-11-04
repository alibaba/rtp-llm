"""Exercise production graph initialization from grad-enabled startup contexts."""
import concurrent.futures
import unittest
import os
from unittest.mock import patch

import torch

from rtp_llm.ops.compute_ops import PyModelOutputs
from rtp_llm.cpp.cuda_graph.tests.libtest_cuda_graph_runner import CudaGraphRunner


class RecordingModel:
    def __init__(self, fail=False):
        self.fail = fail
        self.observations = []
        self.captured_outputs = []
        self.captured_inputs = []

    def prepare_fmha_impl(self, inputs, *args):
        self.observations.append(("prepare", torch.is_grad_enabled(), False))
        return None

    def forward(self, inputs, attn=None):
        capturing = torch.cuda.is_current_stream_capturing()
        self.observations.append(("forward", torch.is_grad_enabled(), capturing))
        if self.fail and (self.fail != "capture" or capturing):
            raise RuntimeError("intentional warmup failure")
        x = inputs.input_hiddens
        # Exercise inference tensors without depending on optional stream optimizations.
        main, side = x + 1, x + 2
        output = PyModelOutputs()
        output.hidden_states = main + side
        if capturing:
            self.captured_outputs.append(output.hidden_states)
            self.captured_inputs.append(x)
        return output


class CudaGraphGradModeTest(unittest.TestCase):
    def test_switch_combinations_restore_baseline_grad_mode(self):
        keys = ("RTP_LLM_GLM5_EH_FUSION", "RTP_LLM_GLM5_CMP_MOE_DUAL_STREAM")
        for eh, moe in ((None, None), ("0", "0"), ("1", "0"), ("0", "1"), ("1", "1")):
            for caller_grad in (False, True):
                with self.subTest(eh=eh, moe=moe, caller_grad=caller_grad), patch.dict(os.environ):
                    for key, value in zip(keys, (eh, moe)):
                        os.environ.pop(key, None)
                        if value is not None:
                            os.environ[key] = value
                    model = RecordingModel()
                    runner = CudaGraphRunner()
                    with torch.set_grad_enabled(caller_grad):
                        runner.init_decode(model, 8, 64, 64, 64, [1])
                        self.assertEqual(torch.is_grad_enabled(), caller_grad)
                    expected = caller_grad if (eh, moe) == ("0", "0") else False
                    self.assertTrue(model.observations)
                    self.assertTrue(all(grad == expected for _, grad, _ in model.observations))
                    del runner

    def check_capture(self, enabled):
        model = RecordingModel()
        runner = CudaGraphRunner()
        with torch.set_grad_enabled(enabled):
            runner.init_decode(model, 8, 64, 64, 64, [1, 4])
            self.assertEqual(torch.is_grad_enabled(), enabled)
        torch.cuda.synchronize()
        self.assertTrue(model.observations)
        self.assertTrue(all(not grad for _, grad, _ in model.observations), model.observations)
        self.assertEqual(sum(c for name, _, c in model.observations if name == "forward"), 2)
        self.assertEqual(len(model.captured_outputs), 2)
        for output in model.captured_outputs:
            self.assertFalse(output.is_inference())
            torch.testing.assert_close(output, torch.full_like(output, 3), rtol=0, atol=0)
        # Persistent capture inputs must remain mutable outside inference_mode.
        with torch.enable_grad():
            for value in model.captured_inputs:
                self.assertFalse(value.is_inference())
                value.fill_(7)
        del runner

    def test_grad_enabled_caller(self):
        self.check_capture(True)

    def test_no_grad_caller(self):
        self.check_capture(False)

    def test_fresh_thread_does_not_inherit_python_no_grad(self):
        def worker():
            self.assertTrue(torch.is_grad_enabled())
            torch.cuda.set_device(0)
            self.check_capture(True)
        with torch.no_grad(), concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(worker).result()
            self.assertFalse(torch.is_grad_enabled())

    def test_grad_restored_after_warmup_exception(self):
        for enabled in (True, False):
            with self.subTest(enabled=enabled), torch.set_grad_enabled(enabled):
                model = RecordingModel(fail=True)
                runner = CudaGraphRunner()
                with self.assertRaisesRegex(RuntimeError, "intentional warmup failure"):
                    runner.init_decode(model, 8, 64, 64, 64, [1])
                self.assertEqual(torch.is_grad_enabled(), enabled)
                self.assertTrue(model.observations)
                self.assertTrue(all(not grad for _, grad, _ in model.observations))
                del runner

    def test_grad_restored_after_capture_exception(self):
        for enabled in (True, False):
            with self.subTest(enabled=enabled), torch.set_grad_enabled(enabled):
                model = RecordingModel(fail="capture")
                runner = CudaGraphRunner()
                with self.assertRaisesRegex(RuntimeError, "intentional warmup failure"):
                    runner.init_decode(model, 8, 64, 64, 64, [1])
                self.assertEqual(torch.is_grad_enabled(), enabled)
                self.assertTrue(any(c for _, _, c in model.observations))
                self.assertTrue(all(not grad for _, grad, _ in model.observations))
                del runner


if __name__ == "__main__":
    unittest.main()
