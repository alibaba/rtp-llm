import unittest

import torch

from rtp_llm.cpp.cuda_graph.tests.libtest_cuda_graph_runner import CudaGraphRunner
from rtp_llm.ops.compute_ops import PyModelOutputs


class _GraphFlagModel:
    def __init__(self) -> None:
        self.prepare_calls = 0
        self.forward_calls = 0

    def prepare_fmha_impl(self, inputs, is_cuda_graph=False):
        if not is_cuda_graph:
            raise AssertionError("CUDA Graph runner did not request graph attention")
        if not inputs.attention_inputs.is_cuda_graph:
            raise AssertionError("prepare_fmha_impl input lost CUDA Graph mode")
        self.prepare_calls += 1
        return None

    def forward(self, inputs, fmha_impl=None):
        del fmha_impl
        if not inputs.attention_inputs.is_cuda_graph:
            raise AssertionError("CUDA Graph warmup/capture input lost graph mode")
        self.forward_calls += 1
        return PyModelOutputs(inputs.input_hiddens + 1)


class CudaGraphModeFlagTest(unittest.TestCase):
    def test_decode_warmup_and_capture_keep_graph_mode(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        model = _GraphFlagModel()
        runner = CudaGraphRunner()
        runner.init_decode(
            model,
            hidden_size=8,
            max_seq_len=64,
            tokens_per_block=64,
            kernel_tokens_per_block=64,
            decode_capture_batch_sizes=[1],
        )
        torch.cuda.synchronize()
        self.assertEqual(model.prepare_calls, 2)
        self.assertGreaterEqual(model.forward_calls, 4)


if __name__ == "__main__":
    unittest.main()
