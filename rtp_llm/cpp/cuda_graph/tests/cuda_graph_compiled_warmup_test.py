import os
import unittest

import torch

from rtp_llm.cpp.cuda_graph.tests.libtest_cuda_graph_runner import CudaGraphRunner
from rtp_llm.ops.compute_ops import (
    PyModelInputs,
    PyModelOutputs,
    cuda_graph_warmup_forward_enabled,
)

WARMUP_ENV = "RTP_LLM_CUDA_GRAPH_WARMUP_FORWARD"
HIDDEN_SIZE = 4


@torch.compile(fullgraph=True, dynamic=False)
def _compiled_forward(hidden_states: torch.Tensor) -> torch.Tensor:
    return hidden_states + 1


class CompiledWarmupModel:
    def __init__(self) -> None:
        self.forward_phases: list[tuple[bool, bool]] = []
        self.warmup_env_values: list[str | None] = []

    def prepare_fmha_impl(self, inputs: PyModelInputs, is_cuda_graph: bool = False):
        return None

    def forward(self, inputs: PyModelInputs, fmha_impl=None) -> PyModelOutputs:
        warmup_flag = cuda_graph_warmup_forward_enabled()
        self.warmup_env_values.append(os.environ.get(WARMUP_ENV))
        is_capturing = torch.cuda.is_current_stream_capturing()
        self.forward_phases.append((warmup_flag, is_capturing))

        # Match the SM120 MoE dispatch contract: the compiled backend is used
        # during the explicit eager warmup and during graph capture.  If the
        # C++ thread-local flag is not read, compilation is deferred until
        # capture and TorchInductor raises BackendCompilerFailed.
        if warmup_flag or is_capturing:
            hidden_states = _compiled_forward(inputs.input_hiddens)
        else:
            hidden_states = inputs.input_hiddens + 1
        return PyModelOutputs(hidden_states)


class TestCudaGraphCompiledWarmup(unittest.TestCase):
    def test_compiled_backend_is_warmed_before_capture(self) -> None:
        os.environ.pop(WARMUP_ENV, None)
        torch._dynamo.reset()

        model = CompiledWarmupModel()
        runner = CudaGraphRunner()
        runner.init_decode(
            model,
            HIDDEN_SIZE,
            64,
            64,
            64,
            [1],
        )

        eager_phases = [
            flag for flag, capturing in model.forward_phases if not capturing
        ]
        capture_phases = [flag for flag, capturing in model.forward_phases if capturing]
        self.assertTrue(eager_phases)
        self.assertTrue(capture_phases)
        self.assertTrue(all(eager_phases))
        self.assertFalse(any(capture_phases))
        self.assertTrue(all(value is None for value in model.warmup_env_values))
        self.assertFalse(cuda_graph_warmup_forward_enabled())
        self.assertNotIn(WARMUP_ENV, os.environ)


if __name__ == "__main__":
    unittest.main()
