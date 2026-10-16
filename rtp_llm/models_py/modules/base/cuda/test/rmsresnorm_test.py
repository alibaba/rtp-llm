import itertools
import statistics
import time
from unittest import SkipTest, TestCase, main, mock

import torch
from rtp_llm.models_py.modules import RMSResNorm, RMSResNormTorch
from rtp_llm.ops.compute_ops import rtp_llm_ops
from torch import dtype as _dtype


class RMSResNormTest(TestCase):
    DTYPES = [torch.half, torch.bfloat16]
    NUM_TOKENS = [7, 83, 4096]
    HIDDEN_SIZES = [768, 769, 770, 771, 5120, 5124, 5125, 5126, 8192, 8199]

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def _run_rms_res_norm_test(self, num_tokens: int, hidden_size: int, dtype: _dtype):
        torch.manual_seed(0)
        w = torch.randn(hidden_size, dtype=dtype)
        rms_res_norm = RMSResNorm(w)
        rms_res_norm_torch = RMSResNormTorch(w)
        x = torch.randn(num_tokens, hidden_size, dtype=dtype)
        residual = torch.randn(num_tokens, hidden_size, dtype=dtype)
        expected_residual = x + residual
        expected_hidden = rms_res_norm_torch(x, residual)
        actual_hidden, actual_residual = rms_res_norm(x, residual)
        torch.testing.assert_close(actual_hidden, expected_hidden, atol=3e-2, rtol=3e-2)
        torch.testing.assert_close(
            actual_residual, expected_residual, atol=3e-2, rtol=3e-2
        )

    def test_rms_res_norm(self):
        for params in itertools.product(
            self.NUM_TOKENS,
            self.HIDDEN_SIZES,
            self.DTYPES,
        ):
            with self.subTest(
                num_tokens=params[0], hidden_size=params[1], dtype=params[2]
            ):
                self._run_rms_res_norm_test(*params)

    def test_current_stream_is_resolved_inside_binding(self):
        width = 4096
        weight = torch.randn(width, dtype=torch.bfloat16)
        hidden_states = torch.randn((97, width), dtype=torch.bfloat16)
        residual = torch.randn_like(hidden_states)
        expected_hidden = hidden_states.clone()
        expected_residual = residual.clone()
        expected_norm = RMSResNormTorch(weight)(expected_hidden, expected_residual)
        expected_residual.add_(expected_hidden)
        norm = RMSResNorm(weight)
        stream = torch.cuda.Stream()

        with torch.cuda.stream(stream):
            with mock.patch.object(
                torch.cuda,
                "current_stream",
                side_effect=AssertionError("Python current_stream must not be called"),
            ):
                actual = norm(hidden_states, residual)
            done = stream.record_event()

        torch.cuda.current_stream().wait_event(done)
        self.assertIs(actual[0], hidden_states)
        self.assertIs(actual[1], residual)
        torch.testing.assert_close(
            actual[0].float(), expected_norm.float(), atol=3e-2, rtol=3e-2
        )
        torch.testing.assert_close(
            actual[1].float(), expected_residual.float(), atol=3e-2, rtol=3e-2
        )

    def test_current_stream_binding_cpu_overhead(self):
        width = 4096
        hidden_states = torch.zeros((419, width), dtype=torch.bfloat16)
        residual = torch.zeros_like(hidden_states)
        weight = torch.ones(width, dtype=torch.bfloat16)

        def baseline():
            rtp_llm_ops.fused_add_rmsnorm(
                hidden_states,
                residual,
                weight,
                1e-6,
                torch.cuda.current_stream().cuda_stream,
            )

        def candidate():
            rtp_llm_ops.fused_add_rmsnorm_current_stream(
                hidden_states, residual, weight, 1e-6
            )

        for _ in range(30):
            baseline()
            candidate()
        torch.cuda.synchronize()

        samples = {"baseline": [], "candidate": []}
        for _ in range(60):
            for name, fn in (("baseline", baseline), ("candidate", candidate)):
                torch.cuda.synchronize()
                begin = time.thread_time_ns()
                for _ in range(100):
                    fn()
                samples[name].append((time.thread_time_ns() - begin) / 100_000.0)
        torch.cuda.synchronize()

        baseline_us = statistics.median(samples["baseline"])
        candidate_us = statistics.median(samples["candidate"])
        print(
            "fused-add rmsnorm current-stream CPU us/call: "
            f"baseline={baseline_us:.3f}, candidate={candidate_us:.3f}"
        )
        self.assertLess(candidate_us, baseline_us)


if __name__ == "__main__":
    main()
