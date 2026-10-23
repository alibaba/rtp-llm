import itertools
from unittest import TestCase, main

import torch
from torch import dtype as _dtype

from rtp_llm.models_py.modules import AddBiasResLayerNorm, AddBiasResLayerNormTorch


class RMSLayerNormTest(TestCase):
    DTYPES = [torch.half, torch.bfloat16]
    NUM_TOKENS = [7, 83, 4096]
    HIDDEN_SIZES = [769, 770, 771, 5120, 5124, 5125, 5126, 8192, 8199]

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("A real CUDA or ROCm GPU is required")
        torch.set_default_device("cuda")

    def _run_res_layernorm_test(self, num_tokens: int, hidden_size: int, dtype: _dtype):
        torch.manual_seed(0)
        w = torch.randn(hidden_size, dtype=dtype)
        beta = torch.randn(hidden_size, dtype=dtype)
        res_layernorm = AddBiasResLayerNorm(w, beta)
        res_layernorm_torch = AddBiasResLayerNormTorch(w, beta)
        x = torch.randn(num_tokens, hidden_size, dtype=dtype)
        residual = torch.randn(num_tokens, hidden_size, dtype=dtype)
        bias = torch.randn(hidden_size, dtype=dtype)
        self.assertTrue(
            torch.allclose(
                res_layernorm_torch(x, residual, bias),
                res_layernorm(x, residual, bias),
                atol=1e-2,
                rtol=1e-2,
            )
        )

    def test_res_layernorm(self):
        for params in itertools.product(
            self.NUM_TOKENS,
            self.HIDDEN_SIZES,
            self.DTYPES,
        ):
            with self.subTest(
                num_tokens=params[0], hidden_size=params[1], dtype=params[2]
            ):
                self._run_res_layernorm_test(*params)

    def test_res_layernorm_graph_replay_on_capture_stream(self):
        # BGE-large uses this fallback at hidden_size=1024. Include both a
        # producer and consumer so a launch on the default stream cannot pass
        # by reading a result left over from warmup.
        for dtype in self.DTYPES:
            with self.subTest(dtype=dtype):
                torch.manual_seed(0)
                hidden_size = 1024
                weight = torch.randn(hidden_size, dtype=dtype)
                beta = torch.randn(hidden_size, dtype=dtype)
                bias = torch.randn(hidden_size, dtype=dtype)
                module = AddBiasResLayerNorm(weight, beta)
                reference = AddBiasResLayerNormTorch(weight, beta)
                x = torch.randn(32, hidden_size, dtype=dtype)
                residual = torch.randn_like(x)
                static_x = x.clone()
                static_residual = residual.clone()

                def forward():
                    # The fused op also writes its input and residual buffers.
                    return module(static_x * 1.25, static_residual.clone(), bias) * 0.75

                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    forward()
                    forward()
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    output = forward()

                for scale in (0.5, -1.5, 2.0):
                    static_x.copy_(x * scale)
                    static_residual.copy_(residual * scale)
                    graph.replay()
                    torch.cuda.synchronize()
                    expected = (
                        reference(x * scale * 1.25, residual * scale, bias) * 0.75
                    )
                    self.assertTrue(torch.isfinite(output).all().item())
                    torch.testing.assert_close(output, expected, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    main()
