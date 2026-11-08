"""SM120 hardware gate for the CUTLASS blockwise FP8 linear backend."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig
from rtp_llm.device.device_type import DeviceType
from rtp_llm.model_loader.per_block_fp8_quant_weight import PerBlockFp8Weight
from rtp_llm.model_loader.weight_module import CompositeWeight
from rtp_llm.models_py.modules.factory.linear import LinearFactory
from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_vllm_blockwise_sm120_linear import (
    CudaFp8VllmBlockwiseLinear,
)
from rtp_llm.models_py.utils.arch import is_sm120


class Sm120Fp8BlockwiseLinearHardwareTest(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("SM120 hardware gate was scheduled without a CUDA device")
        if not is_sm120():
            self.skipTest(
                "SM120 hardware gate was scheduled on "
                f"compute capability {torch.cuda.get_device_capability()}"
            )
        self.device = torch.device("cuda", torch.cuda.current_device())
        torch.manual_seed(7)

    def test_cutlass_blockwise_linear_eager_and_cuda_graph(self) -> None:
        # Deliberately rectangular and non-uniform across output blocks: a
        # loader-side (N,K)->(K,N) reshape cannot hide behind square geometry
        # or all-one scales.
        # Simulate force_cpu_load_weights: post-processing sees a CPU tensor,
        # while exported_device identifies the final SM120 inference target.
        weight = (
            (torch.randn((384, 256), dtype=torch.float32, device=self.device) * 0.05)
            .to(torch.float8_e4m3fn)
            .cpu()
        )
        weight_scale = torch.tensor(
            [[0.50, 0.75], [1.00, 1.25], [1.50, 1.75]], dtype=torch.float32
        )

        loader = PerBlockFp8Weight.__new__(PerBlockFp8Weight)
        loader.kernel = SimpleNamespace(name="kernel")
        loader.scale = SimpleNamespace(name="scale")
        exported_device = SimpleNamespace(
            get_device_type=lambda: DeviceType.Cuda,
            get_device_id=lambda: self.device.index,
            maybe_rewrite_weight_by_key=lambda _key, value, **_kwargs: value,
        )
        with patch.object(
            CompositeWeight,
            "_postprocess",
            return_value={"kernel": weight, "scale": weight_scale},
        ):
            processed = loader._postprocess(
                {"kernel": weight, "scale": weight_scale},
                "cpu",
                SimpleNamespace(exported_device=exported_device, use_swizzleA=False),
            )

        self.assertEqual(processed["kernel"].device.type, "cpu")
        self.assertEqual(tuple(processed["kernel"].shape), (384, 256))
        self.assertEqual(tuple(processed["scale"].shape), (3, 2))
        processed = {key: value.to(self.device) for key, value in processed.items()}
        linear = LinearFactory.create_linear(
            processed["kernel"],
            None,
            processed["scale"],
            Fp8BlockWiseQuantConfig(),
        )
        self.assertIsInstance(linear, CudaFp8VllmBlockwiseLinear)
        dequant_weight = processed["kernel"].float() * processed[
            "scale"
        ].repeat_interleave(128, 0).repeat_interleave(128, 1)

        # 260 and 384 exercise the generic CUTLASS dispatch beyond the tuned
        # 64/65 and 256/257 boundary cases.
        for rows in (64, 65, 256, 257, 260, 384):
            with self.subTest(rows=rows):
                activation = torch.randn(
                    (rows, 256), dtype=torch.bfloat16, device=self.device
                )
                eager = linear(activation)
                reference = activation.float() @ dequant_weight.transpose(0, 1)
                torch.testing.assert_close(
                    eager.float(),
                    reference,
                    rtol=2e-1,
                    atol=2e-1,
                )

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    graph_output = linear(activation)
                activation.copy_(torch.randn_like(activation))
                graph.replay()
                torch.cuda.synchronize(self.device)
                replay_output = graph_output.clone()
                eager_after_update = linear(activation)
                torch.testing.assert_close(replay_output, eager_after_update)

    def test_cutlass_blockwise_linear_partial_blocks(self) -> None:
        for n, k in ((2112, 256), (128, 193), (129, 257)):
            with self.subTest(n=n, k=k):
                weight = (torch.randn(n, k, device=self.device) * 0.05).to(
                    torch.float8_e4m3fn
                )
                scales = (
                    torch.rand((n + 127) // 128, (k + 127) // 128, device=self.device)
                    + 0.5
                )
                bias = torch.randn(n, dtype=torch.bfloat16, device=self.device)
                linear = LinearFactory.create_linear(
                    weight, bias, scales, Fp8BlockWiseQuantConfig()
                )
                self.assertIsInstance(linear, CudaFp8VllmBlockwiseLinear)
                activation = torch.randn(3, k, dtype=torch.bfloat16, device=self.device)
                output = torch.empty(3, n, dtype=torch.bfloat16, device=self.device)
                eager = linear(activation, out=output)
                self.assertIs(eager, output)
                expanded = scales.repeat_interleave(128, 0).repeat_interleave(128, 1)
                reference = activation.float() @ (weight.float() * expanded[:n, :k]).T
                reference = reference.to(torch.bfloat16) + bias
                torch.testing.assert_close(eager, reference, rtol=2e-1, atol=2e-1)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = linear(activation)
                activation.copy_(torch.randn_like(activation))
                graph.replay()
                torch.cuda.synchronize(self.device)
                torch.testing.assert_close(captured, linear(activation))


if __name__ == "__main__":
    unittest.main()
