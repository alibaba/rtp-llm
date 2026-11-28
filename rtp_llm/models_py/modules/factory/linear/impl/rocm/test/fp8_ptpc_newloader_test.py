"""Real ROCm loading and kernel regression; no substituted GPU backends."""

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from safetensors.torch import save_file

from rtp_llm.models_py.layers.linear import ColumnParallelLinear
from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.quant_methods.base import QuantizationConfig
from rtp_llm.models_py.registry import MODEL_REGISTRY


class _PtpcOwner(RtpModule):
    def __init__(self, config, load_config):
        super().__init__()
        self.projection = ColumnParallelLinear(
            input_size=config.input_size,
            output_size=config.output_size,
            tp_size=load_config.tp_size,
            tp_rank=load_config.tp_rank,
            quant_config=load_config.quant_config,
            prefix="projection",
            params_dtype=load_config.compute_dtype,
        )


class Fp8PtpcNewLoaderTest(unittest.TestCase):
    def test_checkpoint_backends_and_tp_local_shards_use_real_executor(self):
        self.assertIsNotNone(torch.version.hip, "target requires ROCm")
        self.assertTrue(torch.cuda.is_available(), "target requires its assigned GPU")
        import aiter
        from aiter.ops.gemm_op_a8w8 import gemm_a8w8_bpreshuffle_cktile
        from aiter.ops.shuffle import shuffle_weight

        from rtp_llm.models_py.kernels.rocm.fp8_kernel import rocm_per_token_quant_fp8
        from rtp_llm.models_py.quant_methods.fp8 import _runtime_fp8_dtype

        n, k = 2048, 1024
        # Checkpoint FN is portable; runtime encoding is discovered on the GPU.
        stored_weight = ((torch.arange(n * k) % 61 - 30).float() / 8).reshape(n, k)
        stored_weight = stored_weight.to(torch.float8_e4m3fn)
        stored_scale = (torch.arange(n).float().reshape(n, 1) % 7 + 1) / 64
        config = SimpleNamespace(
            model_type="rocm_ptpc_checkpoint_fixture", input_size=k, output_size=n
        )
        with tempfile.TemporaryDirectory() as directory:
            save_file(
                {
                    "projection.weight": stored_weight,
                    "projection.weight_scale": stored_scale,
                },
                os.path.join(directory, "model.safetensors"),
            )
            # TP=2 is local rank loading on one assigned GPU, not a distributed
            # TP2 smoke. FST is also checked without requiring a process group.
            for method, tp, rank in (
                ("scratch", 1, 0),
                ("fastsafetensors", 1, 0),
                ("scratch", 2, 0),
                ("scratch", 2, 1),
            ):
                with self.subTest(method=method, tp=tp, rank=rank):
                    with mock.patch.dict(
                        MODEL_REGISTRY, {config.model_type: _PtpcOwner}
                    ):
                        loader = NewModelLoader(
                            config,
                            NewLoaderConfig(
                                device="cuda:0",
                                compute_dtype=torch.bfloat16,
                                load_method=method,
                                tp_size=tp,
                                tp_rank=rank,
                                quant_config=QuantizationConfig(
                                    "fp8_per_channel",
                                    hw_kernel_config=SimpleNamespace(
                                        use_swizzleA=False
                                    ),
                                ),
                            ),
                            model_path=directory,
                        )
                        model = loader.load()
                    self.assertEqual(loader._selected_load_method.value, method)
                    layer = model.projection
                    self.assertEqual(
                        layer.quant_method._runtime_backend, "rocm_aiter_ptpc"
                    )
                    self.assertEqual(
                        layer.quant_method._rocm_executor.__name__,
                        "run_rocm_fp8_ptpc_no_swizzle",
                    )
                    runtime_dtype = _runtime_fp8_dtype(torch.device("cuda:0"))
                    expected_weight = (
                        stored_weight[rank * n // tp : (rank + 1) * n // tp]
                        .clone()
                        .cuda()
                    )
                    expected_scale = (
                        stored_scale[rank * n // tp : (rank + 1) * n // tp]
                        .clone()
                        .cuda()
                    )
                    if runtime_dtype == torch.float8_e4m3fnuz:
                        bits = expected_weight.view(torch.int8)
                        bits[bits == -128] = 0
                        expected_weight = bits.view(runtime_dtype)
                        expected_scale *= 2
                    else:
                        self.assertEqual(runtime_dtype, expected_weight.dtype)
                    expected_weight = shuffle_weight(expected_weight, layout=(16, 16))
                    self.assertEqual(layer.weight.dtype, runtime_dtype)
                    torch.testing.assert_close(
                        layer.weight.view(torch.uint8),
                        expected_weight.view(torch.uint8),
                        rtol=0,
                        atol=0,
                    )
                    torch.testing.assert_close(
                        layer.weight_scale, expected_scale, rtol=0, atol=0
                    )
                    for batch in (4, 1536):
                        x = torch.randn(batch, k, dtype=torch.bfloat16, device="cuda:0")
                        actual = layer(x)
                        qx, sx = rocm_per_token_quant_fp8(x, eps=1e-10)
                        if batch >= 1536:
                            expected = torch.empty(
                                batch, n // tp, device=x.device, dtype=x.dtype
                            )
                            gemm_a8w8_bpreshuffle_cktile(
                                qx,
                                expected_weight,
                                sx.float(),
                                expected_scale,
                                expected,
                            )
                        else:
                            expected = aiter.gemm_a8w8_bpreshuffle(
                                qx,
                                expected_weight,
                                sx.float(),
                                expected_scale,
                                None,
                                x.dtype,
                            )
                        torch.testing.assert_close(
                            actual, expected, rtol=0.02, atol=0.02
                        )


if __name__ == "__main__":
    unittest.main()
