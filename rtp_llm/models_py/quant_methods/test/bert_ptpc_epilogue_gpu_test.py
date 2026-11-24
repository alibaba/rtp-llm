"""Actual BERT projection checkpoints versus the main ROCm hipBLASLt executor."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import load_file, save_file

from rtp_llm.models_py.new_models.bert.parameters import BertLayer
from rtp_llm.models_py.quant_methods.base import QuantizationConfig


class BertPtpcGpuEpilogueTest(unittest.TestCase):
    def test_online_projection_from_float_checkpoint_matches_canonical_ptpc(self):
        from rtp_llm.models_py.layers.linear import ColumnParallelLinear
        from rtp_llm.utils.swizzle_utils import swizzle_tensor

        self.assertTrue(torch.cuda.is_available() and torch.version.hip)
        quant = QuantizationConfig(
            "fp8_per_channel_online", hw_kernel_config=SimpleNamespace(use_swizzleA=True)
        )
        layer = ColumnParallelLinear(
            128, 256, bias=False, params_dtype=torch.bfloat16, quant_config=quant
        ).cuda()
        torch.manual_seed(91403169)
        source = torch.randn(256, 128, dtype=torch.float32)
        source[0].zero_()
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "online.safetensors")
            save_file({"weight": source}, path)
            layer.load_weights(load_file(path))
            layer.process_weights_after_loading()

        # Independent weight layout and quantization used by the legacy linear.
        # Compare bytes before looking at the output so an activation/weight
        # scale cancellation cannot hide a different encoding.
        w = source.to(torch.bfloat16).cuda().t().contiguous()
        s = (w.abs().amax(0, keepdim=True).clamp_min(1e-4) / 448.0).float()
        q = (w / s).to(torch.float8_e4m3fn).t().contiguous()
        if "gfx950" not in torch.cuda.get_device_properties(w.device).gcnArchName:
            raw = q.view(torch.int8)
            raw.masked_fill_(raw == -128, 0)
            q = raw.view(torch.float8_e4m3fnuz)
            s = s * 2
        packed = swizzle_tensor(q, False).T
        self.assertTrue(torch.equal(layer.weight.view(torch.uint8), packed.view(torch.uint8)))
        torch.testing.assert_close(layer.weight_scale, s, rtol=0, atol=0)
        from rtp_llm.models_py.modules.factory.linear.impl.rocm.fp8_ptpc_linear import (
            RocmFp8PTPCLinearWithSwizzle,
        )

        legacy = RocmFp8PTPCLinearWithSwizzle(weight=packed, weight_scales=s, bias=None)
        x = torch.randn(6, 128, dtype=torch.bfloat16, device="cuda")
        torch.testing.assert_close(layer(x), legacy(x), rtol=0, atol=0)

    def test_bert_projection_checkpoint_bias_and_gelu_match_main(self):
        self.assertIsNotNone(torch.version.hip, "requires the ROCm target")
        self.assertTrue(torch.cuda.is_available(), "requires the assigned GPU")
        from rtp_llm.models_py.modules.factory.linear.impl.rocm.fp8_ptpc_linear import (
            RocmFp8PTPCLinearWithSwizzle,
        )

        cfg = SimpleNamespace(
            hidden_size=128,
            inter_size=256,
            layernorm_eps=1e-5,
            attn_config=SimpleNamespace(head_num=4, size_per_head=32),
        )
        load = SimpleNamespace(
            attn_tp_size=1,
            attn_tp_rank=0,
            ffn_tp_size=1,
            ffn_tp_rank=0,
            compute_dtype=torch.bfloat16,
            quant_config=QuantizationConfig(
                "fp8_per_channel", hw_kernel_config=SimpleNamespace(use_swizzleA=True)
            ),
        )
        layer = BertLayer(cfg, load, 0).cuda()
        state = {}
        torch.manual_seed(91296857)
        for name in ("qkv", "attn_output", "intermediate", "output"):
            projection = getattr(layer, name)
            state[name + ".weight"] = (torch.randn(projection.weight.shape) * 0.3).to(
                torch.float8_e4m3fn
            )
            state[name + ".weight_scale"] = torch.linspace(
                0.07, 0.19, projection.weight.shape[0]
            ).reshape(-1, 1)
            state[name + ".bias"] = (torch.randn(projection.bias.shape) * 0.2).to(
                torch.bfloat16
            )
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "bert_layer.safetensors")
            save_file(state, path)
            checkpoint = load_file(path)
            for name in ("qkv", "attn_output", "intermediate", "output"):
                with self.subTest(projection=name):
                    projection = getattr(layer, name)
                    projection.load_weights(
                        {
                            key[len(name) + 1 :]: value
                            for key, value in checkpoint.items()
                            if key.startswith(name + ".")
                        }
                    )
                    projection.process_weights_after_loading()
                    self.assertEqual(
                        projection.quant_method._runtime_backend, "rocm_hipblaslt_ptpc"
                    )
                    # Exactly the legacy LinearFactory executor and epilogue,
                    # using the same final weight/scale storage, not a fake GEMM.
                    main = RocmFp8PTPCLinearWithSwizzle(
                        weight=projection.weight,
                        weight_scales=projection.weight_scale,
                        bias=projection.bias,
                    )
                    for batch in (1, 7, 128):
                        x = torch.randn(
                            batch,
                            projection.input_size,
                            dtype=torch.bfloat16,
                            device="cuda",
                        )
                        if name == "intermediate":
                            expected = main.forward_with_bias_gelu(x)
                            actual = projection.forward_with_bias_gelu(x)
                        else:
                            expected = main(x)
                            actual = projection(x)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
