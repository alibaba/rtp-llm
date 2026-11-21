"""Real BERT modules/checkpoint loading; CPU oracle replaces only GPU primitives."""

import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch
from safetensors.torch import load_file, save_file

from rtp_llm.models_py.layers.linear import RowParallelLinear
from rtp_llm.models_py.new_models.bert.parameters import BertLayer, PackedBertAttention
from rtp_llm.models_py.quant_methods.base import QuantizationConfig


class BertPtpcEpilogueTest(unittest.TestCase):
    def setUp(self):
        aiter = types.ModuleType("aiter")
        aiter.hipb_create_extension = mock.Mock()
        aiter.hipb_mm = mock.Mock(side_effect=self.hipb_oracle)
        aiter.gemm_a8w8_bpreshuffle = mock.Mock()
        gemm = types.ModuleType("aiter.ops.gemm_op_a8w8")
        gemm.gemm_a8w8_bpreshuffle_cktile = mock.Mock()
        quant = types.ModuleType("rtp_llm.models_py.kernels.rocm.fp8_kernel")
        quant.rocm_per_token_quant_fp8 = lambda x, **kwargs: (
            x.to(torch.float8_e4m3fn),
            torch.ones(x.shape[0], 1),
        )
        quant.rocm_per_token_group_quant_fp8 = mock.Mock()
        patch = mock.patch.dict(
            sys.modules,
            {
                "aiter": aiter,
                "aiter.ops": types.ModuleType("aiter.ops"),
                "aiter.ops.gemm_op_a8w8": gemm,
                "rtp_llm.models_py.kernels.rocm.fp8_kernel": quant,
            },
        )
        patch.start()
        self.addCleanup(patch.stop)
        module = importlib.import_module(
            "rtp_llm.models_py.modules.factory.linear.impl.rocm.fp8_ptpc_linear"
        )
        self.backend = importlib.reload(module)
        self.hipb = aiter.hipb_mm
        cache = mock.patch.object(
            self.backend.RocmFp8PTPCLinearWithSwizzle,
            "_get_solution_index",
            return_value=-1,
        )
        cache.start()
        self.addCleanup(cache.stop)

    @staticmethod
    def hipb_oracle(
        x,
        weight,
        solution_index,
        *,
        bias,
        out_dtype,
        scaleA,
        scaleB,
        bpreshuffle,
        use_gelu=False
    ):
        # Arithmetic epilogue oracle, NOT a claim of hipBLASLt kernel execution.
        acc = (x.float() * scaleA).matmul(weight.float() * scaleB)
        if bias is not None:
            acc = acc + bias.float()
        if use_gelu:
            acc = torch.nn.functional.gelu(acc)
        return acc.to(out_dtype)

    def executor(self, projection, bias):
        return self.backend.RocmFp8PTPCLinearWithSwizzle(
            weight=projection.weight.T,
            weight_scales=projection.weight_scale.T,
            bias=bias,
        )

    def make_layer(self):
        torch.manual_seed(91296857)
        cfg = types.SimpleNamespace(
            hidden_size=32,
            inter_size=64,
            layernorm_eps=1e-5,
            attn_config=types.SimpleNamespace(head_num=4, size_per_head=8),
        )
        load = types.SimpleNamespace(
            attn_tp_size=1,
            attn_tp_rank=0,
            ffn_tp_size=1,
            ffn_tp_rank=0,
            compute_dtype=torch.bfloat16,
            quant_config=QuantizationConfig("fp8_per_channel"),
        )
        layer = BertLayer(cfg, load, 0)
        state = {}
        for name in ("qkv", "attn_output", "intermediate", "output"):
            projection = getattr(layer, name)
            state[name + ".weight"] = (
                torch.randn_like(projection.weight.float()) * 0.3
            ).to(torch.float8_e4m3fn)
            state[name + ".weight_scale"] = torch.linspace(
                0.07, 0.19, projection.weight.shape[0]
            ).reshape(-1, 1)
            state[name + ".bias"] = (
                torch.randn_like(projection.bias.float()) * 0.2
            ).to(torch.bfloat16)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "bert_layer.safetensors")
            save_file(state, path)
            checkpoint = load_file(path)
            for name in ("qkv", "attn_output", "intermediate", "output"):
                projection = getattr(layer, name)
                projection.load_weights(
                    {
                        key[len(name) + 1 :]: value
                        for key, value in checkpoint.items()
                        if key.startswith(name + ".")
                    }
                )
                projection.quant_method._runtime_backend = "rocm_hipblaslt_ptpc"
                projection.quant_method._rocm_executor = self.executor(projection, None)
        with torch.no_grad():
            for norm in (layer.attn_norm, layer.output_norm):
                norm.weight.fill_(1)
                norm.bias.zero_()
        return layer

    def test_real_bert_projections_match_main_bias_and_gelu_epilogues(self):
        layer = self.make_layer()
        different = 0
        for name in ("qkv", "attn_output", "intermediate", "output"):
            projection = getattr(layer, name)
            x = torch.randn(7, projection.input_size, dtype=torch.bfloat16)
            main = self.executor(projection, projection.bias)
            use_gelu = name == "intermediate"
            expected = main.forward_with_bias_gelu(x) if use_gelu else main(x)
            actual = projection.forward_with_bias_gelu(x) if use_gelu else projection(x)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            old_order = self.executor(projection, None)(x) + projection.bias
            if use_gelu:
                old_order = torch.nn.functional.gelu(old_order)
            different += int((old_order != expected).sum())
        self.assertGreater(
            different, 0, "fixture must expose the pre-fix BF16 rounding difference"
        )

    def test_actual_bert_layer_uses_main_ffn_epilogue(self):
        layer = self.make_layer()
        x = torch.randn(3, 32, dtype=torch.bfloat16)
        attention = PackedBertAttention([3], 4, 8)
        main = {
            name: self.executor(getattr(layer, name), getattr(layer, name).bias)
            for name in ("qkv", "attn_output", "intermediate", "output")
        }
        context = attention.forward(main["qkv"](x), None, 0).reshape(3, 32)
        residual = layer.attn_norm(main["attn_output"](context), x)
        expected = layer.output_norm(
            main["output"](main["intermediate"].forward_with_bias_gelu(residual)),
            residual,
        )
        self.hipb.reset_mock()
        actual = layer(x, attention, None)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        calls = self.hipb.call_args_list
        self.assertEqual(len(calls), 4)
        self.assertEqual(
            [bool(c.kwargs.get("use_gelu", False)) for c in calls],
            [False, False, True, False],
        )
        self.assertTrue(all(c.kwargs["bias"] is not None for c in calls))

    def test_row_tp2_bias_stays_after_reduction(self):
        projection = RowParallelLinear(
            32,
            16,
            tp_size=2,
            tp_rank=0,
            bias=True,
            quant_config=QuantizationConfig("fp8_per_channel"),
            params_dtype=torch.bfloat16,
        )
        with torch.no_grad():
            projection.weight.fill_(1)
            projection.weight_scale.fill_(0.11)
            projection.bias.fill_(0.13)
        projection.quant_method._runtime_backend = "rocm_hipblaslt_ptpc"
        projection.quant_method._rocm_executor = self.executor(projection, None)
        self.assertIsNone(projection.matmul_bias)
        x = torch.ones(2, projection.input_size, dtype=torch.bfloat16)
        partial = self.executor(projection, None)(x)
        self.hipb.reset_mock()
        with mock.patch(
            "rtp_llm.models_py.distributed.collective_torch.all_reduce",
            side_effect=lambda value, **kw: value * 2,
        ) as reduce:
            actual = projection(x)
        reduce.assert_called_once()
        self.assertIsNone(self.hipb.call_args.kwargs["bias"])
        torch.testing.assert_close(
            actual, partial * 2 + projection.bias, rtol=0, atol=0
        )


if __name__ == "__main__":
    unittest.main()
