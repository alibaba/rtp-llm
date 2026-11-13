"""Shared-layer contracts; CPU shard tests are not distributed GPU smoke."""

import os
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from rtp_llm.models_py.layers.activation import silu_and_mul
from rtp_llm.models_py.layers.collectives import Group
from rtp_llm.models_py.layers.embedding import ParallelLMHead, VocabParallelEmbedding
from rtp_llm.models_py.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from rtp_llm.models_py.layers.norm import RMSNorm, RMSResNorm
from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.quant_methods import QuantizationConfig, UnquantizedLinearMethod


class SharedLayersTest(unittest.TestCase):
    def test_import_and_tp1_forward_do_not_load_native_backends(self):
        script = """
import importlib.abc
import sys
import torch

class RejectBackends(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        blocked = (
            "rtp_llm.ops", "rtp_llm.models_py.new_models",
            "rtp_llm.models_py.distributed.collective_torch",
            "flashinfer", "aiter", "deep_gemm", "deep_ep",
        )
        if any(fullname == name or fullname.startswith(name + ".") for name in blocked):
            raise AssertionError("unexpected runtime import: " + fullname)

sys.meta_path.insert(0, RejectBackends())
from rtp_llm.models_py.layers import ColumnParallelLinear
from rtp_llm.models_py.registry import list_models
layer = ColumnParallelLinear(2, 2, params_dtype=torch.float32)
layer.load_weights({"weight": torch.eye(2)})
layer.process_weights_after_loading()
torch.testing.assert_close(layer(torch.ones(1, 2)), torch.ones(1, 2))
assert list_models() == []
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_real_load_with_shared_layers_and_repeated_finalization(self):
        class TinyModel(RtpModule):
            def __init__(self, model_config, load_config):
                super().__init__()
                self.embed = VocabParallelEmbedding(6, 4, params_dtype=torch.float32)
                self.proj = ColumnParallelLinear(
                    4, 4, bias=True, params_dtype=torch.float32
                )
                self.norm = RMSNorm(4, params_dtype=torch.float32)
                self.head = ParallelLMHead(6, 4, params_dtype=torch.float32)

            def forward(self, ids):
                return self.head(self.norm(self.proj(self.embed(ids))))

        weights = {
            "embed.weight": torch.arange(24, dtype=torch.float32).reshape(6, 4) / 10,
            "proj.weight": torch.eye(4) * 2,
            "proj.bias": torch.arange(4, dtype=torch.float32),
            "norm.weight": torch.tensor([1.0, 2.0, 0.5, 1.5]),
            "head.weight": torch.arange(24, dtype=torch.float32).reshape(6, 4) / 20,
        }
        with tempfile.TemporaryDirectory() as path:
            save_file(weights, os.path.join(path, "model.safetensors"))
            with mock.patch(
                "rtp_llm.models_py.model_loader.get_model_class",
                return_value=TinyModel,
            ):
                model = NewModelLoader(
                    types.SimpleNamespace(model_type="shared_layer_test"),
                    NewLoaderConfig(device="cpu", compute_dtype=torch.float32),
                    model_path=path,
                ).load()
        ids = torch.tensor([[0, 2, 5]])
        hidden = F.linear(
            F.embedding(ids, weights["embed.weight"]),
            weights["proj.weight"],
            weights["proj.bias"],
        )
        hidden = hidden * torch.rsqrt(hidden.square().mean(-1, keepdim=True) + 1e-6)
        expected = F.linear(hidden * weights["norm.weight"], weights["head.weight"])
        torch.testing.assert_close(model(ids), expected)
        self.assertFalse(model.training)
        self.assertEqual(model.proj.weight.stride(), (1, 4))
        with mock.patch.object(
            model.proj.quant_method,
            "process_weights_after_loading",
            wraps=model.proj.quant_method.process_weights_after_loading,
        ) as convert:
            NewModelLoader._run_post_load_hooks(model)
            convert.assert_not_called()
        torch.testing.assert_close(model(ids), expected)

    def test_column_and_row_partition_algebra(self):
        weight = torch.arange(24, dtype=torch.float32).reshape(6, 4) / 10
        x = torch.tensor([[1.0, -2.0, 3.0, 0.5]])
        columns, rows = [], []
        for rank in range(2):
            column = ColumnParallelLinear(
                4, 6, tp_size=2, tp_rank=rank, params_dtype=torch.float32
            )
            row = RowParallelLinear(
                4,
                6,
                tp_size=2,
                tp_rank=rank,
                reduce_output=False,
                params_dtype=torch.float32,
            )
            for layer in (column, row):
                layer.load_weights({"weight": weight})
                NewModelLoader._validate_loaded_weights(layer)
                layer.process_weights_after_loading()
            columns.append(column(x))
            rows.append(row(x[:, rank * 2 : (rank + 1) * 2]))
        expected = F.linear(x, weight)
        torch.testing.assert_close(torch.cat(columns, dim=-1), expected)
        torch.testing.assert_close(sum(rows), expected)

    def test_row_reduces_before_bias_and_rejects_biased_deferral(self):
        layer = RowParallelLinear(
            4, 2, tp_size=2, bias=True, params_dtype=torch.float32
        )
        layer.load_weights(
            {
                "weight": torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]]),
                "bias": torch.tensor([3.0, 5.0]),
            }
        )
        with mock.patch(
            "rtp_llm.models_py.layers.linear.all_reduce",
            side_effect=lambda tensor, group: tensor * 2,
        ) as reduce:
            torch.testing.assert_close(
                layer(torch.tensor([[2.0, 4.0]])), torch.tensor([[7.0, 13.0]])
            )
        self.assertEqual(reduce.call_args.kwargs["group"], Group.TP)
        with self.assertRaisesRegex(ValueError, "bias"):
            layer(torch.ones(1, 2), reduce_output=False)

    def test_merged_fused_and_separate_shards_match(self):
        gate = torch.arange(8, dtype=torch.float32).reshape(4, 2)
        up = gate + 10
        for rank in range(2):
            results = []
            for fused in (False, True):
                layer = MergedColumnParallelLinear(
                    2,
                    8,
                    tp_size=2,
                    tp_rank=rank,
                    shard_names=["gate_proj", "up_proj"],
                    params_dtype=torch.float32,
                )
                checkpoint = (
                    {"gate_up_proj.weight": torch.cat((gate, up))}
                    if fused
                    else {"gate_proj.weight": gate, "up_proj.weight": up}
                )
                layer.load_weights(checkpoint)
                NewModelLoader._validate_loaded_weights(layer)
                layer.process_weights_after_loading()
                results.append(layer(torch.ones(1, 2)))
            expected_weight = torch.cat(
                (
                    gate[rank * 2 : (rank + 1) * 2],
                    up[rank * 2 : (rank + 1) * 2],
                )
            )
            for result in results:
                torch.testing.assert_close(
                    result, F.linear(torch.ones(1, 2), expected_weight)
                )

    def test_qkv_replicates_kv_heads_in_contiguous_rank_groups(self):
        q = torch.arange(16, dtype=torch.float32).reshape(8, 2)
        k = torch.tensor([[20.0, 21.0], [30.0, 31.0]])
        v = k + 20
        for rank, kv_head in enumerate((0, 0, 1, 1)):
            layer = QKVParallelLinear(
                hidden_size=2,
                num_heads=8,
                num_kv_heads=2,
                head_dim=1,
                tp_size=4,
                tp_rank=rank,
                params_dtype=torch.float32,
            )
            layer.load_weights(
                {"q_proj.weight": q, "k_proj.weight": k, "v_proj.weight": v}
            )
            NewModelLoader._validate_loaded_weights(layer)
            layer.process_weights_after_loading()
            expected = torch.cat(
                (
                    q[rank * 2 : rank * 2 + 2],
                    k[kv_head : kv_head + 1],
                    v[kv_head : kv_head + 1],
                )
            )
            torch.testing.assert_close(layer.weight, expected)

    def test_missing_duplicate_and_invalid_linear_weights_fail(self):
        layer = MergedColumnParallelLinear(
            2, 8, shard_names=["gate_proj", "up_proj"], params_dtype=torch.float32
        )
        with self.assertRaisesRegex(ValueError, "rows"):
            layer.load_weights({"gate_proj.weight": torch.ones(5, 2)})
        layer.load_weights({"gate_proj.weight": torch.ones(4, 2)})
        with self.assertRaisesRegex(RuntimeError, "missing"):
            NewModelLoader._validate_loaded_weights(layer)
        with self.assertRaisesRegex(RuntimeError, "Duplicate"):
            layer.load_weights({"gate_proj.weight": torch.zeros(4, 2)})
        with self.assertRaisesRegex(RuntimeError, "mix fused"):
            layer.load_weights({"gate_up_proj.weight": torch.ones(8, 2)})
        with self.assertRaises(TypeError):
            ColumnParallelLinear(2, 4, tp_szie=2)
        with self.assertRaises(ValueError):
            ColumnParallelLinear(2, 3, tp_size=2)
        with self.assertRaises(ValueError):
            ColumnParallelLinear(2, 4, gather_output=True)

    def test_embedding_partition_and_collective_boundary(self):
        weight = torch.arange(24, dtype=torch.float32).reshape(6, 4)
        ids = torch.tensor([[0, 2, 3, 5]])
        expected = F.embedding(ids, weight)
        for rank in range(2):
            layer = VocabParallelEmbedding(
                6, 4, tp_size=2, tp_rank=rank, params_dtype=torch.float32
            )
            layer.load_weights({"weight": weight})
            with mock.patch(
                "rtp_llm.models_py.layers.embedding.all_reduce",
                return_value=expected,
            ) as reduce:
                torch.testing.assert_close(layer(ids), expected)
            mask = (ids >= rank * 3) & (ids < (rank + 1) * 3)
            torch.testing.assert_close(
                reduce.call_args.args[0], expected * mask.unsqueeze(-1)
            )
            self.assertEqual(reduce.call_args.kwargs["group"], Group.TP)
            for invalid in (-1, 6):
                with self.assertRaises(IndexError):
                    layer(torch.tensor([invalid]))

    def test_lm_head_restores_vocab_order_for_multidimensional_input(self):
        weight = torch.arange(24, dtype=torch.float32).reshape(6, 4) / 10
        x = torch.arange(16, dtype=torch.float32).reshape(2, 2, 4) / 10
        partials = [F.linear(x, shard).reshape(-1, 3) for shard in weight.chunk(2)]
        for rank in range(2):
            layer = ParallelLMHead(
                6, 4, tp_size=2, tp_rank=rank, params_dtype=torch.float32
            )
            layer.load_weights({"weight": weight})
            with mock.patch(
                "rtp_llm.models_py.layers.embedding.all_gather",
                return_value=torch.cat(partials),
            ) as gather:
                torch.testing.assert_close(layer(x), F.linear(x, weight))
            torch.testing.assert_close(gather.call_args.args[0], partials[rank])
            self.assertEqual(gather.call_args.kwargs["group"], Group.TP)

    def test_norm_and_activation_cpu_values_and_aliases(self):
        torch.manual_seed(1)
        for width in (4, 769):
            x, residual = torch.randn(3, width), torch.randn(3, width)
            weight = torch.linspace(0.5, 1.5, width)
            norm = RMSNorm(width, params_dtype=torch.float32)
            norm.load_weights({"weight": weight})
            expected = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + norm.eps)
            torch.testing.assert_close(norm(x), expected * weight)
            fused = RMSResNorm(width, params_dtype=torch.float32)
            fused.load_weights({"weight": weight})
            summed = x + residual
            expected = (
                summed
                * torch.rsqrt(summed.square().mean(-1, keepdim=True) + fused.eps)
                * weight
            )
            actual, actual_residual = fused(x, residual)
            self.assertIs(actual, x)
            self.assertIs(actual_residual, residual)
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(actual_residual, summed)
        gate_up = torch.randn(2, 12)
        gate, up = gate_up.chunk(2, dim=-1)
        torch.testing.assert_close(silu_and_mul(gate_up), F.silu(gate) * up)
        with self.assertRaises(ValueError):
            silu_and_mul(torch.zeros(2, 3))

    def test_unavailable_quantization_fails_without_backend_import(self):
        for method in ("fp8", "fp8_online", "typo"):
            with self.subTest(method=method):
                with self.assertRaisesRegex(ValueError, "not supported"):
                    ColumnParallelLinear(2, 4, quant_config=QuantizationConfig(method))

    def test_exclusion_contract_for_fused_projections(self):
        config = QuantizationConfig(
            "fp8", ignored_layers=["mlp.gate_proj", "mlp.up_proj"]
        )
        layer = MergedColumnParallelLinear(
            2,
            8,
            shard_names=["gate_proj", "up_proj"],
            prefix="mlp.gate_up_proj",
            quant_config=config,
        )
        self.assertIsInstance(layer.quant_method, UnquantizedLinearMethod)
        with self.assertRaisesRegex(ValueError, "partially match"):
            MergedColumnParallelLinear(
                2,
                8,
                shard_names=["gate_proj", "up_proj"],
                prefix="mlp.gate_up_proj",
                quant_config=QuantizationConfig(
                    "fp8", ignored_layers=["mlp.gate_proj"]
                ),
            )


if __name__ == "__main__":
    unittest.main()
