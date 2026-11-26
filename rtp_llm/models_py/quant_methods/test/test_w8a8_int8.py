"""Real CPU checkpoint contracts; no PPU kernel or GPU result is implied."""

import os
import tempfile
import types
import unittest
from unittest import mock

import torch
from safetensors.torch import save_file

from rtp_llm.config.quant_config import CompressedW8A8Int8PerChannelQuantConfig
from rtp_llm.models_py.checkpoint_reader_test_support import requires_native_fst
from rtp_llm.models_py.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.quant_methods.base import QuantizationConfig
from rtp_llm.models_py.quant_methods.w8a8_int8 import (
    W8A8_METHOD,
    W8A8Int8LinearMethod,
    _dense_executor_class,
)
from rtp_llm.models_py.registry import MODEL_REGISTRY


class _DenseModel(RtpModule):
    def __init__(self, config, load_config):
        super().__init__()
        kwargs = dict(
            tp_size=load_config.tp_size,
            tp_rank=load_config.tp_rank,
            quant_config=load_config.quant_config,
            prefix="projection",
            bias=True,
            params_dtype=torch.bfloat16,
        )
        if config.kind == "qkv":
            self.projection = QKVParallelLinear(
                hidden_size=8, num_heads=8, num_kv_heads=2, head_dim=2, **kwargs
            )
        elif config.kind == "merged":
            self.projection = MergedColumnParallelLinear(
                input_size=8,
                output_size=32,
                shard_names=["gate_proj", "up_proj"],
                **kwargs,
            )
        elif config.kind == "row":
            self.projection = RowParallelLinear(input_size=8, output_size=16, **kwargs)
        else:
            self.projection = ColumnParallelLinear(
                input_size=8, output_size=16, **kwargs
            )


class W8A8Int8Test(unittest.TestCase):
    @torch.inference_mode()
    def test_complete_weight_scale_update_and_partial_qkv_bias(self):
        from rtp_llm.models_py.runtime_weight_update import RuntimeWeightTransaction

        files = self.checkpoint("qkv")
        leaf = self.load(files, "qkv").projection
        old = leaf.weight.clone()
        before = [p.data_ptr() for p in (leaf.weight, leaf.weight_scale, leaf.bias)]
        with self.assertRaisesRegex(ValueError, "complete weight/scale unit"):
            leaf.prepare_weight_updates({"q_proj.weight": files["projection.q_proj.weight"]})
        torch.testing.assert_close(leaf.weight, old)
        tensor = old.bitwise_not()
        scale = leaf.weight_scale * 3
        tx = RuntimeWeightTransaction(leaf.prepare_weight_updates({"weight": tensor, "weight_scale": scale}))
        tx.install()
        tx.commit()
        torch.testing.assert_close(leaf.weight, tensor)
        torch.testing.assert_close(leaf.weight_scale, scale)
        bias = leaf.bias.clone()
        updates = leaf.prepare_weight_updates({"q_proj.bias": bias[:16] + 2})
        self.assertEqual([u.name for u in updates], ["bias"])
        tx = RuntimeWeightTransaction(updates)
        tx.install()
        tx.commit()
        torch.testing.assert_close(leaf.bias[:16], bias[:16] + 2)
        torch.testing.assert_close(leaf.bias[16:], bias[16:])
        self.assertEqual(before, [p.data_ptr() for p in (leaf.weight, leaf.weight_scale, leaf.bias)])

    def checkpoint(self, kind, fused=False, flat_scales=False):
        projections = {
            "column": [("", 16)],
            "row": [("", 16)],
            "qkv": [("q_proj", 16), ("k_proj", 4), ("v_proj", 4)],
            "merged": [("gate_proj", 16), ("up_proj", 16)],
        }[kind]
        values = {}
        for index, (projection, rows) in enumerate(projections):
            prefix = "projection." + (projection + "." if projection else "")
            values[prefix + "weight"] = (
                ((torch.arange(rows * 8) + index * 7) % 255 - 127)
                .to(torch.int8)
                .reshape(rows, 8)
            )
            scale = (torch.arange(rows).float() + 1 + index) / 31
            values[prefix + "weight_scale"] = (
                scale if flat_scales else scale.reshape(rows, 1)
            )
            values[prefix + "bias"] = torch.arange(rows).to(torch.bfloat16) + index
        if fused and len(projections) > 1:
            return {
                "projection."
                + field: torch.cat(
                    [values[f"projection.{name}.{field}"] for name, _ in projections]
                )
                for field in ("weight", "weight_scale", "bias")
            }
        return values

    def load(self, weights, kind, tp=1, rank=0, method="scratch"):
        with tempfile.TemporaryDirectory() as directory:
            save_file(weights, os.path.join(directory, "model.safetensors"))
            source = CompressedW8A8Int8PerChannelQuantConfig()
            config = types.SimpleNamespace(model_type="ppu_w8a8_cpu_fixture", kind=kind)
            with mock.patch.dict(MODEL_REGISTRY, {config.model_type: _DenseModel}):
                loader = NewModelLoader(
                    config,
                    NewLoaderConfig(
                        device="cpu",
                        compute_dtype=torch.bfloat16,
                        load_method="fastsafetensors" if method == "fst" else method,
                        force_cpu_load_weights=method == "fst",
                        tp_size=tp,
                        tp_rank=rank,
                        quant_config=QuantizationConfig(
                            source.get_method(), source_config=source
                        ),
                    ),
                    model_path=directory,
                )
                result = loader.load()
                self.assertEqual(
                    loader._selected_load_method.value,
                    "fastsafetensors" if method == "fst" else "scratch",
                )
                return result

    @requires_native_fst
    def test_native_fst_cpu_real_checkpoint_matches_scratch(self):
        for kind in ("column", "row", "qkv", "merged"):
            for fused in (False, True):
                for rank in range(4):
                    with self.subTest(kind=kind, fused=fused, rank=rank):
                        source = self.checkpoint(kind, fused)
                        actual = self.load(source, kind, 4, rank, "fst").projection
                        expected = self.load(source, kind, 4, rank).projection
                        for name in ("weight", "weight_scale", "bias"):
                            torch.testing.assert_close(
                                getattr(actual, name),
                                getattr(expected, name),
                                rtol=0,
                                atol=0,
                            )

    def expected(self, kind, tp, rank):
        source = self.checkpoint(kind)
        values = {field: [] for field in ("weight", "weight_scale", "bias")}
        for name, tensor in source.items():
            field = name.rsplit(".", 1)[-1]
            if kind == "row":
                selected = (
                    tensor[:, rank * 8 // tp : (rank + 1) * 8 // tp]
                    if field == "weight"
                    else tensor
                )
            else:
                shards, shard_rank = tp, rank
                if (
                    kind == "qkv"
                    and any(part in name for part in ("k_proj", "v_proj"))
                    and tp > 2
                ):
                    shards, shard_rank = 2, rank // (tp // 2)
                width = tensor.shape[0] // shards
                selected = tensor[shard_rank * width : (shard_rank + 1) * width]
            values[field].append(selected)
        return {field: torch.cat(tensors) for field, tensors in values.items()}

    def test_real_checkpoint_tp_gqa_fusion_and_cpu_numeric_oracle(self):
        for kind in ("column", "row", "qkv", "merged"):
            for fused in (False, True):
                for tp in (1, 2, 4):
                    for rank in range(tp):
                        with self.subTest(kind=kind, fused=fused, tp=tp, rank=rank):
                            model = self.load(
                                self.checkpoint(kind, fused), kind, tp, rank
                            )
                            expected = self.expected(kind, tp, rank)
                            for name, tensor in expected.items():
                                torch.testing.assert_close(
                                    getattr(model.projection, name),
                                    tensor,
                                    rtol=0,
                                    atol=0,
                                )
                            layer = model.projection
                            x = torch.arange(
                                layer.input_size, dtype=torch.float32
                            ).reshape(1, -1)
                            actual = torch.nn.functional.linear(
                                x,
                                layer.weight.float() * layer.weight_scale,
                                layer.bias.float(),
                            )
                            reference = torch.nn.functional.linear(
                                x,
                                expected["weight"].float() * expected["weight_scale"],
                                expected["bias"].float(),
                            )
                            torch.testing.assert_close(
                                actual, reference, rtol=0, atol=0
                            )
                            self.assertIsNone(layer.quant_method._executor)

    def test_flat_scales_are_channel_rows_not_row_tp_inputs(self):
        # A flat per-output scale remains replicated for RowParallel weights.
        weights = self.checkpoint("row", flat_scales=True)
        model = self.load(weights, "row", 2, 1)
        torch.testing.assert_close(
            model.projection.weight_scale[:, 0], weights["projection.weight_scale"]
        )

    def test_missing_or_invalid_tensors_fail_before_publication(self):
        cases = []
        missing = self.checkpoint("column")
        del missing["projection.weight_scale"]
        cases.append(missing)
        for dtype in (torch.uint8, torch.bfloat16):
            value = self.checkpoint("column")
            value["projection.weight"] = value["projection.weight"].to(dtype)
            cases.append(value)
        for invalid in (0.0, -1.0, float("nan"), float("inf")):
            value = self.checkpoint("column")
            value["projection.weight_scale"][0] = invalid
            cases.append(value)
        for value in cases:
            with self.assertRaises((ValueError, TypeError, RuntimeError)):
                self.load(value, "column")

    def test_ignore_uses_unquantized_owner_and_partial_fusion_is_rejected(self):
        quant = QuantizationConfig(W8A8_METHOD, ignored_layers=["projection"])
        layer = ColumnParallelLinear(
            8, 16, quant_config=quant, prefix="projection", params_dtype=torch.bfloat16
        )
        self.assertEqual(layer.weight.dtype, torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "partially match"):
            MergedColumnParallelLinear(
                8,
                32,
                shard_names=["gate_proj", "up_proj"],
                prefix="mlp.gate_up",
                quant_config=QuantizationConfig(
                    W8A8_METHOD, ignored_layers=["mlp.gate_proj"]
                ),
            )

    def test_cpu_load_does_not_claim_cpu_inference_or_cuda_support(self):
        layer = self.load(self.checkpoint("column"), "column").projection
        with self.assertRaisesRegex(RuntimeError, "requires PPU"):
            layer(torch.ones(1, 8))
        with mock.patch(
            "rtp_llm.models_py.quant_methods.w8a8_int8.get_device_type", return_value=1
        ):
            with self.assertRaisesRegex(RuntimeError, "requires PPU"):
                layer.validate_runtime_device(torch.device("cuda"))

    def test_missing_backend_import_and_actual_gemm_symbol_fail_fast(self):
        for error in (ImportError("missing PPU"), OSError("bad ABI")):
            with mock.patch(
                "rtp_llm.models_py.quant_methods.w8a8_int8.importlib.import_module",
                side_effect=error,
            ):
                with self.assertRaisesRegex(RuntimeError, "backend is unavailable"):
                    _dense_executor_class()
        module = types.SimpleNamespace(
            PpuInt8DeepGemmLinear=lambda: None,
            per_token_quant_int8=lambda: None,
            has_deep_gemm_int8_dense=lambda: False,
        )
        with mock.patch(
            "rtp_llm.models_py.quant_methods.w8a8_int8.importlib.import_module",
            return_value=module,
        ):
            with self.assertRaisesRegex(RuntimeError, "GEMM symbol"):
                _dense_executor_class()


if __name__ == "__main__":
    unittest.main()
