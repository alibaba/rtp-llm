"""Dense NVFP4 checkpoint contracts; CPU tests do not execute a CUDA kernel."""

import os
import tempfile
import types
import unittest
from unittest import mock

import torch
from safetensors.torch import save_file

from rtp_llm.models_py.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.quant_methods.base import QuantizationConfig
from rtp_llm.models_py.quant_methods.fp4 import (
    ModelOptFp4LinearMethod,
    swizzle_nvfp4_scales,
)
from rtp_llm.models_py.registry import MODEL_REGISTRY


class _CaptureExecutor:
    def __init__(self, **weights):
        self.weights = weights


def _cpu_layout_only(method, device):
    # Isolate hardware/executor construction. Real loading, validation, TP
    # mapping, scale fusion and scale swizzle remain production implementations.
    method._backend = "cutlass"
    method._executor_type = _CaptureExecutor


class _DenseCheckpointModel(RtpModule):
    def __init__(self, config, load_config):
        super().__init__()
        common = dict(
            tp_size=load_config.tp_size,
            tp_rank=load_config.tp_rank,
            quant_config=load_config.quant_config,
            params_dtype=torch.bfloat16,
            prefix="projection",
            bias=True,
        )
        if config.kind == "qkv":
            self.projection = QKVParallelLinear(
                hidden_size=64, num_heads=8, num_kv_heads=2, head_dim=16, **common
            )
        elif config.kind == "merged":
            self.projection = MergedColumnParallelLinear(
                input_size=64,
                output_size=128,
                shard_names=["gate_proj", "up_proj"],
                **common,
            )
        elif config.kind == "row":
            self.projection = RowParallelLinear(input_size=64, output_size=32, **common)
        else:
            self.projection = ColumnParallelLinear(
                input_size=64, output_size=64, **common
            )


def _reference_swizzle(scale):
    rows, columns = scale.shape
    m, k = (rows + 127) // 128 * 128, (columns + 3) // 4 * 4
    packed = torch.zeros(m * k, dtype=torch.uint8)
    source = scale.contiguous().view(torch.uint8)
    for row in range(rows):
        for column in range(columns):
            offset = (
                ((row // 128 * (k // 4) + column // 4) * 32 + row % 32) * 4
                + row % 128 // 32
            ) * 4 + column % 4
            packed[offset] = source[row, column]
    return packed.reshape(m, k).view(torch.float8_e4m3fn)


class Fp4LinearTest(unittest.TestCase):
    def checkpoint(self, kind, fused=False):
        sizes = {
            "qkv": [("q_proj", 128), ("k_proj", 32), ("v_proj", 32)],
            "merged": [("gate_proj", 64), ("up_proj", 64)],
            "column": [("", 64)],
            "row": [("", 32)],
        }[kind]
        weights = {}
        for index, (name, rows) in enumerate(sizes):
            prefix = "projection." + (name + "." if name else "")
            weights[prefix + "weight"] = (
                (torch.arange(rows * 32) + 17 * index).to(torch.uint8).reshape(rows, 32)
            )
            weights[prefix + "weight_scale"] = (
                ((torch.arange(rows * 4) % 16 + 1 + index) / 16)
                .reshape(rows, 4)
                .to(torch.float8_e4m3fn)
            )
            weights[prefix + "weight_scale_2"] = torch.tensor([0.5 + index])
            weights[prefix + "input_scale"] = torch.tensor(1.25 + index)
            weights[prefix + "bias"] = (torch.arange(rows) + index).to(torch.bfloat16)
        if fused and len(sizes) > 1:
            combined = {}
            for parameter in (
                "weight",
                "weight_scale",
                "weight_scale_2",
                "input_scale",
                "bias",
            ):
                pieces = [
                    weights[f"projection.{name}.{parameter}"] for name, _ in sizes
                ]
                combined["projection." + parameter] = (
                    torch.stack(pieces).max()
                    if parameter in ("input_scale", "weight_scale_2")
                    else torch.cat(pieces)
                )
            return combined
        return weights

    def load(self, directory, kind, tp, rank):
        config = types.SimpleNamespace(model_type="fp4_dense_fixture", kind=kind)
        with mock.patch.dict(
            MODEL_REGISTRY, {"fp4_dense_fixture": _DenseCheckpointModel}
        ), mock.patch.object(
            ModelOptFp4LinearMethod, "validate_runtime_device", _cpu_layout_only
        ):
            return NewModelLoader(
                config,
                NewLoaderConfig(
                    device="cpu",
                    compute_dtype=torch.bfloat16,
                    load_method="scratch",
                    tp_size=tp,
                    tp_rank=rank,
                    quant_config=QuantizationConfig("modelopt_fp4"),
                ),
                model_path=directory,
            ).load()

    def expected(self, kind, tp, rank):
        source = self.checkpoint(kind)
        selected = {parameter: [] for parameter in ("weight", "weight_scale", "bias")}
        scales = {parameter: [] for parameter in ("weight_scale_2", "input_scale")}
        for name, tensor in source.items():
            parameter = name.rsplit(".", 1)[-1]
            if parameter in scales:
                scales[parameter].append(tensor.reshape(()))
                continue
            if kind == "row":
                piece = (
                    tensor
                    if parameter == "bias"
                    else tensor[
                        :,
                        rank
                        * (tensor.shape[1] // tp) : (rank + 1)
                        * (tensor.shape[1] // tp),
                    ]
                )
            elif kind == "qkv" and ("k_proj" in name or "v_proj" in name):
                shards = min(2, tp)
                shard_rank = rank // (tp // shards)
                width = tensor.shape[0] // shards
                piece = tensor[shard_rank * width : (shard_rank + 1) * width]
            else:
                width = tensor.shape[0] // tp
                piece = tensor[rank * width : (rank + 1) * width]
            selected[parameter].append(piece)
        expected = {name: torch.cat(pieces) for name, pieces in selected.items()}
        expected["weight_scale"] = _reference_swizzle(expected["weight_scale"])
        expected.update(
            {name: torch.stack(values).max() for name, values in scales.items()}
        )
        return expected

    def test_real_safetensors_loads_dense_modules_with_tp_fusion_and_gqa(self):
        for kind in ("column", "row", "merged", "qkv"):
            for fused in (False, True):
                with tempfile.TemporaryDirectory() as folder:
                    save_file(
                        self.checkpoint(kind, fused),
                        os.path.join(folder, "model.safetensors"),
                    )
                    for tp in (1, 2, 4):
                        for rank in range(tp):
                            with self.subTest(kind=kind, fused=fused, tp=tp, rank=rank):
                                layer = self.load(folder, kind, tp, rank).projection
                                for name, expected in self.expected(
                                    kind, tp, rank
                                ).items():
                                    actual = getattr(layer, name).detach()
                                    self.assertEqual(actual.shape, expected.shape)
                                    self.assertEqual(actual.dtype, expected.dtype)
                                    torch.testing.assert_close(
                                        actual.contiguous()
                                        .reshape(-1)
                                        .view(torch.uint8),
                                        expected.contiguous()
                                        .reshape(-1)
                                        .view(torch.uint8),
                                        rtol=0,
                                        atol=0,
                                    )
                                self.assertIs(
                                    layer.quant_method._executor.weights["weight"],
                                    layer.weight,
                                )
                                self.assertIs(
                                    layer.quant_method._executor.weights[
                                        "weight_scales"
                                    ],
                                    layer.weight_scale,
                                )

    def test_streaming_order_does_not_change_scalar_fusion(self):
        weights = self.checkpoint("qkv")
        common = dict(
            hidden_size=64,
            num_heads=8,
            num_kv_heads=2,
            head_dim=16,
            tp_size=4,
            tp_rank=3,
            quant_config=QuantizationConfig("modelopt_fp4"),
            params_dtype=torch.bfloat16,
            bias=True,
        )
        layer = QKVParallelLinear(**common)
        for name, value in reversed(list(weights.items())):
            layer.load_weights({name.removeprefix("projection."): value})
        layer.validate_weights_loaded()
        self.assertEqual(layer.weight_scale_2.item(), 2.5)
        self.assertEqual(layer.input_scale.item(), 3.25)

    def test_missing_or_wrong_packed_tensor_does_not_complete_loading(self):
        for kind in ("column", "row", "qkv", "merged"):
            weights = self.checkpoint(kind)
            weight_key = next(name for name in weights if name.endswith(".weight"))
            scale_key = next(name for name in weights if name.endswith(".input_scale"))
            cases = [
                (
                    {
                        name: value
                        for name, value in weights.items()
                        if name != scale_key
                    },
                    RuntimeError,
                ),
                ({**weights, weight_key: weights[weight_key].float()}, TypeError),
                ({**weights, scale_key: torch.tensor(float("nan"))}, ValueError),
                (
                    {**weights, weight_key: weights[weight_key][:, :-1].contiguous()},
                    ValueError,
                ),
            ]
            for values, error in cases:
                with self.subTest(
                    kind=kind, error=error
                ), tempfile.TemporaryDirectory() as folder:
                    save_file(values, os.path.join(folder, "model.safetensors"))
                    with self.assertRaises(error):
                        self.load(folder, kind, 2, 0)

    def test_duplicate_fused_and_separate_projection_is_rejected(self):
        layer = QKVParallelLinear(
            hidden_size=64,
            num_heads=8,
            num_kv_heads=2,
            head_dim=16,
            quant_config=QuantizationConfig("modelopt_fp4"),
            params_dtype=torch.bfloat16,
        )
        layer.load_weights(
            {"weight": self.checkpoint("qkv", True)["projection.weight"]}
        )
        with self.assertRaisesRegex(RuntimeError, "Duplicate"):
            layer.load_weights(
                {"q_proj.weight": self.checkpoint("qkv")["projection.q_proj.weight"]}
            )

    def test_row_shard_alignment_is_checked_before_loading(self):
        with self.assertRaisesRegex(ValueError, "16-value"):
            RowParallelLinear(
                input_size=48,
                output_size=32,
                tp_size=2,
                quant_config=QuantizationConfig("modelopt_fp4"),
                params_dtype=torch.bfloat16,
            )

    def test_trtllm_fp16_is_rejected_before_loading(self):
        with mock.patch.dict(os.environ, {"RTP_LLM_FP4_GEMM_BACKEND": "trtllm"}):
            with self.assertRaisesRegex(ValueError, "bfloat16 compute"):
                ColumnParallelLinear(
                    input_size=64,
                    output_size=32,
                    quant_config=QuantizationConfig("modelopt_fp4"),
                    params_dtype=torch.float16,
                )

    def test_scale_swizzle_preserves_bits_and_zero_padding(self):
        for rows, columns in ((1, 1), (128, 4), (129, 5)):
            source = (
                (torch.arange(rows * columns) % 64)
                .reshape(rows, columns)
                .to(torch.float8_e4m3fn)
            )
            actual = swizzle_nvfp4_scales(source)
            torch.testing.assert_close(
                actual.view(torch.uint8),
                _reference_swizzle(source).view(torch.uint8),
                rtol=0,
                atol=0,
            )
            self.assertEqual(actual.device, source.device)

    def test_backend_preflight_rejects_cpu_and_h20(self):
        method = ModelOptFp4LinearMethod(QuantizationConfig("modelopt_fp4"))
        with self.assertRaisesRegex(RuntimeError, "CUDA accelerator"):
            method.validate_runtime_device(torch.device("cpu"))
        with mock.patch.object(
            torch.cuda, "is_available", return_value=True
        ), mock.patch.object(
            torch.cuda, "get_device_capability", return_value=(9, 0)
        ), mock.patch.object(
            torch.version, "hip", None
        ):
            with self.assertRaisesRegex(RuntimeError, "SM100"):
                method.validate_runtime_device(torch.device("cuda:0"))

    def test_backend_preflight_checks_import_and_required_symbols(self):
        method = ModelOptFp4LinearMethod(QuantizationConfig("modelopt_fp4"))
        with mock.patch.object(
            torch.cuda, "is_available", return_value=True
        ), mock.patch.object(
            torch.cuda, "get_device_capability", return_value=(10, 0)
        ), mock.patch.object(
            torch.version, "hip", None
        ), mock.patch.dict(
            os.environ, {"RTP_LLM_FP4_GEMM_BACKEND": "cutlass"}
        ):
            with mock.patch(
                "rtp_llm.models_py.quant_methods.fp4.importlib.import_module",
                side_effect=ImportError("ABI mismatch"),
            ):
                with self.assertRaisesRegex(RuntimeError, "loading-time validation"):
                    method.validate_runtime_device(torch.device("cuda:0"))
            with mock.patch(
                "rtp_llm.models_py.quant_methods.fp4.importlib.import_module",
                return_value=types.SimpleNamespace(fp4_quantize=lambda: None),
            ):
                with self.assertRaisesRegex(RuntimeError, "mm_fp4"):
                    method.validate_runtime_device(torch.device("cuda:0"))
            with mock.patch.dict(
                os.environ, {"RTP_LLM_FP4_GEMM_BACKEND": "sgl_cutlass"}
            ), mock.patch(
                "rtp_llm.models_py.quant_methods.fp4.importlib.import_module",
                side_effect=[
                    types.SimpleNamespace(
                        fp4_quantize=lambda: None, mm_fp4=lambda: None
                    ),
                    types.SimpleNamespace(CudaFp4GEMMLinear=_CaptureExecutor),
                ],
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "cutlass_scaled_fp4_mm_wrapper"
                ):
                    method.validate_runtime_device(torch.device("cuda:0"))


if __name__ == "__main__":
    unittest.main()
