"""CPU checkpoint contracts. Backend doubles here do not claim GPU coverage."""

import os
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace as NS
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
from rtp_llm.models_py.quant_methods import fp8
from rtp_llm.models_py.quant_methods.base import QuantizationConfig
from rtp_llm.models_py.registry import register_model


def config(online=False, dynamic=True):
    return QuantizationConfig(
        "fp8_online" if online else "fp8",
        source_config=NS(is_dynamic=lambda: dynamic),
    )


def reference_quantize(weight, dtype, scale=None):
    maximum = torch.finfo(dtype).max
    if scale is None:
        scale = weight.float().abs().amax().clamp_min(1e-12).reshape(1) / maximum
    return (weight.float() / scale).clamp(-maximum, maximum).to(dtype), scale


@register_model("fp8_per_tensor_contract_test")
class TinyFp8Model(RtpModule):
    def __init__(self, model_config, load_config):
        super().__init__()
        self.proj = ColumnParallelLinear(
            16,
            16,
            quant_config=load_config.quant_config,
            params_dtype=load_config.compute_dtype,
            prefix="proj",
        )

    def forward(self, x):
        return self.proj(x)


def load_tiny(weights, quant_config, device="cpu", compute_dtype=torch.bfloat16):
    with tempfile.TemporaryDirectory() as path:
        save_file(weights, os.path.join(path, "model.safetensors"))
        return NewModelLoader(
            NS(model_type="fp8_per_tensor_contract_test"),
            NewLoaderConfig(
                device=device, compute_dtype=compute_dtype, quant_config=quant_config
            ),
            model_path=path,
        ).load()


class Fp8PerTensorTest(unittest.TestCase):
    def test_import_is_lazy_in_fresh_process(self):
        script = """
import importlib.abc
import sys
import torch
class RejectBackend(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('aiter', 'deep_gemm', 'deep_ep', 'flashinfer', 'librtp_compute_ops') or fullname == 'rtp_llm.ops':
            raise AssertionError('eager backend import: ' + fullname)
sys.meta_path.insert(0, RejectBackend())
from rtp_llm.models_py.layers.linear import ColumnParallelLinear
from rtp_llm.models_py.quant_methods.base import QuantizationConfig
layer = ColumnParallelLinear(16, 16, quant_config=QuantizationConfig('fp8'), params_dtype=torch.bfloat16)
assert layer.weight.dtype == torch.float8_e4m3fn
"""
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=60
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_full_online_load_converts_once_across_both_stages(self):
        weight = torch.arange(256, dtype=torch.bfloat16).reshape(16, 16) / 64
        with mock.patch.object(
            fp8, "_runtime_dtype", return_value=torch.float8_e4m3fn
        ), mock.patch.object(
            fp8, "_quantize", side_effect=reference_quantize
        ) as convert:
            model = load_tiny({"proj.weight": weight}, config(online=True))
            NewModelLoader._run_post_load_hooks(model)
        self.assertEqual(convert.call_count, 1)
        expected, scale = reference_quantize(weight, torch.float8_e4m3fn)
        torch.testing.assert_close(model.proj.weight.float(), expected.float())
        torch.testing.assert_close(model.proj.weight_scale, scale)
        model.validate_weights_loaded()
        self.assertFalse(model.training)

    def test_prequantized_full_load_dynamic_and_static(self):
        weight = torch.arange(256).reshape(16, 16).remainder(7).to(torch.float8_e4m3fn)
        for dynamic in (True, False):
            with self.subTest(dynamic=dynamic):
                weights = {
                    "proj.weight": weight,
                    "proj.weight_scale": torch.tensor([0.25]),
                }
                if not dynamic:
                    weights["proj.input_scale"] = torch.tensor([0.5])
                with mock.patch.object(
                    fp8, "_runtime_dtype", return_value=torch.float8_e4m3fn
                ), mock.patch.object(fp8, "_quantize") as quantize:
                    model = load_tiny(weights, config(dynamic=dynamic))
                quantize.assert_not_called()
                torch.testing.assert_close(model.proj.weight.float(), weight.float())
                torch.testing.assert_close(
                    model.proj.weight_scale, torch.tensor([0.25])
                )

    def test_missing_and_invalid_scales_fail(self):
        weight = torch.ones(16, 16).to(torch.float8_e4m3fn)
        for missing in ("weight_scale", "input_scale"):
            weights = {
                "proj.weight": weight,
                "proj.weight_scale": torch.ones(1),
                "proj.input_scale": torch.ones(1),
            }
            del weights["proj." + missing]
            with self.subTest(missing=missing), self.assertRaisesRegex(
                RuntimeError, missing
            ):
                load_tiny(weights, config(dynamic=False))
        for name in ("weight_scale", "input_scale"):
            for value in (0.0, -1.0, float("nan"), float("inf")):
                weights = {
                    "proj.weight": weight,
                    "proj.weight_scale": torch.ones(1),
                    "proj.input_scale": torch.ones(1),
                }
                weights["proj." + name] = torch.tensor([value])
                with self.subTest(name=name, value=value), mock.patch.object(
                    fp8, "_runtime_dtype", return_value=torch.float8_e4m3fn
                ), self.assertRaisesRegex(ValueError, "finite positive"):
                    load_tiny(weights, config(dynamic=False))

    def test_rocm_fnuz_conversion_preserves_values_and_source(self):
        source = torch.tensor([-128, 0, 56, -72], dtype=torch.int8).view(
            torch.float8_e4m3fn
        )
        before = source.view(torch.int8).clone()
        weight, scale = fp8._convert_checkpoint(
            source, torch.tensor([0.25]), torch.float8_e4m3fnuz
        )
        torch.testing.assert_close(weight.float() * scale, source.float() * 0.25)
        torch.testing.assert_close(source.view(torch.int8), before)
        self.assertTrue(torch.isfinite(weight.float()).all())

    def test_rocm_static_scale_adjusted_once(self):
        weights = {
            "proj.weight": torch.ones(16, 16).to(torch.float8_e4m3fn),
            "proj.weight_scale": torch.tensor([0.25]),
            "proj.input_scale": torch.tensor([0.5]),
        }
        with mock.patch.object(
            fp8, "_runtime_dtype", return_value=torch.float8_e4m3fnuz
        ):
            model = load_tiny(weights, config(dynamic=False))
            NewModelLoader._run_post_load_hooks(model)
        torch.testing.assert_close(model.proj.weight_scale, torch.tensor([0.5]))
        torch.testing.assert_close(model.proj.input_scale, torch.tensor([1.0]))

    def test_merged_and_qkv_shards_keep_scaled_values_on_each_tp_rank(self):
        for rank in (0, 1):
            for qkv in (False, True):
                with self.subTest(rank=rank, qkv=qkv):
                    if qkv:
                        layer = QKVParallelLinear(
                            32,
                            4,
                            2,
                            16,
                            tp_size=2,
                            tp_rank=rank,
                            quant_config=config(dynamic=False),
                            params_dtype=torch.bfloat16,
                        )
                        projections = [("q_proj", 64), ("k_proj", 32), ("v_proj", 32)]
                    else:
                        layer = MergedColumnParallelLinear(
                            32,
                            128,
                            tp_size=2,
                            tp_rank=rank,
                            shard_names=["gate_proj", "up_proj"],
                            quant_config=config(dynamic=False),
                            params_dtype=torch.bfloat16,
                        )
                        projections = [("gate_proj", 64), ("up_proj", 64)]
                    expected = []
                    for index, (name, rows) in enumerate(projections):
                        scale = float(2**index)
                        weight = (
                            torch.arange(rows * 32)
                            .reshape(rows, 32)
                            .remainder(7)
                            .to(torch.float8_e4m3fn)
                        )
                        expected.append(weight.float().chunk(2, dim=0)[rank] * scale)
                        for suffix, tensor in (
                            ("weight_scale", torch.tensor([scale])),
                            ("weight", weight),
                            ("input_scale", torch.tensor([scale * 0.5])),
                        ):
                            layer.load_weights({name + "." + suffix: tensor})
                    layer.validate_weights_loaded()
                    with mock.patch.object(
                        fp8, "_runtime_dtype", return_value=torch.float8_e4m3fn
                    ):
                        layer.process_weights_after_loading()
                        layer.process_weights_after_loading()
                    torch.testing.assert_close(
                        layer.weight.float() * layer.weight_scale, torch.cat(expected)
                    )
                    torch.testing.assert_close(
                        layer.input_scale,
                        torch.tensor([2 ** (len(projections) - 2)]).float(),
                    )

    def test_row_sharding_and_exclusion(self):
        weight = (
            torch.arange(32 * 32).reshape(32, 32).remainder(5).to(torch.float8_e4m3fn)
        )
        for rank in (0, 1):
            layer = RowParallelLinear(
                32,
                32,
                tp_size=2,
                tp_rank=rank,
                quant_config=config(),
                params_dtype=torch.bfloat16,
            )
            layer.load_weights({"weight": weight, "weight_scale": torch.tensor([0.25])})
            layer.validate_weights_loaded()
            torch.testing.assert_close(
                layer.weight.float(), weight.float().chunk(2, dim=1)[rank]
            )
        qc = QuantizationConfig("fp8", ignored_layers=["proj"])
        layer = ColumnParallelLinear(
            16, 16, quant_config=qc, prefix="proj", params_dtype=torch.bfloat16
        )
        self.assertEqual(layer.weight.dtype, torch.bfloat16)

    def test_backend_failure_precedes_migration(self):
        # The real backend preflight rejects CPU before any .to() call.
        weights = {"proj.weight": torch.ones(16, 16, dtype=torch.bfloat16)}
        with mock.patch.object(
            TinyFp8Model, "to", side_effect=AssertionError("must not migrate")
        ) as migrate, self.assertRaisesRegex(RuntimeError, "accelerator"):
            load_tiny(weights, config(online=True))
        migrate.assert_not_called()

    def test_missing_native_symbols_are_not_cached_as_success(self):
        # Inject before the resolver's first import, not after backend loading.
        fp8._quant_ops.cache_clear()
        try:
            with mock.patch.object(fp8.importlib, "import_module", return_value=NS()):
                with self.assertRaisesRegex(ImportError, "dynamic_per_tensor_quant"):
                    fp8._quant_ops(True)
            dynamic, static = mock.Mock(), mock.Mock()
            with mock.patch.object(
                fp8.importlib,
                "import_module",
                return_value=NS(
                    dynamic_per_tensor_quant=dynamic, static_per_tensor_quant=static
                ),
            ):
                self.assertEqual(fp8._quant_ops(True), (dynamic, static))
        finally:
            fp8._quant_ops.cache_clear()

    def test_layout_and_unsupported_methods_fail_explicitly(self):
        with self.assertRaisesRegex(TypeError, "E4M3 dtype"):
            fp8._quantize(torch.ones(1, 16, dtype=torch.bfloat16), torch.bfloat16)
        for method in (
            "fp8",
            "FP8_PER_TENSOR_COMPRESSED",
            "fp8_online",
            "FP8_DYNAMIC_PER_TENSOR",
        ):
            layer = ColumnParallelLinear(
                16, 16, quant_config=QuantizationConfig(method)
            )
            self.assertIn(
                type(layer.quant_method),
                (fp8.Fp8LinearMethod, fp8.Fp8OnlineLinearMethod),
            )
        with self.assertRaisesRegex(ValueError, "divisible by 16"):
            ColumnParallelLinear(17, 16, quant_config=config())
        for method in ("fp8_block", "fp8_per_channel"):
            with self.assertRaisesRegex(ValueError, "not supported"):
                ColumnParallelLinear(16, 16, quant_config=QuantizationConfig(method))
        with self.assertRaisesRegex(ValueError, "dynamic activation"):
            ColumnParallelLinear(
                16, 16, quant_config=config(online=True, dynamic=False)
            )


class Fp8BackendAdmissionTest(unittest.TestCase):
    def test_platform_and_explicit_device_admission(self):
        cases = [
            (None, (8, 0), "", None),
            (None, (9, 0), "", torch.float8_e4m3fn),
            ("7.2", (9, 0), "gfx942:sramecc+", torch.float8_e4m3fnuz),
            ("7.2", (9, 0), "gfx950", torch.float8_e4m3fn),
            ("7.2", (9, 0), "gfx90a", None),
        ]
        for hip, capability, arch, expected in cases:
            with self.subTest(
                hip=hip, arch=arch, capability=capability
            ), mock.patch.object(torch.version, "hip", hip), mock.patch.object(
                torch.cuda, "is_available", return_value=True
            ), mock.patch.object(
                torch.cuda, "get_device_capability", return_value=capability
            ) as get_capability, mock.patch.object(
                torch.cuda, "get_device_properties", return_value=NS(gcnArchName=arch)
            ) as get_properties, mock.patch.object(
                fp8, "_quant_ops", return_value=(mock.Mock(), mock.Mock())
            ):
                if expected is None:
                    with self.assertRaises(RuntimeError):
                        fp8._runtime_dtype(torch.device("cuda:3"))
                else:
                    self.assertEqual(
                        fp8._runtime_dtype(torch.device("cuda:3")), expected
                    )
                (get_properties if hip else get_capability).assert_called_once_with(3)

    def test_backend_import_error_retains_cause(self):
        error = ImportError("missing native symbol")
        with mock.patch.object(torch.version, "hip", None), mock.patch.object(
            torch.cuda, "is_available", return_value=True
        ), mock.patch.object(
            torch.cuda, "get_device_capability", return_value=(9, 0)
        ), mock.patch.object(
            fp8, "_quant_ops", side_effect=error
        ):
            with self.assertRaisesRegex(
                RuntimeError, "backend is unavailable"
            ) as raised:
                fp8._runtime_dtype(torch.device("cuda:0"))
        self.assertIs(raised.exception.__cause__, error)


if __name__ == "__main__":
    unittest.main()
