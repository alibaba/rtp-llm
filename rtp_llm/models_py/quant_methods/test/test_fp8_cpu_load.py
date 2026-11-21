"""Real scratch/FST CPU preparation, not a GPU forward or golden test."""

import math
import struct
import tempfile
import types
import unittest
from unittest import mock

import torch
from safetensors.torch import save_file

from rtp_llm.config.quant_config import Fp8DynamicPerTensorQuantConfig
from rtp_llm.models_py.layers.linear import ColumnParallelLinear
from rtp_llm.models_py.layers.moe_experts import BaseMoEExperts
from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.quant_methods.base import QuantizationConfig
from rtp_llm.models_py.quant_methods.fp8_cpu import \
    quantize_weight_per_tensor_cpu
from rtp_llm.models_py.registry import MODEL_REGISTRY


def reference_quantize(weight):
    """Independent scalar IEEE-FP32 and E4M3FN nearest-even reference.

    No production quantizer, legacy loader, native library or FP8 conversion op
    is used to generate expected bytes for checkpoint fixtures.
    """

    def f32(value):
        return struct.unpack("<f", struct.pack("<f", value))[0]

    values = weight.flatten().tolist()
    scale = max(f32(max(abs(v) for v in values) / 448), f32(1 / (512 * 448)))
    positive = [
        (byte & 7) / 512 if byte < 8 else (1 + (byte & 7) / 8) * 2 ** ((byte >> 3) - 7)
        for byte in range(127)
    ]
    encoded = []
    for value in values:
        target = min(abs(f32(value / scale)), 448)
        byte = min(range(127), key=lambda b: (abs(positive[b] - target), b & 1))
        encoded.append(byte | (128 if math.copysign(1, value) < 0 else 0))
    return torch.tensor(encoded, dtype=torch.uint8).reshape(weight.shape).view(
        torch.float8_e4m3fn
    ), torch.tensor(scale, dtype=torch.float32)


class Tiny(RtpModule):
    supports_expert_parallel = True

    def __init__(self, config, loading):
        super().__init__()
        self.dense = ColumnParallelLinear(
            8, 8, quant_config=loading.quant_config, params_dtype=torch.bfloat16
        )
        self.ordinary = ColumnParallelLinear(8, 8, params_dtype=torch.bfloat16)
        self.block = ColumnParallelLinear(
            8,
            8,
            quant_config=QuantizationConfig("fp8_block_online"),
            params_dtype=torch.bfloat16,
        )
        parallel = types.SimpleNamespace(dp_size=1, dp_rank=0)
        self.experts = BaseMoEExperts(
            num_experts=4,
            top_k=1,
            hidden_size=8,
            moe_intermediate_size=8,
            tp_size=1,
            tp_rank=0,
            ep_size=loading.ep_size,
            ep_rank=loading.ep_rank,
            params_dtype=torch.bfloat16,
            model_config=config,
            parallelism_config=parallel,
            moe_config=None,
            quant_config=loading.quant_config,
            layer_idx=0,
            prefix="experts",
        )


class CpuPhaseComplete(Exception):
    pass


def source_weights():
    # A small maximum exercises the main CPU scale floor. Every expert and
    # projection has a distinct pattern; up/gate order cannot accidentally pass.
    base = (torch.arange(64).reshape(8, 8) - 32).float() / 65536
    values = {
        name + ".weight": base.to(torch.bfloat16)
        for name in ("dense", "ordinary", "block")
    }
    for expert in range(4):
        for i, name in enumerate(("gate_proj", "up_proj", "down_proj")):
            values[f"experts.{expert}.{name}.weight"] = (
                base * (expert + 1) + i / 65536
            ).to(torch.bfloat16)
    return values


class CpuFp8LoadTest(unittest.TestCase):
    def test_fixed_main_bf16_vectors_exact_bytes_and_scale(self):
        # Captured with actual main f559bb4 quantize_weight_to_fp8 on CPU.
        # First vector covers the scale floor and signed zero; second covers a
        # BF16 midpoint where division differs from reciprocal multiplication.
        vectors = (
            (
                [
                    0,
                    32768,
                    14979,
                    47747,
                    14546,
                    47314,
                    14674,
                    47442,
                    12425,
                    45193,
                    12567,
                    45335,
                ],
                [0, 128, 118, 246, 91, 219, 99, 227, 0, 128, 0, 128],
                915556645,
            ),
            (
                [
                    0,
                    32768,
                    15821,
                    48589,
                    13773,
                    46541,
                    15173,
                    47941,
                    15396,
                    48164,
                    15693,
                    48461,
                ],
                [0, 128, 126, 254, 4, 132, 85, 213, 99, 227, 118, 246],
                963266853,
            ),
        )
        for source_bits, expected_bytes, scale_bits in vectors:
            weight = (
                torch.tensor(source_bits, dtype=torch.uint16)
                .view(torch.bfloat16)
                .reshape(3, 4)
            )
            for quantizer in (quantize_weight_per_tensor_cpu, reference_quantize):
                actual, scale = quantizer(weight)
                self.assertEqual(
                    actual.view(torch.uint8).flatten().tolist(), expected_bytes
                )
                self.assertEqual(scale.view(torch.int32).item(), scale_bits)

    def test_independent_reference_exact_bytes_and_scale(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for maximum in (0, 0.001, 0.1, 1):
                weight = (
                    torch.linspace(-maximum, maximum, 256).reshape(16, 16).to(dtype)
                )
                expected, scale = reference_quantize(weight)
                actual, actual_scale = quantize_weight_per_tensor_cpu(weight)
                torch.testing.assert_close(
                    actual.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0
                )
                torch.testing.assert_close(actual_scale, scale, rtol=0, atol=0)

    def stage(self, method, force, rank=0):
        model = None

        def stop_before_runtime(loaded, device):
            nonlocal model
            model = loaded
            raise CpuPhaseComplete

        with tempfile.TemporaryDirectory() as directory:
            values = source_weights()
            save_file(
                {k: v for i, (k, v) in enumerate(values.items()) if i % 2 == 0},
                directory + "/one.safetensors",
            )
            save_file(
                {k: v for i, (k, v) in enumerate(values.items()) if i % 2 == 1},
                directory + "/two.safetensors",
            )
            cfg = types.SimpleNamespace(
                model_type="_tiny_cpu_fp8_stage",
                model_path=directory,
                activation_type="SiGLU",
                expert_num=4,
                moe_layer_index=[0],
                num_layers=1,
                quant_config=Fp8DynamicPerTensorQuantConfig(is_quanted=False),
            )
            loading = NewLoaderConfig(
                device="cpu",
                load_method=method,
                force_cpu_load_weights=force,
                ep_size=2,
                ep_rank=rank,
                compute_dtype=torch.bfloat16,
                quant_config=QuantizationConfig("fp8_online"),
            )
            loader = NewModelLoader(cfg, loading, model_path=directory)
            # Deliberately stop after real CPU reading/dispatch/conversion. No
            # pretend GPU backend, migration, or forward result is produced.
            with mock.patch.dict(
                MODEL_REGISTRY, {"_tiny_cpu_fp8_stage": Tiny}
            ), mock.patch.object(
                loader, "_validate_runtime_backends"
            ), mock.patch.object(
                loader, "_migrate_staged_modules", side_effect=stop_before_runtime
            ):
                with self.assertRaises(CpuPhaseComplete):
                    loader.load()
        return model, values

    def test_real_scratch_and_native_fst_cpu_stage_matches_main(self):
        for method in ("scratch", "fastsafetensors"):
            for rank in (0, 1):
                with self.subTest(method=method, ep_rank=rank):
                    model, values = self.stage(method, True, rank)
                    expected, scale = reference_quantize(values["dense.weight"])
                    torch.testing.assert_close(
                        model.dense.weight.view(torch.uint8),
                        expected.view(torch.uint8),
                        rtol=0,
                        atol=0,
                    )
                    torch.testing.assert_close(
                        model.dense.weight_scale.reshape(()), scale, rtol=0, atol=0
                    )
                    for local, logical in enumerate(range(rank * 2, rank * 2 + 2)):
                        prefix = f"experts.{logical}."
                        upgate = torch.cat(
                            [
                                values[prefix + "up_proj.weight"],
                                values[prefix + "gate_proj.weight"],
                            ]
                        )
                        for name, source in (
                            ("w13", upgate),
                            ("w2", values[prefix + "down_proj.weight"]),
                        ):
                            expected, scale = reference_quantize(source)
                            torch.testing.assert_close(
                                getattr(model.experts, name)[local].view(torch.uint8),
                                expected.view(torch.uint8),
                                rtol=0,
                                atol=0,
                            )
                            torch.testing.assert_close(
                                getattr(model.experts, name + "_scale")[local],
                                scale,
                                rtol=0,
                                atol=0,
                            )
                    for leaf in (model.block, model.ordinary):
                        self.assertEqual(leaf.weight.dtype, torch.bfloat16)
                        self.assertFalse(hasattr(leaf, "_fp8_cpu_weight_prepared"))

    def test_force_false_keeps_existing_online_device_quantization(self):
        model, values = self.stage("scratch", False)
        self.assertEqual(model.dense.weight.dtype, torch.bfloat16)
        self.assertEqual(model.experts.w13.dtype, torch.bfloat16)
        self.assertFalse(hasattr(model.dense, "_fp8_cpu_weight_prepared"))
        self.assertFalse(hasattr(model.experts, "_fp8_cpu_weight_prepared"))

    def test_prepared_cpu_quantization_is_idempotent(self):
        model, _ = self.stage("scratch", True)
        for leaf in (model.dense, model.experts):
            pointers = {name: p.data_ptr() for name, p in leaf.named_parameters()}
            leaf.quant_method.prepare_weights_on_cpu(leaf)
            self.assertEqual(
                pointers, {name: p.data_ptr() for name, p in leaf.named_parameters()}
            )

    def test_prepared_weights_are_not_quantized_twice_at_runtime_transition(self):
        model, _ = self.stage("scratch", True)
        # Contract-only transition test, not a GPU execution result. The actual
        # method conversion runs; only native backend selection is isolated.
        with mock.patch(
            "rtp_llm.models_py.quant_methods.fp8._select_fp8_runtime_backend",
            return_value="cuda_scaled_mm",
        ), mock.patch(
            "rtp_llm.models_py.quant_methods.fp8._resolve_per_tensor_quant",
            side_effect=AssertionError("must not requantize FP8"),
        ):
            for leaf in (model.dense, model.experts):
                before = {
                    name: p.detach().clone() for name, p in leaf.named_parameters()
                }
                leaf.quant_method.process_weights_after_loading(leaf)
                for name, param in leaf.named_parameters():
                    original = before[name]
                    if original.dtype == torch.float8_e4m3fn:
                        torch.testing.assert_close(
                            param.view(torch.uint8),
                            original.view(torch.uint8),
                            rtol=0,
                            atol=0,
                        )
                    else:
                        torch.testing.assert_close(param, original, rtol=0, atol=0)
                self.assertFalse(hasattr(leaf, "_fp8_cpu_weight_prepared"))


if __name__ == "__main__":
    unittest.main()
