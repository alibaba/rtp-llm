"""CPU checkpoint/layout regressions; not a CUDA executor or distributed test."""

import os
import tempfile
import types
import unittest
from unittest import mock

import torch
from safetensors.torch import save_file

from rtp_llm.config.quant_config import ModelOptFp4Config
from rtp_llm.models_py.layers.moe_experts import BaseMoEExperts
from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.quant_methods.base import QuantizationConfig
from rtp_llm.models_py.quant_methods.fp4_moe import ModelOptFp4MoEMethod
from rtp_llm.models_py.quant_methods.moe_quant_key import moe_checkpoint_method_key
from rtp_llm.models_py.quant_methods.nvfp4_quantization import quantize_expert_to_nvfp4
from rtp_llm.models_py.quant_methods.unquantized import UnquantizedLinearMethod
from rtp_llm.models_py.registry import MODEL_REGISTRY
from rtp_llm.utils.model_weight import W


def _experts(
    tp_rank=0,
    tp_size=1,
    ep_rank=0,
    ep_size=1,
    online=False,
    redundant=0,
    backend="cutedsl",
    moe_strategy="auto",
    model_type="qwen3_moe",
    **source_kwargs,
):
    source = ModelOptFp4Config(
        bits=4, group_size=16, is_quanted=not online, **source_kwargs
    )
    runtime = QuantizationConfig(
        moe_checkpoint_method_key(source), source_config=source
    )
    config = types.SimpleNamespace(
        model_type=model_type,
        activation_type="SiGLU",
        quant_config=source,
        phy2log_path="",
        num_layers=1,
        eplb_config=types.SimpleNamespace(
            redundant_expert=redundant,
            phy_exp_num=lambda n: n + redundant,
            enable_eplb=False,
        ),
    )
    return BaseMoEExperts(
        num_experts=2,
        top_k=1,
        hidden_size=64,
        moe_intermediate_size=64,
        tp_size=tp_size,
        tp_rank=tp_rank,
        ep_size=ep_size,
        ep_rank=ep_rank,
        params_dtype=torch.bfloat16,
        model_config=config,
        parallelism_config=types.SimpleNamespace(dp_size=1, dp_rank=0),
        moe_config=types.SimpleNamespace(fp4_moe_op=backend, moe_strategy=moe_strategy),
        quant_config=runtime,
        layer_idx=0,
        prefix="experts",
    )


class _CheckpointModel(RtpModule):
    def __init__(self, config, load_config):
        super().__init__()
        self.experts = _experts(**config.expert_args)


def _weights(online=False):
    result = {}
    for expert in range(2):
        for index, projection in enumerate(("gate_proj", "up_proj", "down_proj")):
            stem = f"experts.{expert}.{projection}."
            if online:
                result[stem + "weight"] = (
                    torch.arange(4096).reshape(64, 64).float().sin()
                    * (expert + 1)
                    * (index + 1)
                ).bfloat16()
            else:
                result[stem + "weight"] = (
                    torch.arange(2048).reshape(64, 32) + expert * 7 + index
                ).to(torch.uint8)
                result[stem + "weight_scale"] = torch.full(
                    (64, 4), 1 + expert + index
                ).to(torch.float8_e4m3fn)
                result[stem + "weight_scale_2"] = torch.tensor(0.25 * (index + 1))
                result[stem + "input_scale"] = torch.tensor(0.5 * (index + 1))
    return result


def _cpu_no_executor(layer):
    # Production quant postprocessing is real; only GPU executor construction
    # is excluded from the CPU loading fixture.
    weights = {W.moe_w1: layer.w13.data, W.moe_w2: layer.w2.data}
    layer.quant_method.add_weight_tensors(layer, weights)
    layer._runtime_weights_dict = weights


class Fp4MoETest(unittest.TestCase):
    def test_online_allocates_packed_parameters_from_start(self):
        layer = _experts(online=True)
        self.assertEqual(layer.w13.dtype, torch.uint8)
        self.assertEqual(layer.w13.shape, (2, 128, 32))
        self.assertEqual(layer.w2.shape, (2, 64, 32))

    def test_fp8_source_and_explicit_activation_range(self):
        source, decoded = {}, {}
        for name, value in _weights(True).items():
            fp8 = value.to(torch.float8_e4m3fn)
            source[name] = fp8
            source[name + "_scale_inv"] = torch.tensor([[0.25]])
            decoded[name] = (fp8.float() * 0.25).bfloat16()
        for rank in range(2):
            actual = self._load(
                source,
                online=True,
                tp_size=2,
                tp_rank=rank,
                source_fp8_block_size=128,
                activation_amax=32,
            )
            expected = self._load(
                decoded, online=True, tp_size=2, tp_rank=rank, activation_amax=32
            )
            for name, value in expected.named_parameters():
                torch.testing.assert_close(dict(actual.named_parameters())[name], value)
            torch.testing.assert_close(
                actual.w13_input_scale, torch.full((2,), 32 / (6 * 448))
            )
            self.assertFalse(actual._nvfp4_pending_source)
            self.assertFalse(actual._nvfp4_pending_float)

    def test_fp8_source_missing_scale_fails(self):
        source = {
            name: value.to(torch.float8_e4m3fn)
            for name, value in _weights(True).items()
        }
        with self.assertRaisesRegex(RuntimeError, "missing required MoE"):
            self._load(source, online=True, source_fp8_block_size=128)

    def test_invalid_activation_range_fails(self):
        for value in (0, -1, float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "activation amax"):
                _experts(online=True, activation_amax=value)

    def test_fp8_source_scale_may_arrive_first(self):
        layer = _experts(online=True, source_fp8_block_size=128)
        weights = _weights(True)
        for name, value in weights.items():
            local = name.removeprefix("experts.")
            layer.load_weights({local + "_scale_inv": torch.tensor([[0.5]])})
            layer.load_weights({local: value.to(torch.float8_e4m3fn)})
        layer._check_load_complete()
        self.assertFalse(layer._nvfp4_pending_source)
        self.assertFalse(layer._nvfp4_pending_source_scale)
        self.assertFalse(layer._nvfp4_pending_float)

    def test_contiguous_recipe_rejects_redundant_placement(self):
        with self.assertRaisesRegex(ValueError, "contiguous logical"):
            _experts(
                online=True, ep_size=2, redundant=2, require_contiguous_experts=True
            )

    def test_fp8_source_wrong_scale_geometry_fails(self):
        layer = _experts(online=True, source_fp8_block_size=128)
        layer.load_weights({"0.gate_proj.weight_scale_inv": torch.ones(2, 2)})
        with self.assertRaisesRegex(ValueError, "geometry"):
            layer.load_weights(
                {"0.gate_proj.weight": torch.ones(64, 64).to(torch.float8_e4m3fn)}
            )

    def _load(self, weights, finalize_layout=True, **expert_args):
        with tempfile.TemporaryDirectory() as folder, mock.patch.dict(
            MODEL_REGISTRY, {"fp4_moe_fixture": _CheckpointModel}
        ):
            save_file(weights, os.path.join(folder, "model.safetensors"))
            loader = NewModelLoader(
                types.SimpleNamespace(
                    model_type="fp4_moe_fixture", expert_args=expert_args
                ),
                NewLoaderConfig(device="cpu", load_method="scratch"),
                model_path=folder,
            )
            with mock.patch.object(
                ModelOptFp4MoEMethod, "validate_runtime_device", lambda *args: None
            ):
                layer = loader.load().experts
            # CPU load is staging, not admission to a CUDA executor. Exercise
            # runtime layout/auxiliary-scale semantics explicitly via the same
            # executor-free preparation used for EPLB reloads.
            self.assertFalse(layer._post_load_done)
            self.assertIsNone(layer._runtime_weights_dict)
            self.assertIsNone(layer.fused_moe)
            if finalize_layout:
                with torch.inference_mode():
                    layer.prepare_eplb_reload_weights()
                self.assertTrue(layer._post_load_done)
                self.assertIsNone(layer.fused_moe)
            return layer

    def test_cpu_staging_does_not_finalize_or_publish_runtime_weights(self):
        weights = _weights()
        layer = self._load(weights, finalize_layout=False)
        for expert in range(2):
            torch.testing.assert_close(
                layer.w13[expert],
                torch.cat(
                    (
                        weights[f"experts.{expert}.up_proj.weight"],
                        weights[f"experts.{expert}.gate_proj.weight"],
                    )
                ),
            )
        with self.assertRaisesRegex(RuntimeError, "not initialized"):
            layer.eplb_runtime_weight_view()

    def test_real_checkpoint_packed_tp_ranks(self):
        weights = _weights()
        for rank in range(2):
            layer = self._load(weights, tp_size=2, tp_rank=rank)
            # CuteDSL postprocess restores [gate | up] order.
            for expert in range(2):
                gate = weights[f"experts.{expert}.gate_proj.weight"][
                    rank * 32 : (rank + 1) * 32
                ]
                up = weights[f"experts.{expert}.up_proj.weight"][
                    rank * 32 : (rank + 1) * 32
                ]
                torch.testing.assert_close(layer.w13[expert], torch.cat((gate, up)))
                down = weights[f"experts.{expert}.down_proj.weight"][
                    :, rank * 16 : (rank + 1) * 16
                ]
                torch.testing.assert_close(layer.w2[expert], down)
            self.assertEqual(layer.w13.dtype, torch.uint8)
            torch.testing.assert_close(layer.w13_scale_2, torch.full((2,), 0.5))
            torch.testing.assert_close(layer.w13_input_scale, torch.ones(2))

    def test_ep_and_redundant_placement(self):
        for rank in range(2):
            layer = self._load(_weights(), ep_size=2, ep_rank=rank, redundant=2)
            for logical in range(2):
                for local in layer._expert_placement.local_indices(logical):
                    source = _weights()[f"experts.{logical}.gate_proj.weight"]
                    torch.testing.assert_close(layer.w13[local, :64], source)

    def test_online_quantize_before_tp_bytes(self):
        weights = _weights(online=True)
        for rank in range(2):
            layer = self._load(weights, online=True, tp_size=2, tp_rank=rank)
            for expert in range(2):
                gate = weights[f"experts.{expert}.gate_proj.weight"]
                up = weights[f"experts.{expert}.up_proj.weight"]
                full, _, s2 = quantize_expert_to_nvfp4(torch.cat((gate, up)))
                expected = torch.cat(
                    (
                        full[rank * 32 : (rank + 1) * 32],
                        full[64 + rank * 32 : 64 + (rank + 1) * 32],
                    )
                )
                torch.testing.assert_close(layer.w13[expert], expected)
                torch.testing.assert_close(layer.w13_scale_2[expert], s2)
                full_down, _, _ = quantize_expert_to_nvfp4(
                    weights[f"experts.{expert}.down_proj.weight"]
                )
                torch.testing.assert_close(
                    layer.w2[expert], full_down[:, rank * 16 : (rank + 1) * 16]
                )

    def test_missing_scale_fails(self):
        weights = _weights()
        del weights["experts.0.gate_proj.weight_scale_2"]
        with self.assertRaisesRegex(RuntimeError, "missing required MoE"):
            self._load(weights)

    def test_stacked_checkpoint_offline_and_online(self):
        for online in (False, True):
            weights = _weights(online)
            expected = self._load(weights, online=online)
            stacked = {}
            names = (
                ("weight",)
                if online
                else ("weight", "weight_scale", "weight_scale_2", "input_scale")
            )
            for name in names:
                fused, down = [], []
                for expert in range(2):
                    gate = weights[f"experts.{expert}.gate_proj.{name}"]
                    up = weights[f"experts.{expert}.up_proj.{name}"]
                    if gate.ndim == 0:
                        fused.append(torch.stack((gate, up)))
                    else:
                        fused.append(
                            torch.cat(
                                (gate.view(torch.uint8), up.view(torch.uint8))
                            ).view(gate.dtype)
                        )
                    down.append(weights[f"experts.{expert}.down_proj.{name}"])
                stacked[f"experts.gate_up_proj.{name}"] = (
                    torch.stack(
                        [t.view(torch.uint8) if t.ndim else t for t in fused]
                    ).view(fused[0].dtype)
                    if name == "weight_scale"
                    else torch.stack(fused)
                )
                stacked[f"experts.down_proj.{name}"] = (
                    torch.stack(
                        [t.view(torch.uint8) if t.ndim else t for t in down]
                    ).view(down[0].dtype)
                    if name == "weight_scale"
                    else torch.stack(down)
                )
            actual = self._load(stacked, online=online)
            for name, expected_tensor in expected.named_parameters():
                torch.testing.assert_close(
                    dict(actual.named_parameters())[name], expected_tensor
                )
            for name, expected_tensor in expected.named_buffers():
                torch.testing.assert_close(
                    dict(actual.named_buffers())[name].reshape(-1).view(torch.uint8),
                    expected_tensor.reshape(-1).view(torch.uint8),
                )

    def test_backend_import_failure_is_load_time_error(self):
        method = _experts().quant_method
        with mock.patch("torch.version.hip", None), mock.patch(
            "torch.cuda.get_device_capability", return_value=(10, 0)
        ), mock.patch(
            "rtp_llm.models_py.quant_methods.fp4_moe.importlib.import_module",
            side_effect=ImportError("bad ABI"),
        ):
            with self.assertRaisesRegex(RuntimeError, "NVFP4 backend requires"):
                method.validate_runtime_device("cuda:0")

    def test_invalid_packed_dtype_fails(self):
        weights = _weights()
        weights["experts.0.gate_proj.weight"] = weights[
            "experts.0.gate_proj.weight"
        ].float()
        with self.assertRaisesRegex(TypeError, "dtype"):
            self._load(weights)

    def test_online_and_mixed_attention_dispatch(self):
        online = ModelOptFp4Config(bits=4, group_size=16, is_quanted=False)
        mixed = ModelOptFp4Config(
            bits=4, group_size=16, is_quanted=True, mixed_attention=True
        )
        for source in (online, mixed):
            runtime = QuantizationConfig(
                moe_checkpoint_method_key(source), source_config=source
            )
            for prefix in (
                "layers.0.self_attn.qkv_proj",
                "layers.0.linear_attn.in_proj_qkvz",
            ):
                self.assertIsInstance(
                    runtime.get_quant_method(types.SimpleNamespace(), prefix),
                    UnquantizedLinearMethod,
                )
        runtime = QuantizationConfig(
            moe_checkpoint_method_key(mixed), source_config=mixed
        )
        self.assertEqual(
            type(
                runtime.get_quant_method(
                    types.SimpleNamespace(), "layers.0.mlp.gate_proj"
                )
            ).__name__,
            "ModelOptFp4LinearMethod",
        )

    def test_eplb_companion_scales_are_validated(self):
        layer = self._load(_weights())
        weights = layer.eplb_runtime_weight_view()
        for name in layer.quant_method.eplb_aux_weight_names():
            missing = dict(weights)
            del missing[name]
            with self.assertRaisesRegex(ValueError, "Missing EPLB"):
                layer.validate_eplb_reload_weights(missing)

    def test_eplb_scales_follow_backend_activation_granularity(self):
        for backend in ("trtllm", "cutedsl"):
            source = _weights()
            for projection in ("gate_proj", "up_proj"):
                source[f"experts.1.{projection}.input_scale"] = torch.tensor(4.0)
            # Permuting packed bytes needs FlashInfer; scale derivation is real.
            with mock.patch(
                "rtp_llm.models_py.quant_methods.fp4_moe._prepare_trtllm",
                side_effect=lambda weight, scale, gate_up: (weight, scale),
            ):
                layer = self._load(source, backend=backend)
            input_scale = torch.tensor([4.0] if backend == "trtllm" else [1.0, 4.0])
            torch.testing.assert_close(layer.nvfp4_w13_input_scale_inv, 1 / input_scale)
            torch.testing.assert_close(
                layer.nvfp4_w13_alpha, input_scale * layer.w13_scale_2
            )
            torch.testing.assert_close(
                layer.nvfp4_g1_scale_c, layer.nvfp4_w13_alpha / layer.w2_input_scale
            )

    def test_eplb_derived_scales_install_and_rollback_in_place(self):
        layer = self._load(_weights())
        layer._dynamic_eplb_enabled = True
        runtime = layer.eplb_runtime_weight_view()
        snapshots = {name: runtime[name].clone() for name in layer._eplb_weight_names()}
        pointers = {name: runtime[name].data_ptr() for name in snapshots}
        updates = {
            name: value.flip(0).contiguous() for name, value in snapshots.items()
        }
        # Force a scale change even when both experts have equal calibration.
        for name in layer.quant_method.eplb_aux_weight_names():
            updates[name] = updates[name] * 2
        counts, mapping = layer._routing_tensors(layer._expert_placement)
        placement = torch.tensor(
            layer._expert_placement.physical_to_logical, dtype=torch.int32
        )
        layer.apply_eplb_plan(placement, counts, mapping, updates, defer_commit=True)
        for name, expected in updates.items():
            self.assertEqual(runtime[name].data_ptr(), pointers[name])
            torch.testing.assert_close(
                runtime[name].view(torch.uint8), expected.view(torch.uint8)
            )
        layer.finalize_eplb_install(commit=False)
        for name, expected in snapshots.items():
            torch.testing.assert_close(
                runtime[name].view(torch.uint8), expected.view(torch.uint8)
            )

    def test_online_reference_matches_oldloader_bytes(self):
        from rtp_llm.model_loader.online_modelopt_fp4_quant_weight import (
            quantize_expert_to_nvfp4 as legacy,
        )

        for value in (
            torch.zeros(8, 64),
            torch.arange(-256, 256).reshape(8, 64).float() / 17,
        ):
            for dtype in (torch.bfloat16, torch.float16):
                actual = quantize_expert_to_nvfp4(value.to(dtype))
                expected = legacy(value.to(dtype))
                for left, right in zip(actual, expected):
                    torch.testing.assert_close(
                        left.reshape(-1).view(torch.uint8),
                        right.reshape(-1).view(torch.uint8),
                    )

    def test_h20_cannot_execute_native_nvfp4_backend(self):
        method = _experts().quant_method
        with mock.patch("torch.version.hip", None), mock.patch(
            "torch.cuda.get_device_capability", return_value=(9, 0)
        ):
            with self.assertRaisesRegex(ValueError, "SM100"):
                method.validate_runtime_device(torch.device("cuda:0"))


if __name__ == "__main__":
    unittest.main()
