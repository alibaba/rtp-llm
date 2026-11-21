"""CPU checkpoint/parameter regressions; not a gfx950 kernel or golden test."""

import builtins
import os
import sys
import tempfile
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import torch
from safetensors.torch import save_file

from rtp_llm.config.quant_config import MXFp4QuarkQuantConfig
from rtp_llm.models_py import weight_mapper
from rtp_llm.models_py.layers.moe_experts import BaseMoEExperts
from rtp_llm.models_py.model_loader import (
    NewLoaderConfig,
    NewModelLoader,
    _ExpertRangeFilter,
)
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.quant_methods.base import QuantizationConfig
from rtp_llm.models_py.quant_methods.moe_quant_key import moe_checkpoint_method_key
from rtp_llm.models_py.quant_methods.mxfp4_moe import MXFp4QuarkMoEMethod
from rtp_llm.models_py.quant_methods.unquantized import UnquantizedLinearMethod
from rtp_llm.utils.model_weight import W

PREFIX = "model.layers.0.mlp.experts."


@contextmanager
def mock_device_arch(arch):
    with (
        mock.patch.object(torch.version, "hip", "7.2"),
        mock.patch.object(torch.cuda, "is_available", return_value=True),
        mock.patch.object(
            torch.cuda,
            "get_device_properties",
            return_value=SimpleNamespace(gcnArchName=arch),
        ),
        mock.patch.object(torch, "float4_e2m1fn_x2", object(), create=True),
    ):
        yield


def make_experts(tp_size=1, tp_rank=0, ep_size=1, ep_rank=0, redundant=0, inter=128):
    source = MXFp4QuarkQuantConfig(is_quanted=True)
    quant = QuantizationConfig(moe_checkpoint_method_key(source), source_config=source)
    parallelism = SimpleNamespace(
        dp_size=1,
        dp_rank=0,
        world_size=max(tp_size, ep_size),
        local_world_size=max(tp_size, ep_size),
        tp_size=tp_size,
        tp_rank=tp_rank,
        ep_size=ep_size,
        ep_rank=ep_rank,
    )
    model = SimpleNamespace(
        activation_type="SiGLU",
        quant_config=source,
        data_type="bf16",
        expert_num=4,
        hidden_size=64,
        num_layers=1,
        phy2log_path="",
        exported_device=None,
        eplb_config=SimpleNamespace(
            redundant_expert=redundant,
            phy_exp_num=lambda n: n + redundant,
            enable_eplb=False,
        ),
    )
    return BaseMoEExperts(
        num_experts=4,
        top_k=1,
        hidden_size=64,
        moe_intermediate_size=inter,
        tp_size=tp_size,
        tp_rank=tp_rank,
        ep_size=ep_size,
        ep_rank=ep_rank,
        params_dtype=torch.bfloat16,
        model_config=model,
        parallelism_config=parallelism,
        moe_config=SimpleNamespace(),
        quant_config=quant,
        layer_idx=0,
        prefix=PREFIX[:-1],
    )


def checkpoint_tensors(stacked=False):
    weights = {}
    for expert in range(4):
        for projection_index, projection in enumerate(BaseMoEExperts.PROJ_NAMES):
            rows, columns = (64, 128) if projection == "down_proj" else (128, 64)
            for suffix, divisor in (("weight", 2), ("weight_scale", 32)):
                # Include 0 and large UE8M0 exponents, but not NaN byte 255.
                value = (
                    torch.arange(rows * columns // divisor).reshape(
                        rows, columns // divisor
                    )
                    + expert * 13
                    + projection_index * 23
                ) % (255 if divisor == 32 else 256)
                weights[f"{PREFIX}{expert}.{projection}.{suffix}"] = value.to(
                    torch.uint8
                )
    if not stacked:
        return weights
    result = {}
    for suffix in ("weight", "weight_scale"):
        result[f"{PREFIX}gate_up_proj.{suffix}"] = torch.stack(
            [
                torch.cat(
                    (
                        weights[f"{PREFIX}{e}.gate_proj.{suffix}"],
                        weights[f"{PREFIX}{e}.up_proj.{suffix}"],
                    ),
                    dim=0,
                )
                for e in range(4)
            ]
        )
        result[f"{PREFIX}down_proj.{suffix}"] = torch.stack(
            [weights[f"{PREFIX}{e}.down_proj.{suffix}"] for e in range(4)]
        )
    return result


def load_checkpoint(layer, tensors):
    """Use actual safetensors I/O and the production EP slice-expansion path."""
    placement_filter = _ExpertRangeFilter(
        4,
        layer.ep_size,
        layer.ep_rank,
        model_config=layer._model_config,
        parallelism_config=layer._parallelism_config,
    )
    name_filter = NewModelLoader._checkpoint_name_filter(
        NewModelLoader._model_checkpoint_name_filter(RtpModule()), placement_filter
    )
    with tempfile.TemporaryDirectory(prefix="mxfp4_checkpoint_") as directory:
        path = os.path.join(directory, "model.safetensors")
        save_file(tensors, path)
        for name, tensor in weight_mapper.get_all_weights(
            [path], name_filter=name_filter, safetensor_slice_expander=placement_filter
        ):
            layer.load_weights({name.removeprefix(PREFIX): tensor})


class MXFp4CheckpointTest(unittest.TestCase):
    def assert_layout(self, layer):
        source = checkpoint_tensors()
        local_m = layer.moe_inter_tp
        rank = layer.moe_expert_tp_rank
        for local in range(layer.num_local_experts):
            logical = layer._expert_placement.physical_to_logical[
                layer._expert_placement.local_physical_start + local
            ]
            for suffix, divisor in (("weight", 2), ("weight_scale", 32)):
                w13 = layer.w13 if suffix == "weight" else layer.w13_scale
                w2 = layer.w2 if suffix == "weight" else layer.w2_scale
                for projection, offset in (("up_proj", 0), ("gate_proj", local_m)):
                    expected = source[f"{PREFIX}{logical}.{projection}.{suffix}"][
                        rank * local_m : (rank + 1) * local_m
                    ]
                    torch.testing.assert_close(
                        w13[local, offset : offset + local_m], expected
                    )
                local_k = local_m // divisor
                expected = source[f"{PREFIX}{logical}.down_proj.{suffix}"][
                    :, rank * local_k : (rank + 1) * local_k
                ]
                torch.testing.assert_close(w2[local], expected)
        self.assertTrue(layer.checkpoint_load_complete())
        layer.validate_weights_loaded()

    def test_real_qwen3_model_tree_checkpoint_dispatch(self):
        from rtp_llm.config.model_config import ModelConfig
        from rtp_llm.models_py.new_models.qwen3_moe.language import Qwen3MoeForCausalLM

        # This test loads the CPU parameter owner. The router is only
        # constructed on an accelerator; do not import it as a precondition.
        config = ModelConfig()
        config.model_type = "qwen_3_moe"
        config.num_layers = 1
        config.vocab_size = 8
        config.hidden_size = 64
        config.inter_size = 128
        config.expert_num = 4
        config.moe_inter_size = 128
        config.moe_k = 1
        config.moe_layer_index = [0]
        config.qk_norm = True
        config.moe_topk_group = 1
        config.attn_config.head_num = 2
        config.attn_config.kv_head_num = 1
        config.attn_config.size_per_head = 32
        config.layernorm_eps = 1e-6
        config.enable_fp32_lm_head = False
        config.tie_word_embeddings = True
        config.data_type = "bf16"
        config.quant_config = MXFp4QuarkQuantConfig(is_quanted=True)
        config.activation_type = "SiGLU"
        parallelism = SimpleNamespace(
            tp_size=1,
            tp_rank=0,
            ep_size=1,
            ep_rank=0,
            dp_size=1,
            dp_rank=0,
            pp_size=1,
            pp_rank=0,
            world_size=1,
            local_world_size=1,
            local_rank=0,
            get_attn_tp_size=lambda: 1,
            get_attn_tp_rank=lambda: 0,
            get_ffn_tp_size=lambda: 1,
            get_ffn_tp_rank=lambda: 0,
            prefill_cp_config=SimpleNamespace(
                is_enabled=lambda: False,
                is_prefill_enabled=lambda: False,
            ),
            ffn_disaggregate_config=SimpleNamespace(enable_ffn_disaggregate=False),
        )
        load_config = NewLoaderConfig(
            device="cpu",
            compute_dtype=torch.bfloat16,
            quant_config=QuantizationConfig(
                moe_checkpoint_method_key(config.quant_config),
                source_config=config.quant_config,
            ),
            parallelism_config=parallelism,
            moe_config=SimpleNamespace(
                ll_num_max_token=1,
                masked_max_token_num=1,
                moe_strategy=0,
                use_mori_ep=False,
                use_deepep_moe=False,
                use_deepep_low_latency=False,
                use_all_gather=True,
                fake_balance_expert=False,
            ),
        )
        model = Qwen3MoeForCausalLM(config, load_config)
        weights = checkpoint_tensors()
        dense_shapes = {
            "model.embed_tokens.weight": (8, 64),
            "model.layers.0.input_layernorm.weight": (64,),
            "model.layers.0.self_attn.q_proj.weight": (64, 64),
            "model.layers.0.self_attn.k_proj.weight": (32, 64),
            "model.layers.0.self_attn.v_proj.weight": (32, 64),
            "model.layers.0.self_attn.q_norm.weight": (32,),
            "model.layers.0.self_attn.k_norm.weight": (32,),
            "model.layers.0.self_attn.o_proj.weight": (64, 64),
            "model.layers.0.post_attention_layernorm.weight": (64,),
            "model.layers.0.mlp.gate.weight": (4, 64),
            "model.norm.weight": (64,),
        }
        weights.update(
            {
                name: torch.ones(shape, dtype=torch.bfloat16)
                for name, shape in dense_shapes.items()
            }
        )
        with tempfile.TemporaryDirectory(prefix="mxfp4_qwen3_checkpoint_") as directory:
            path = os.path.join(directory, "model.safetensors")
            save_file(weights, path)
            model.load_weights(weight_mapper.get_all_weights([path]))
        NewModelLoader._validate_loaded_weights(model)
        self.assert_layout(model.layers[0].mlp.experts)
        torch.testing.assert_close(model.lm_head.weight, model.embed_tokens.weight)
        # Intentionally stop before GPU runtime construction, not a forward test.

    def test_real_checkpoint_per_expert_tp(self):
        for rank in range(2):
            with self.subTest(rank=rank):
                layer = make_experts(tp_size=2, tp_rank=rank)
                load_checkpoint(layer, checkpoint_tensors())
                self.assert_layout(layer)
                self.assertEqual(layer.w13.dtype, torch.uint8)
                self.assertEqual(layer.w13.numel(), 4 * 128 * 32)

    def test_real_checkpoint_stacked_tp(self):
        for rank in range(2):
            layer = make_experts(tp_size=2, tp_rank=rank)
            load_checkpoint(layer, checkpoint_tensors(stacked=True))
            self.assert_layout(layer)

    def test_real_checkpoint_stacked_ep_redundant(self):
        for redundant in (0, 2):
            for rank in range(2):
                with self.subTest(redundant=redundant, rank=rank):
                    layer = make_experts(ep_size=2, ep_rank=rank, redundant=redundant)
                    load_checkpoint(layer, checkpoint_tensors(stacked=True))
                    self.assert_layout(layer)

    def test_real_checkpoint_per_expert_ep_redundant(self):
        for rank in range(2):
            layer = make_experts(ep_size=2, ep_rank=rank, redundant=2)
            load_checkpoint(layer, checkpoint_tensors())
            self.assert_layout(layer)

    def test_missing_scale_fails_completeness(self):
        layer = make_experts()
        weights = checkpoint_tensors()
        del weights[f"{PREFIX}0.gate_proj.weight_scale"]
        load_checkpoint(layer, weights)
        with self.assertRaisesRegex(RuntimeError, "missing required MoE auxiliary"):
            layer.validate_weights_loaded()

    def test_bad_packed_dtype_rejected(self):
        layer = make_experts()
        with self.assertRaisesRegex(TypeError, "byte storage"):
            layer.load_weights({"0.gate_proj.weight": torch.zeros(128, 32)})

    def test_bad_scale_dtype_rejected_without_numeric_cast(self):
        layer = make_experts()
        with self.assertRaisesRegex(TypeError, "byte storage"):
            layer.load_weights({"0.gate_proj.weight_scale": torch.ones(128, 2)})

    def test_ue8m0_nan_rejected(self):
        layer = make_experts()
        with self.assertRaisesRegex(ValueError, "NaN encoding"):
            layer.load_weights(
                {
                    "0.gate_proj.weight_scale": torch.full(
                        (128, 2), 255, dtype=torch.uint8
                    )
                }
            )

    def test_tp_cannot_split_scale_group(self):
        with self.assertRaisesRegex(ValueError, "multiples of 32"):
            make_experts(tp_size=4, inter=64)

    def test_bad_packed_shape_rejected(self):
        layer = make_experts()
        with self.assertRaisesRegex(ValueError, "expected"):
            layer.load_weights(
                {"0.down_proj.weight": torch.zeros(64, 128, dtype=torch.uint8)}
            )

    def test_runtime_aux_is_module_owned_bytes(self):
        layer = make_experts()
        load_checkpoint(layer, checkpoint_tensors())
        weights = {}
        layer.quant_method.add_weight_tensors(layer, weights)
        self.assertIs(weights[W.moe_s1], layer.w13_scale)
        self.assertIs(weights[W.moe_s2], layer.w2_scale)

    def test_duplicate_scale_rejected(self):
        layer = make_experts()
        value = {"0.gate_proj.weight_scale": torch.zeros(128, 2, dtype=torch.uint8)}
        layer.load_weights(value)
        with self.assertRaisesRegex(RuntimeError, "Duplicate MoE checkpoint"):
            layer.load_weights(value)

    def test_shell_ue8m0_keeps_storage_bytes(self):
        dtype = getattr(torch, "float8_e8m0fnu", None)
        if dtype is None:
            self.skipTest("torch lacks UE8M0 shell dtype")
        layer = make_experts()
        value = torch.arange(128 * 2).reshape(128, 2).remainder(255).to(torch.uint8)
        layer.load_weights({"0.up_proj.weight_scale": value.view(dtype)})
        torch.testing.assert_close(layer.w13_scale[0, :128], value)

    def test_ep_execution_does_not_claim_loading_as_gpu_support(self):
        layer = make_experts(ep_size=2)
        with self.assertRaisesRegex(RuntimeError, "pure TP only"):
            layer.validate_runtime_device(torch.device("cuda:0"))

    def test_malformed_group_size_rejected(self):
        source = MXFp4QuarkQuantConfig(is_quanted=True, group_size=64)
        with self.assertRaisesRegex(ValueError, "groups of 32"):
            MXFp4QuarkMoEMethod(QuantizationConfig("mxfp4_quark", source_config=source))

    def test_dense_and_shared_expert_remain_unquantized(self):
        layer = make_experts()
        for prefix in (
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.mlp.shared_expert.gate_proj",
        ):
            self.assertIsInstance(
                layer._quant_config.get_quant_method(torch.nn.Module(), prefix),
                UnquantizedLinearMethod,
            )

    def test_no_online_quark_claim(self):
        with self.assertRaisesRegex(ValueError, "prequantized"):
            moe_checkpoint_method_key(MXFp4QuarkQuantConfig(is_quanted=False))
        with self.assertRaisesRegex(ValueError, "already-quantized"):
            MXFp4QuarkMoEMethod(
                QuantizationConfig(
                    "mxfp4_quark", source_config=MXFp4QuarkQuantConfig(is_quanted=False)
                )
            )

    def test_cpu_execution_is_not_claimed(self):
        with self.assertRaisesRegex(RuntimeError, "ROCm gfx950"):
            MXFp4QuarkMoEMethod().validate_runtime_device(torch.device("cpu"))

    def test_mi308_not_misidentified_as_mi355(self):
        with mock_device_arch("gfx942:sramecc+:xnack-"):
            with self.assertRaisesRegex(RuntimeError, "gfx942/MI308"):
                MXFp4QuarkMoEMethod().validate_runtime_device(torch.device("cuda:3"))

    def test_missing_aiter_fails_preflight(self):
        original = builtins.__import__

        def fail_aiter(name, *args, **kwargs):
            if name == "aiter":
                raise OSError("incompatible AITER ABI")
            return original(name, *args, **kwargs)

        with mock_device_arch("gfx950"), mock.patch(
            "builtins.__import__", side_effect=fail_aiter
        ):
            with self.assertRaisesRegex(RuntimeError, "AITER MXFP4 runtime"):
                MXFp4QuarkMoEMethod().validate_runtime_device(torch.device("cuda:0"))

    def test_missing_backend_symbol_fails_preflight(self):
        fake_modules = {
            "aiter": SimpleNamespace(
                QuantType=SimpleNamespace(per_1x32=1),
                ActivationType=SimpleNamespace(Silu=1),
            ),
            "aiter.fused_moe": SimpleNamespace(fused_moe=lambda: None),
            "aiter.ops.shuffle": SimpleNamespace(shuffle_weight=None),
            "aiter.utility.fp4_utils": SimpleNamespace(e8m0_shuffle=lambda: None),
        }
        with mock_device_arch("gfx950"), mock.patch.dict(sys.modules, fake_modules):
            with self.assertRaisesRegex(RuntimeError, "AITER MXFP4 runtime"):
                MXFp4QuarkMoEMethod().validate_runtime_device(torch.device("cuda:0"))


if __name__ == "__main__":
    unittest.main()
