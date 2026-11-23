"""Real checkpoint -> generic stacked experts; sequential rank views, no GPU."""

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from rtp_llm.models_py.layers.moe_experts import BaseMoEExperts
from rtp_llm.models_py.model_loader import (
    NewLoaderConfig,
    NewLoaderLoadMethod,
    NewModelLoader,
)
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.registry import register_model


@register_model("_glm4_nonlite_main_expert_test")
class _ExpertModel(RtpModule):
    fastsafetensors_load_isolated = True

    def __init__(self, config, loader):
        super().__init__()
        self.experts = BaseMoEExperts(
            num_experts=4,
            top_k=2,
            hidden_size=4,
            moe_intermediate_size=4,
            tp_size=config.tp,
            tp_rank=config.tr,
            ep_size=config.ep,
            ep_rank=config.er,
            params_dtype=torch.bfloat16,
            model_config=config,
            parallelism_config=SimpleNamespace(
                dp_size=1,
                dp_rank=0,
                world_size=max(config.tp, config.ep),
                local_world_size=max(config.tp, config.ep),
            ),
            moe_config=None,
            quant_config=None,
            layer_idx=0,
        )


class GenericExpertsCheckpointTest(unittest.TestCase):
    def test_fp8_expert_checkpoint_staging(self):
        from safetensors import safe_open

        from rtp_llm.config.quant_config import (
            Fp8BlockWiseQuantConfig,
            Fp8PerTensorCompressedQuantConfig,
        )
        from rtp_llm.models_py.quant_methods.base import QuantizationConfig
        from rtp_llm.models_py.quant_methods.moe_quant_key import (
            moe_checkpoint_method_key,
        )

        configs = [
            (
                Fp8BlockWiseQuantConfig(bits=8, group_size=128, is_quanted=True),
                "weight_scale_inv",
            ),
            (
                Fp8PerTensorCompressedQuantConfig(is_quanted=True, dynamic=False),
                "weight_scale",
            ),
        ]
        # On EP rank 1 only logical expert 1 is represented. Verify weight,
        # calibration and scale planes independently from the checkpoint bytes.
        for source, scale_suffix in configs:
            with self.subTest(
                method=source.get_method()
            ), tempfile.TemporaryDirectory() as tmp:

                def make_experts():
                    return BaseMoEExperts(
                        num_experts=2,
                        top_k=1,
                        hidden_size=4,
                        moe_intermediate_size=4,
                        tp_size=1,
                        tp_rank=0,
                        ep_size=2,
                        ep_rank=1,
                        params_dtype=torch.bfloat16,
                        model_config=SimpleNamespace(
                            num_layers=1, activation_type="SiGLU", quant_config=source
                        ),
                        parallelism_config=SimpleNamespace(
                            dp_size=1, dp_rank=0, world_size=2, local_world_size=2
                        ),
                        moe_config=None,
                        quant_config=QuantizationConfig(
                            moe_checkpoint_method_key(source), source
                        ),
                        layer_idx=0,
                    )

                inputs = {}
                for eid in range(2):
                    for i, proj in enumerate(("up", "gate", "down"), 1):
                        prefix = f"{eid}.{proj}_proj."
                        inputs[prefix + "weight"] = torch.full((4, 4), 8 * eid + i).to(
                            torch.float8_e4m3fn
                        )
                        inputs[prefix + scale_suffix] = torch.tensor([[8.0 * eid + i]])
                        if scale_suffix == "weight_scale":
                            inputs[prefix + "input_scale"] = torch.tensor(
                                [16.0 * eid + i]
                            )
                path = str(Path(tmp) / "experts.safetensors")
                save_file(inputs, path)
                owner = make_experts()
                incomplete = make_experts()
                with safe_open(path, framework="pt", device="cpu") as checkpoint:
                    for key in checkpoint.keys():
                        entry = {key: checkpoint.get_tensor(key)}
                        owner.load_weights(entry)
                        if key != "1.gate_proj." + scale_suffix:
                            incomplete.load_weights(entry)
                owner.validate_weights_loaded()
                with self.assertRaises(RuntimeError):
                    incomplete.validate_weights_loaded()
                torch.testing.assert_close(
                    owner.w13[0].float(),
                    torch.cat(
                        [inputs[f"1.{p}_proj.weight"].float() for p in ("up", "gate")]
                    ),
                )
                torch.testing.assert_close(
                    owner.w2[0].float(), inputs["1.down_proj.weight"].float()
                )
                if scale_suffix == "weight_scale_inv":
                    torch.testing.assert_close(
                        owner.w13_scale.flatten(), torch.tensor([9.0, 10.0])
                    )
                    torch.testing.assert_close(
                        owner.w2_scale.flatten(), torch.tensor([11.0])
                    )
                else:
                    torch.testing.assert_close(
                        owner.checkpoint_input_scales[0],
                        torch.tensor([18.0, 17.0, 19.0]),
                    )
                    torch.testing.assert_close(owner._gate_scales, torch.tensor([10.0]))
                    torch.testing.assert_close(owner._up_scales, torch.tensor([9.0]))
                    torch.testing.assert_close(owner._down_scales, torch.tensor([11.0]))
                # The CUDA-build e4m3fn canonical packing is pure tensor
                # arithmetic. Do not claim CPU can select/run a fused executor.
                if torch.version.hip is None:
                    owner.quant_method.process_weights_after_loading(owner)
                    if scale_suffix == "weight_scale":
                        self.assertEqual(owner.w13_scale.item(), 10.0)
                        self.assertEqual(owner.w2_scale.item(), 11.0)
                        self.assertEqual(owner.w13_input_scale.item(), 18.0)
                        self.assertEqual(owner.w2_input_scale.item(), 19.0)
                        torch.testing.assert_close(
                            owner.w13_input_scale_inv, torch.tensor(1.0 / 18)
                        )
                        torch.testing.assert_close(
                            owner.w2_input_scale_inv, torch.tensor(1.0 / 19)
                        )
                        expected_up = (inputs["1.up_proj.weight"].float() * 0.9).to(
                            torch.float8_e4m3fn
                        )
                        torch.testing.assert_close(
                            owner.w13[0, :4].float(), expected_up.float()
                        )
                    else:
                        torch.testing.assert_close(
                            owner.w13_scale.flatten(), torch.tensor([9.0, 10.0])
                        )
                    self.assertIsNone(owner.fused_moe)

    def test_actual_main_quant_config_keys(self):
        from rtp_llm.config.quant_config import (
            CompressedW4A8Int4PerChannelQuantConfig,
            Fp8BlockWiseQuantConfig,
            Fp8PerTensorQuantConfig,
        )
        from rtp_llm.models_py.quant_methods.moe_quant_key import (
            moe_checkpoint_method_key,
        )

        w4 = CompressedW4A8Int4PerChannelQuantConfig(
            bits=4, group_size=32, is_quanted=True
        )
        self.assertEqual(
            moe_checkpoint_method_key(w4), "W4A8_INT4_PER_CHANNEL_COMPRESSED"
        )
        for encoded in (True, False):
            for quant, group, key in (
                (Fp8PerTensorQuantConfig, 0, "fp8"),
                (Fp8BlockWiseQuantConfig, 128, "fp8_block"),
            ):
                cfg = quant(bits=8, group_size=group, is_quanted=encoded)
                self.assertEqual(
                    moe_checkpoint_method_key(cfg), key if encoded else key + "_online"
                )

    def test_shared_ffn_and_correction_router(self):
        from rtp_llm.models_py.new_models.deepseek_v3.moe import (
            DeepSeekV32MoEBlock,
            _select_deepseek_noaux_topk,
        )

        config = SimpleNamespace(
            activation_type="SiGLU",
            num_layers=1,
            hidden_size=4,
            expert_num=4,
            moe_k=2,
            has_moe_norm=True,
            moe_n_group=2,
            moe_topk_group=1,
            scoring_func=1,
            routed_scaling_factor=2.5,
        )
        block = DeepSeekV32MoEBlock(
            hidden_size=4,
            moe_intermediate_size=4,
            num_experts=4,
            top_k=2,
            layer_idx=0,
            tp_size=1,
            tp_rank=0,
            ep_size=1,
            ep_rank=0,
            model_config=config,
            parallelism_config=SimpleNamespace(
                dp_size=1, dp_rank=0, world_size=1, local_world_size=1
            ),
            moe_config=None,
            quant_config=None,
            params_dtype=torch.float32,
            has_shared_expert=True,
            shared_expert_intermediate_size=4,
            scoring_func=1,
            routed_scaling_factor=2.5,
            n_group=2,
            topk_group=1,
            topk_method="noaux_tc",
            has_moe_norm=True,
            correction_bias=True,
        )
        eye = torch.eye(4)
        block.load_weights(
            {
                "gate.weight": eye,
                "gate.e_score_correction_bias": torch.tensor([0.0, 0.0, 4.0, 4.0]),
                "shared_experts.gate_proj.weight": eye * 3,
                "shared_experts.up_proj.weight": eye * 2,
                "shared_experts.down_proj.weight": eye,
            }
        )
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        torch.testing.assert_close(
            block.shared_experts(x), torch.nn.functional.silu(x * 3) * (x * 2)
        )
        weights, ids = _select_deepseek_noaux_topk(
            block.gate(x),
            block.gate.e_score_correction_bias,
            top_k=2,
            n_group=2,
            topk_group=1,
            renormalize=True,
            routed_scaling_factor=2.5,
        )
        self.assertEqual(set(ids[0].tolist()), {2, 3})
        scores = torch.sigmoid(x.gather(1, ids))
        torch.testing.assert_close(weights, 2.5 * scores / scores.sum(-1, keepdim=True))

    def exercise(self, method):
        src = {}
        for expert in range(4):
            for index, name in enumerate(("up", "gate", "down")):
                src[f"experts.{expert}.{name}_proj.weight"] = (
                    torch.arange(16).reshape(4, 4) + expert * 48 + index * 16
                ).bfloat16()
        with tempfile.TemporaryDirectory() as folder:
            save_file(src, str(Path(folder) / "model.safetensors"))
            for tp, tr, ep, er in (
                (1, 0, 1, 0),
                (2, 0, 1, 0),
                (2, 1, 1, 0),
                (1, 0, 2, 0),
                (1, 0, 2, 1),
            ):
                with self.subTest(tp=tp, tr=tr, ep=ep, er=er, method=method):
                    cfg = SimpleNamespace(
                        model_type="_glm4_nonlite_main_expert_test",
                        num_layers=1,
                        activation_type="SiGLU",
                        tp=tp,
                        tr=tr,
                        ep=ep,
                        er=er,
                    )
                    loader = NewModelLoader(
                        cfg,
                        NewLoaderConfig(device="cpu", load_method=method),
                        model_path=folder,
                    )
                    experts = loader.load().experts
                    self.assertEqual(
                        loader._selected_load_method, NewLoaderLoadMethod(method)
                    )
                    self.assertTrue(experts.is_cpu_checkpoint_staging())
                    self.assertIsNone(experts.fused_moe)
                    for local in range(4 // ep):
                        gid = er * (4 // ep) + local
                        pair = [
                            src[f"experts.{gid}.{name}_proj.weight"].chunk(tp, dim=0)[
                                tr
                            ]
                            for name in ("up", "gate")
                        ]
                        torch.testing.assert_close(
                            experts.w13[local], torch.cat(pair), rtol=0, atol=0
                        )
                        torch.testing.assert_close(
                            experts.w2[local],
                            src[f"experts.{gid}.down_proj.weight"].chunk(tp, dim=1)[tr],
                            rtol=0,
                            atol=0,
                        )

    def test_scratch(self):
        self.exercise("scratch")

    @unittest.skipIf(
        os.environ.get("NEWLOADER_TEST_NATIVE_FST") == "0",
        "this platform lock does not ship the native FST reader",
    )
    def test_native_fst(self):
        import fastsafetensors.cpp as native

        self.assertTrue(native.__file__.endswith(".so"))
        self.exercise("fastsafetensors")


if __name__ == "__main__":
    unittest.main()
