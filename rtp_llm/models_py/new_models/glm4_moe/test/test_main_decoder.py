"""CPU tests of the production GLM decoder tree and target-owned layer filter."""

import gc
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from rtp_llm.models_py.model_loader import (
    NewLoaderConfig,
    NewLoaderLoadMethod,
    NewModelLoader,
)
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.new_models.glm4_moe.parameters import install_glm4_parameters
from rtp_llm.models_py.new_models.glm4_moe.test.checkpoint_fixture import checkpoint
from rtp_llm.models_py.registry import register_model
from rtp_llm.models_py.weight_mapper import WeightsMapper


@register_model("_glm4_main_decoders")
class _Decoders(RtpModule):
    fastsafetensors_load_isolated = True
    WEIGHTS_MAPPER = WeightsMapper(prefix_mapping={"model.": ""})

    def __init__(self, config, load_config):
        super().__init__()
        cfg = dict(
            num_layers=2,
            vocab_size=8,
            lm_head_tp_size=1,
            lm_head_tp_rank=0,
            lm_head_params_dtype=torch.float32,
            hidden_size=4,
            num_heads=2,
            num_kv_heads=1,
            head_dim=2,
            attn_tp_size=1,
            attn_tp_rank=0,
            ffn_tp_size=1,
            ffn_tp_rank=0,
            ep_size=1,
            ep_rank=0,
            num_experts=4,
            top_k=2,
            moe_intermediate_size=4,
            model_config=config,
            params_dtype=torch.float32,
            moe_config=None,
            quant_config=None,
            rms_norm_eps=1e-6,
            parallelism_config=SimpleNamespace(
                dp_size=1, dp_rank=0, world_size=1, local_world_size=1
            ),
        )
        install_glm4_parameters(self, cfg, layer_ids=range(2), partitioned=False)

    def load_weights(self, weights):
        return super().load_weights(self.WEIGHTS_MAPPER.apply(weights))

    def checkpoint_weight_name_filter(self):
        return self._layout.target_owns


def config(path):
    return SimpleNamespace(
        model_type="_glm4_main_decoders",
        ckpt_path=str(path),
        num_layers=2,
        tie_word_embeddings=False,
        normalize_lm_head_weight=False,
        logit_scale=1.0,
        hidden_size=4,
        expert_num=4,
        moe_k=2,
        moe_inter_size=4,
        moe_layer_index=[1],
        qk_norm=True,
        activation_type="SiGLU",
        has_moe_norm=True,
        moe_n_group=2,
        moe_topk_group=1,
        scoring_func=1,
        routed_scaling_factor=2.5,
    )


class MainDecoderTest(unittest.TestCase):
    def make(self, path):
        (path / "config.json").write_text(
            json.dumps(
                dict(
                    model_type="glm4_moe",
                    num_hidden_layers=2,
                    num_nextn_predict_layers=1,
                    first_k_dense_replace=1,
                    intermediate_size=8,
                    n_shared_experts=1,
                    use_qk_norm=True,
                )
            )
        )
        src = checkpoint()
        src["model.layers.2.enorm.weight"] = torch.ones(4)
        return src

    def load(self, path, method):
        loader = NewModelLoader(
            config(path),
            NewLoaderConfig(device="cpu", load_method=method),
            model_path=str(path),
        )
        result = loader.load()
        self.assertEqual(loader._selected_load_method, NewLoaderLoadMethod(method))
        return result

    def test_real_scratch_and_fst(self):
        import fastsafetensors.cpp as native

        self.assertTrue(native.__file__.endswith(".so"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            src = self.make(path)
            save_file(src, str(path / "model.safetensors"))
            scratch = self.load(path, "scratch")
            fast = self.load(path, "fastsafetensors")
        gc.collect()
        self.assertEqual(
            set(dict(fast.named_parameters())), set(dict(scratch.named_parameters()))
        )
        for name, param in fast.named_parameters():
            torch.testing.assert_close(
                param, dict(scratch.named_parameters())[name], rtol=0, atol=0
            )
        torch.testing.assert_close(
            fast.embed_tokens.weight, src["model.embed_tokens.weight"]
        )
        torch.testing.assert_close(fast.lm_head.weight, src["lm_head.weight"])
        torch.testing.assert_close(fast.norm.weight, src["model.norm.weight"])
        self.assertEqual(fast._layout.dense_width, 8)
        self.assertEqual(fast._layout.shared_width, 4)
        self.assertEqual(fast.layers[0].mlp.gate_up_proj.weight.shape, (16, 4))
        self.assertTrue(fast.layers[1].mlp.experts.is_cpu_checkpoint_staging())
        self.assertTrue(fast._layout.correction_bias)
        qkv = fast.layers[0].self_attn.qkv_proj
        torch.testing.assert_close(
            qkv.bias,
            torch.cat([src[f"model.layers.0.self_attn.{p}_proj.bias"] for p in "qkv"]),
        )
        w = fast.layers[1].mlp.experts.w13[2]
        torch.testing.assert_close(
            w,
            torch.cat(
                [
                    src[f"model.layers.1.mlp.experts.2.{p}_proj.weight"]
                    for p in ("up", "gate")
                ]
            ),
        )

    def test_missing_target_tensor_not_masked_by_draft(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            src = self.make(path)
            del src["model.layers.0.self_attn.q_proj.bias"]
            save_file(src, str(path / "model.safetensors"))
            with self.assertRaises((RuntimeError, ValueError)):
                self.load(path, "scratch")

    def test_draft_only_correction_is_not_target_bias(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            src = self.make(path)
            src["model.layers.2.mlp.gate.e_score_correction_bias"] = src.pop(
                "model.layers.1.mlp.gate.e_score_correction_bias"
            )
            save_file(src, str(path / "model.safetensors"))
            model = self.load(path, "scratch")
            self.assertFalse(model._layout.correction_bias)
            self.assertIsNone(model.layers[1].mlp.gate.e_score_correction_bias)
            self.assertTrue(model._layout.target_owns("model.layers.3.enorm.weight"))


if __name__ == "__main__":
    unittest.main()
