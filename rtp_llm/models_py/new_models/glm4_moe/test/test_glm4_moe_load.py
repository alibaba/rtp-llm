"""Real HF tiny-checkpoint CPU loading; not a GPU/smoke execution claim."""

import json
import os
import tempfile
import unittest

import torch
from safetensors.torch import save_file

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models_py.model_loader import NewModelLoader
from rtp_llm.models_py.new_models.glm4_moe.language import Glm4MoeForCausalLM
from rtp_llm.models_py.new_models.glm4_moe.test.checkpoint_fixture import checkpoint
from rtp_llm.models_py.new_models.glm4_moe.test.load_config_fixture import make_load_config


def make_config(path, *, qk_norm=True, correction=True, top_k=2, normalize=True):
    config = ModelConfig()
    config.model_type = "glm4_moe"
    config.ckpt_path = path
    config.num_layers = 2
    config.vocab_size = 8
    config.hidden_size = 4
    config.inter_size = 4
    config.expert_num = 4
    config.moe_inter_size = 4
    config.moe_k = top_k
    config.moe_layer_index = [1]
    config.moe_n_group = 2
    config.moe_topk_group = 1
    config.has_moe_norm = normalize
    config.scoring_func = 1
    config.routed_scaling_factor = 2.5
    config.qk_norm = qk_norm
    config.attn_config.head_num = 2
    config.attn_config.kv_head_num = 1
    config.attn_config.size_per_head = 2
    config.layernorm_eps = 1e-6
    config.data_type = "fp32"
    config.enable_fp32_lm_head = False
    config.tie_word_embeddings = False
    config.quant_config = None
    config.activation_type = "SiGLU"
    hf = dict(
        model_type="glm4_moe",
        intermediate_size=8,
        first_k_dense_replace=1,
        n_shared_experts=1,
        use_qk_norm=qk_norm,
    )
    if correction:
        hf["topk_method"] = "noaux_tc"
    with open(os.path.join(path, "config.json"), "w") as handle:
        json.dump(hf, handle)
    return config


class Glm4MoeLoadTest(unittest.TestCase):
    def load(self, directory, weights, **options):
        config = make_config(directory, **options)
        save_file(weights, os.path.join(directory, "model.safetensors"))
        return config

    def test_real_hf_checkpoint_tp_ep_parameter_owners(self):
        with tempfile.TemporaryDirectory() as directory:
            weights = checkpoint()
            config = self.load(directory, weights)
            for tp, rank, ep, ep_rank in (
                (1, 0, 1, 0),
                (2, 0, 1, 0),
                (2, 1, 1, 0),
                (1, 0, 2, 0),
                (1, 0, 2, 1),
                (2, 1, 2, 1),
            ):
                with self.subTest(tp=tp, rank=rank, ep=ep, ep_rank=ep_rank):
                    model = NewModelLoader(
                        config,
                        make_load_config(tp, rank, ep, ep_rank),
                        model_path=directory,
                    ).load()
                    self.assertIsInstance(model, Glm4MoeForCausalLM)
                    attn = model.layers[1].self_attn
                    expected = torch.cat(
                        [
                            weights["model.layers.1.self_attn.q_proj.weight"].chunk(tp)[
                                rank
                            ],
                            weights["model.layers.1.self_attn.k_proj.weight"],
                            weights["model.layers.1.self_attn.v_proj.weight"],
                        ]
                    )
                    torch.testing.assert_close(attn.qkv_proj.weight, expected)
                    expected_bias = torch.cat(
                        [
                            weights["model.layers.1.self_attn.q_proj.bias"].chunk(tp)[
                                rank
                            ],
                            weights["model.layers.1.self_attn.k_proj.bias"],
                            weights["model.layers.1.self_attn.v_proj.bias"],
                        ]
                    )
                    torch.testing.assert_close(attn.qkv_proj.bias, expected_bias)
                    torch.testing.assert_close(
                        attn.q_norm.weight,
                        weights["model.layers.1.self_attn.q_norm.weight"],
                    )
                    dense = model.layers[0].mlp
                    torch.testing.assert_close(
                        dense.gate_up_proj.weight,
                        torch.cat(
                            [
                                weights[f"model.layers.0.mlp.{role}_proj.weight"].chunk(
                                    tp
                                )[rank]
                                for role in ("gate", "up")
                            ]
                        ),
                    )
                    experts = model.layers[1].mlp.experts
                    self.assertIsNone(
                        experts.fused_moe
                    )  # CPU staging is not a backend.
                    for local, logical in enumerate(
                        range(ep_rank * (4 // ep), (ep_rank + 1) * (4 // ep))
                    ):
                        expert_tp = tp if ep == 1 else 1
                        expert_rank = rank % expert_tp
                        # The fused MoE owner and main GLM descriptor both use
                        # [up, gate], unlike dense MergedColumnParallelLinear.
                        merged = torch.cat(
                            [
                                weights[
                                    f"model.layers.1.mlp.experts.{logical}.{role}_proj.weight"
                                ].chunk(expert_tp)[expert_rank]
                                for role in ("up", "gate")
                            ]
                        )
                        torch.testing.assert_close(experts.w13[local], merged)
                    torch.testing.assert_close(
                        model.layers[1].mlp.gate.e_score_correction_bias,
                        weights["model.layers.1.mlp.gate.e_score_correction_bias"],
                    )

    def test_missing_required_checkpoint_tensors_fail(self):
        for missing in (
            "self_attn.q_proj.bias",
            "self_attn.q_norm.weight",
            "mlp.gate.e_score_correction_bias",
            "mlp.experts.3.down_proj.weight",
        ):
            with self.subTest(
                missing=missing
            ), tempfile.TemporaryDirectory() as directory:
                weights = checkpoint()
                del weights["model.layers.1." + missing]
                config = self.load(directory, weights)
                with self.assertRaisesRegex(
                    (ValueError, RuntimeError), "[Mm]issing|[Ii]ncomplete|not loaded"
                ):
                    NewModelLoader(
                        config, make_load_config(), model_path=directory
                    ).load()

    def test_expanded_gqa_width_is_not_hidden_size(self):
        # GLM4.5 uses 96 * 128 attention channels with hidden_size=5120.
        # Retain that distinction in the tiny fixture, unlike a Qwen alias.
        with tempfile.TemporaryDirectory() as directory:
            weights = checkpoint()
            for layer in range(2):
                prefix = f"model.layers.{layer}.self_attn."
                weights[prefix + "q_proj.weight"] = torch.arange(
                    32, dtype=torch.float32
                ).reshape(8, 4)
                weights[prefix + "q_proj.bias"] = torch.arange(8, dtype=torch.float32)
                weights[prefix + "o_proj.weight"] = torch.arange(
                    32, dtype=torch.float32
                ).reshape(4, 8)
            config = self.load(directory, weights)
            config.attn_config.head_num = 4
            for rank in (0, 1):
                model = NewModelLoader(
                    config, make_load_config(2, rank), model_path=directory
                ).load()
                attn = model.layers[1].self_attn
                self.assertEqual(tuple(attn.qkv_proj.weight.shape), (8, 4))
                torch.testing.assert_close(
                    attn.qkv_proj.weight[:4],
                    weights["model.layers.1.self_attn.q_proj.weight"].chunk(2)[rank],
                )
                torch.testing.assert_close(
                    attn.o_proj.weight,
                    weights["model.layers.1.self_attn.o_proj.weight"].chunk(2, dim=1)[
                        rank
                    ],
                )

    def test_no_qk_norm_and_metadata_detected_correction(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.load(
                directory, checkpoint(qk_norm=False), qk_norm=False, correction=False
            )
            model = NewModelLoader(
                config, make_load_config(), model_path=directory
            ).load()
            self.assertFalse(hasattr(model.layers[1].self_attn, "q_norm"))
            self.assertIsNotNone(model.layers[1].mlp.gate.e_score_correction_bias)

    def test_qkv_bias_and_per_head_qk_norm_before_attention(self):
        with tempfile.TemporaryDirectory() as directory:
            weights = checkpoint()
            config = self.load(directory, weights)
            model = NewModelLoader(
                config, make_load_config(), model_path=directory
            ).load()
            attn = model.layers[0].self_attn
            hidden = torch.tensor([[0.25, -0.5, 1.0, 0.1]])
            prefix = "model.layers.0.self_attn."
            qkv = torch.nn.functional.linear(
                hidden,
                torch.cat(
                    [
                        weights[prefix + role + "_proj.weight"]
                        for role in ("q", "k", "v")
                    ]
                ),
                torch.cat(
                    [weights[prefix + role + "_proj.bias"] for role in ("q", "k", "v")]
                ),
            )

            def normalize(tensor, role):
                shaped = tensor.reshape(1, -1, 2)
                return (
                    shaped
                    * torch.rsqrt(shaped.square().mean(-1, keepdim=True) + 1e-6)
                    * weights[prefix + role + "_norm.weight"]
                ).flatten(1)

            expected = torch.cat(
                [normalize(qkv[:, :4], "q"), normalize(qkv[:, 4:6], "k"), qkv[:, 6:]],
                -1,
            )
            captured = []

            class CaptureAttention:
                def forward(self, value, kv_cache, layer_idx):
                    captured.append(value.detach().clone())
                    return value[:, :4]

            with torch.inference_mode():
                output = attn(hidden, CaptureAttention())
            torch.testing.assert_close(captured[0], expected)
            torch.testing.assert_close(
                output,
                torch.nn.functional.linear(
                    expected[:, :4], weights[prefix + "o_proj.weight"]
                ),
            )

    def test_missing_dense_width_or_wrong_layer_map_is_explicit(self):
        for corrupt in ("intermediate_size", "first_k_dense_replace"):
            with self.subTest(
                corrupt=corrupt
            ), tempfile.TemporaryDirectory() as directory:
                config = self.load(directory, checkpoint())
                path = os.path.join(directory, "config.json")
                with open(path) as handle:
                    data = json.load(handle)
                if corrupt == "intermediate_size":
                    del data[corrupt]
                else:
                    data[corrupt] = 0
                with open(path, "w") as handle:
                    json.dump(data, handle)
                with self.assertRaisesRegex(
                    ValueError, "dense intermediate_size|moe_layer_index"
                ):
                    NewModelLoader(
                        config, make_load_config(), model_path=directory
                    ).load()

    def test_actual_legacy_router_semantics(self):
        for correction in (True, False):
            for top_k in (1, 2):
                for normalize in (True, False):
                    with self.subTest(
                        correction=correction, top_k=top_k, normalize=normalize
                    ), tempfile.TemporaryDirectory() as directory:
                        config = self.load(
                            directory,
                            checkpoint(correction=correction),
                            correction=correction,
                            top_k=top_k,
                            normalize=normalize,
                        )
                        model = NewModelLoader(
                            config, make_load_config(), model_path=directory
                        ).load()
                        block = model.layers[1].mlp
                        logits = torch.tensor([[1.5, 0.2, -0.8, -1.9]])
                        if correction:
                            with torch.inference_mode():
                                block.gate.e_score_correction_bias.copy_(
                                    torch.tensor([-3.0, -2.0, 5.0, 4.0])
                                )
                            # Bias selects group 1 although raw logits prefer group 0.
                            ids = torch.tensor([[2, 3][:top_k]])
                            expected = logits.sigmoid().gather(1, ids)
                        else:
                            expected, ids = logits.softmax(-1).topk(top_k, sorted=False)
                        if normalize:
                            expected /= expected.sum(-1, keepdim=True)
                        if correction:
                            expected *= 2.5
                        actual, actual_ids = block.route(logits, ids_dtype=torch.int64)
                        order = actual_ids.argsort(-1)
                        torch.testing.assert_close(
                            actual_ids.gather(1, order), ids.sort(-1).values
                        )
                        torch.testing.assert_close(
                            actual.gather(1, order), expected.gather(1, ids.argsort(-1))
                        )


if __name__ == "__main__":
    unittest.main()
