"""Whole tiny Kimi vision/text loading; explicit CPU MLA reference, not GPU smoke."""

import json
import os
import tempfile
import types
import unittest

import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import save_file

from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.models.direct_checkpoint import RuntimeWeightView
from rtp_llm.models.kimi_k25.kimi_k25 import KimiK25
from rtp_llm.models_py.model_loader import (
    NewLoaderConfig,
    NewLoaderLoadMethod,
    NewModelLoader,
)
from rtp_llm.models_py.new_models.kimi_k25.language import KimiK25ForCausalLM
from rtp_llm.models_py.new_models.kimi_k25.vision import (
    KimiK25ForVisionEmbedding,
)
from rtp_llm.models_py.quant_methods import QuantizationConfig
from rtp_llm.models_py.registry import get_model_class
from rtp_llm.models_py.new_models.kimi_k25.vision_runtime import mm_projector_forward
from rtp_llm.multimodal.multimodal_mixin_factory import MultimodalMixinFactory
from rtp_llm.ops import ParallelismConfig
from rtp_llm.multimodal.multimodal_mixins.kimi_k25.kimi_k25_mixin import KimiK25Mixin
from rtp_llm.utils.base_model_datatypes import VitParameters


class _CpuKimi(KimiK25):
    def _get_device_str(self):
        return "cpu"

    def load_tokenizer(self):
        pass  # No tokenizer artifacts in this tiny checkpoint.

    def _init_custom_module(self):
        return None


def load_config(method=NewLoaderLoadMethod.SCRATCH):
    parallel = types.SimpleNamespace(
        tp_size=1,
        tp_rank=0,
        ep_size=1,
        ep_rank=0,
        dp_size=1,
        prefill_cp_config=types.SimpleNamespace(
            is_enabled=lambda: False, is_prefill_enabled=lambda: False
        ),
        ffn_disaggregate_config=types.SimpleNamespace(enable_ffn_disaggregate=False),
        get_attn_tp_size=lambda: 1,
        get_attn_tp_rank=lambda: 0,
        get_ffn_tp_size=lambda: 1,
        get_ffn_tp_rank=lambda: 0,
    )
    return NewLoaderConfig(
        device="cpu",
        load_method=method,
        compute_dtype=torch.float32,
        quant_config=QuantizationConfig("none"),
        parallelism_config=parallel,
    )


def config():
    return dict(
        model_type="kimi_k25",
        media_placeholder_token_id=17,
        text_config=dict(
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            vocab_size=16,
            intermediate_size=16,
            q_lora_rank=4,
            kv_lora_rank=4,
            qk_nope_head_dim=2,
            qk_rope_head_dim=2,
            v_head_dim=2,
            max_seq_len=16,
            n_routed_experts=0,
            num_experts_per_tok=0,
            moe_intermediate_size=8,
            n_shared_experts=0,
            first_k_dense_replace=1,
            moe_layer_freq=1,
            rms_norm_eps=1e-6,
            tie_word_embeddings=True,
            enable_fp32_lm_head=False,
            rope_interleave=True,
            rope_theta=10000.0,
        ),
        vision_config=dict(
            vt_hidden_size=16,
            vt_intermediate_size=24,
            vt_num_hidden_layers=2,
            vt_num_attention_heads=2,
            patch_size=2,
            num_channels=3,
            merge_kernel_size=[2, 2],
            init_pos_emb_height=2,
            init_pos_emb_width=2,
            init_pos_emb_time=4,
            text_hidden_size=8,
            max_pos_emb_height=8,
            max_pos_emb_width=8,
        ),
    )


def language_weights():
    generator = torch.Generator().manual_seed(73)

    def normal(*shape):
        return torch.randn(shape, generator=generator) * 0.15

    root = "model.layers.0."
    return {
        "model.embed_tokens.weight": normal(16, 8),
        "lm_head.weight": normal(16, 8),
        "model.norm.weight": torch.ones(8),
        root + "input_layernorm.weight": torch.ones(8),
        root + "post_attention_layernorm.weight": torch.ones(8),
        root + "self_attn.q_a_proj.weight": normal(4, 8),
        root + "self_attn.q_a_layernorm.weight": torch.ones(4),
        root + "self_attn.q_b_proj.weight": normal(16, 4),
        root + "self_attn.kv_a_proj_with_mqa.weight": normal(6, 8),
        root + "self_attn.kv_a_layernorm.weight": torch.ones(4),
        root + "self_attn.kv_b_proj.weight": normal(16, 4),
        root + "self_attn.o_proj.weight": normal(8, 8),
        root + "mlp.gate_proj.weight": normal(16, 8),
        root + "mlp.up_proj.weight": normal(16, 8),
        root + "mlp.down_proj.weight": normal(8, 16),
    }


def checkpoint(directory, cfg, weights):
    with open(os.path.join(directory, "config.json"), "w") as stream:
        json.dump(cfg, stream)
    names, index = list(weights), {}
    for shard_id in range(2):
        filename = f"model-{shard_id + 1:05d}-of-00002.safetensors"
        shard = {name: weights[name] for name in names[shard_id::2]}
        save_file(shard, os.path.join(directory, filename))
        index.update({name: filename for name in shard})
    with open(os.path.join(directory, "model.safetensors.index.json"), "w") as stream:
        json.dump({"weight_map": index}, stream)


class TorchCausalMLA:
    def __init__(self, weights):
        self.kv_b = weights["model.layers.0.self_attn.kv_b_proj.weight"]

    def forward(self, q, compressed_kv, k_pe, kv_cache, layer_idx, topk_indices):
        length = q.shape[0]
        kv = F.linear(compressed_kv, self.kv_b).reshape(length, 4, 4)
        k_nope, v = kv.split(2, dim=-1)
        q_nope, q_pe = q.split(2, dim=-1)
        phase = torch.arange(length, dtype=q.dtype).reshape(length, 1, 1)

        def rope(x):
            return (
                x * phase.cos()
                + torch.stack((-x[..., 1], x[..., 0]), dim=-1) * phase.sin()
            )

        q = torch.cat((q_nope, rope(q_pe)), dim=-1)
        k = torch.cat((k_nope, rope(k_pe.unsqueeze(1)).expand(-1, 4, -1)), dim=-1)
        output = F.scaled_dot_product_attention(
            q.transpose(0, 1).unsqueeze(0),
            k.transpose(0, 1).unsqueeze(0),
            v.transpose(0, 1).unsqueeze(0),
            is_causal=True,
        )
        return output.squeeze(0).transpose(0, 1).reshape(length, 8)


def reference_forward(weights, hidden):
    def rms(x, name):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * weights[name]

    root = "model.layers.0."
    normed = rms(hidden, root + "input_layernorm.weight")
    q_a = F.linear(normed, weights[root + "self_attn.q_a_proj.weight"])
    q = F.linear(
        rms(q_a, root + "self_attn.q_a_layernorm.weight"),
        weights[root + "self_attn.q_b_proj.weight"],
    ).reshape(-1, 4, 4)
    kv_a = F.linear(normed, weights[root + "self_attn.kv_a_proj_with_mqa.weight"])
    compressed = rms(kv_a[:, :4], root + "self_attn.kv_a_layernorm.weight")
    attn = TorchCausalMLA(weights).forward(q, compressed, kv_a[:, 4:], None, 0, None)
    hidden = hidden + F.linear(attn, weights[root + "self_attn.o_proj.weight"])
    normed = rms(hidden, root + "post_attention_layernorm.weight")
    mlp = F.silu(F.linear(normed, weights[root + "mlp.gate_proj.weight"])) * F.linear(
        normed, weights[root + "mlp.up_proj.weight"]
    )
    hidden = hidden + F.linear(mlp, weights[root + "mlp.down_proj.weight"])
    return rms(hidden, "model.norm.weight")


class KimiLanguageTest(unittest.TestCase):
    def test_registered_language_class(self):
        self.assertIs(get_model_class("kimi_k25"), KimiK25ForCausalLM)

    def test_explicit_mm_service_uses_direct_projector_without_old_weight_loader(self):
        cfg = config()
        visual = KimiK25ForVisionEmbedding(cfg, load_config())
        weights = visual.state_dict()
        mm_params = VitParameters()
        mm_params.config = {"vision_config": cfg["vision_config"]}
        with tempfile.TemporaryDirectory() as directory:
            checkpoint(directory, cfg, weights)
            service = KimiK25Mixin.from_new_loader(
                torch.float32,
                "cpu",
                mm_params,
                NewLoaderLoadMethod.FASTSAFETENSORS,
                VitConfig(),
                directory,
            )
        self.assertIsNone(mm_params.vit_weights)
        self.assertFalse(hasattr(service, "mm_mixin_loader"))
        for name, value in service.mm_part.mm_projector.state_dict().items():
            torch.testing.assert_close(
                value, weights["mm_projector." + name].half().float(), rtol=0, atol=0
            )

    def test_actual_config_dense_moe_shared_expert_checkpoint_ownership(self):
        cfg = config()
        cfg["text_config"].update(
            num_hidden_layers=2,
            n_routed_experts=2,
            num_experts_per_tok=1,
            n_shared_experts=1,
        )
        source = language_weights()
        language = dict(source)
        for name, tensor in source.items():
            if name.startswith("model.layers.0.") and ".mlp." not in name:
                language[name.replace("layers.0.", "layers.1.")] = tensor.clone()
        root = "model.layers.1.mlp."
        language[root + "gate.weight"] = torch.randn(2, 8) * 0.1
        for expert in ("experts.0", "experts.1", "shared_experts"):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                language[root + expert + "." + projection + ".weight"] = (
                    torch.randn(8, 8) * 0.1
                )
        weights = {"language_model." + name: value for name, value in language.items()}
        with tempfile.TemporaryDirectory() as directory:
            checkpoint(directory, cfg, weights)
            model_config = KimiK25._create_config(directory)
            model_config.model_type = "kimi_k25"
            model_config.enable_fp32_lm_head = False
            model_config.attn_config.kernel_tokens_per_block = 64
            model = NewModelLoader(
                model_config, load_config(), model_path=directory
            ).load()
        moe = model.layers[1].mlp
        for expert in range(2):
            expected = torch.cat(
                [
                    language[root + f"experts.{expert}." + projection + ".weight"]
                    for projection in ("up_proj", "gate_proj")
                ]
            )
            torch.testing.assert_close(
                moe.experts.w13[expert], expected, rtol=0, atol=0
            )
            torch.testing.assert_close(
                moe.experts.w2[expert],
                language[root + f"experts.{expert}.down_proj.weight"],
                rtol=0,
                atol=0,
            )
        torch.testing.assert_close(
            moe.shared_experts.down_proj.weight,
            language[root + "shared_experts.down_proj.weight"],
            rtol=0,
            atol=0,
        )

    def test_actual_main_config_parser_and_tied_head_fallback(self):
        cfg = config()
        language = language_weights()
        del language["lm_head.weight"]
        weights = {"language_model." + key: value for key, value in language.items()}
        with tempfile.TemporaryDirectory() as directory:
            checkpoint(directory, cfg, weights)
            model_config = KimiK25._create_config(directory)
            model_config.model_type = "kimi_k25"
            model_config.enable_fp32_lm_head = False
            # Runtime cache block size is finalized after the checkpoint parser.
            model_config.attn_config.kernel_tokens_per_block = 64
            model = NewModelLoader(
                model_config, load_config(), model_path=directory
            ).load()
        torch.testing.assert_close(
            model.lm_head.weight, language["model.embed_tokens.weight"], rtol=0, atol=0
        )
        self.assertEqual(model.layers[0].mlp.intermediate_size, 16)

    def test_complete_sharded_vision_language_checkpoint_and_embeddings(self):
        for method in (
            NewLoaderLoadMethod.SCRATCH,
            NewLoaderLoadMethod.FASTSAFETENSORS,
        ):
            with self.subTest(method=method.value):
                for model, _, inputs, expected in complete_checkpoint_cases(self, method):
                    torch.testing.assert_close(model.embedding(inputs), expected)


def complete_checkpoint_cases(case, method):
    cfg = config()
    cfg["text_config"]["num_nextn_predict_layers"] = 1
    # Exercise the official 14x14 image preprocessor, with tiny tower widths.
    cfg["vision_config"]["patch_size"] = 14
    vision = KimiK25ForVisionEmbedding(cfg, load_config())
    torch.manual_seed(79)
    with torch.no_grad():
        for parameter in vision.parameters():
            parameter.uniform_(-0.1, 0.1)
    language = language_weights()
    weights = {"language_model." + key: value for key, value in language.items()}
    weights.update(
        {name: value.clone() for name, value in vision.state_dict().items()}
    )
    # An appended predictor layer is not part of the target text decoder.
    weights["language_model.model.layers.1.unused_predictor.weight"] = torch.ones(1)
    with tempfile.TemporaryDirectory() as directory:
        checkpoint(directory, cfg, weights)
        model_config = KimiK25._create_config(directory)
        model_config.model_type = "kimi_k25"
        model_config.data_type = "fp32"
        model_config.enable_fp32_lm_head = False
        model_config.attn_config.kernel_tokens_per_block = 64
        server = _CpuKimi.from_config(
            model_config=model_config,
            parallelism_config=ParallelismConfig(),
            hw_kernel_config=types.SimpleNamespace(
                enable_cuda_graph=False, use_swizzleA=False
            ),
            kv_cache_config=types.SimpleNamespace(
                multi_task_prompt="", multi_task_prompt_str=""
            ),
            fmha_config=None,
            moe_config=None,
            max_generate_batch_size=7,
            merge_lora=False,
            load_method=method.value,
            vit_config=VitConfig(),
        )
        model = server.py_model
        case.assertIs(server.checkpoint_owner, model)
        case.assertIsInstance(model, KimiK25ForCausalLM)
        case.assertIsInstance(server.weight, RuntimeWeightView)
        case.assertEqual(server.model_weights_loader.loaded_method, method)
        case.assertEqual(len(server.weight.weights), len(model.layers))
        for key, param in (
            ("embedding", model.embed_tokens.weight),
            ("final_layernorm.gamma", model.norm.weight),
            ("lm_head", model.lm_head.weight),
        ):
            case.assertIs(server.weight.get_global_weight(key), param)
        service_visual = MultimodalMixinFactory._create_multimodal_mixin(
            model_config,
            types.SimpleNamespace(
                load_config=types.SimpleNamespace(load_method=method.value)
            ),
            VitConfig(),
            device="cpu",
        )
        case.assertIsInstance(service_visual, KimiK25Mixin)
        case.assertFalse(hasattr(service_visual, "mm_mixin_loader"))
        visual = service_visual.mm_part
    case.assertEqual(len(model.layers), 1)
    torch.testing.assert_close(
        model.lm_head.weight, language["lm_head.weight"], rtol=0, atol=0
    )
    for grid in (None, [[1, 2, 2]], [[2, 2, 2]]):
        features = None
        if grid == [[1, 2, 2]]:
            image = Image.new("RGB", (28, 28), color=(17, 45, 63))
            features, positions = visual.embedding(image)
            case.assertIsNone(positions)
            case.assertEqual(visual.media_tokens_for(image), 1)
            case.assertEqual(tuple(features.shape), (1, 8))
        elif grid is not None:
            grid_tensor = torch.tensor(grid)
            with torch.inference_mode():
                pixels = torch.randn(int(grid_tensor.prod(-1).sum()), 3, 14, 14)
                features = mm_projector_forward(
                    visual.mm_projector, visual.vision_tower(pixels, grid_tensor)
                )[0]
        input_ids = torch.tensor([0, 2, 3, 15])
        expected_input = F.embedding(
            input_ids, language["model.embed_tokens.weight"]
        )
        mask = torch.ones(4, dtype=torch.bool)
        if features is not None:
            input_ids[1 : 1 + len(features)] = 17
            mask[1 : 1 + len(features)] = False
            expected_input[1 : 1 + len(features)] = features
        inputs = types.SimpleNamespace(
            input_ids=input_ids,
            capture_hidden_states=False,
            attention_inputs=object(),
            embedding_inputs=(
                None
                if features is None
                else types.SimpleNamespace(text_tokens_mask=mask)
            ),
            multimodal_inputs=(
                None
                if features is None
                else types.SimpleNamespace(
                    multimodal_features=[features],
                    mm_features_locs=torch.tensor([1]),
                    mm_extra_input=[],
                )
            ),
        )
        yield model, language, inputs, expected_input


if __name__ == "__main__":
    unittest.main()
