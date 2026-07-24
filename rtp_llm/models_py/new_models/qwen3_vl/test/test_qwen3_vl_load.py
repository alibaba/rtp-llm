import tempfile
import types
import unittest
from unittest import mock

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.new_models.qwen3_vl import (
    Qwen3VLForCausalLM as ExportedQwen3VLForCausalLM,
)
from rtp_llm.models_py.new_models.qwen3_vl.model import Qwen3VLForCausalLM
from rtp_llm.models_py.new_models.qwen3_vl.vision import Qwen3VLForVisionEmbedding
from rtp_llm.models_py.quant_methods import QuantizationConfig


def _parallelism(ep_size=1):
    return types.SimpleNamespace(
        tp_size=1,
        tp_rank=0,
        ep_size=ep_size,
        ep_rank=0,
        prefill_cp_config=types.SimpleNamespace(
            is_enabled=lambda: False,
            is_prefill_enabled=lambda: False,
        ),
        ffn_disaggregate_config=types.SimpleNamespace(enable_ffn_disaggregate=False),
        get_attn_tp_size=lambda: 1,
        get_attn_tp_rank=lambda: 0,
        get_ffn_tp_size=lambda: 1,
        get_ffn_tp_rank=lambda: 0,
    )


def _load_config(dtype=torch.float32, ep_size=1) -> NewLoaderConfig:
    return NewLoaderConfig(
        compute_dtype=dtype,
        device="cpu",
        quant_config=QuantizationConfig("none"),
        parallelism_config=_parallelism(ep_size),
        ep_size=ep_size,
        ep_rank=0,
    )


def _language_config():
    return types.SimpleNamespace(
        model_type="qwen3_vl",
        num_layers=1,
        vocab_size=8,
        hidden_size=4,
        inter_size=4,
        attn_config=types.SimpleNamespace(
            head_num=2,
            kv_head_num=1,
            size_per_head=2,
        ),
        layernorm_eps=1e-6,
        enable_fp32_lm_head=False,
        tie_word_embeddings=True,
        expert_num=0,
    )


def _language_weights():
    prefix = "model.language_model."
    return {
        prefix
        + "embed_tokens.weight": torch.arange(32, dtype=torch.float32).reshape(8, 4),
        prefix + "layers.0.input_layernorm.weight": torch.ones(4),
        prefix
        + "layers.0.self_attn.q_proj.weight": torch.arange(
            16, dtype=torch.float32
        ).reshape(4, 4),
        prefix
        + "layers.0.self_attn.k_proj.weight": torch.arange(
            8, dtype=torch.float32
        ).reshape(2, 4),
        prefix
        + "layers.0.self_attn.v_proj.weight": torch.arange(
            8, 16, dtype=torch.float32
        ).reshape(2, 4),
        prefix + "layers.0.self_attn.q_norm.weight": torch.ones(2),
        prefix + "layers.0.self_attn.k_norm.weight": torch.ones(2),
        prefix + "layers.0.self_attn.o_proj.weight": torch.eye(4),
        prefix + "layers.0.post_attention_layernorm.weight": torch.ones(4),
        prefix + "layers.0.mlp.gate_proj.weight": torch.ones(4, 4),
        prefix + "layers.0.mlp.up_proj.weight": torch.full((4, 4), 2.0),
        prefix + "layers.0.mlp.down_proj.weight": torch.ones(4, 4),
        prefix + "norm.weight": torch.ones(4),
    }


def _vision_config():
    return {
        "depth": 1,
        "hidden_size": 8,
        "hidden_act": "gelu_pytorch_tanh",
        "intermediate_size": 16,
        "num_heads": 2,
        "in_channels": 1,
        "patch_size": 2,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
        "out_hidden_size": 6,
        "num_position_embeddings": 16,
        "deepstack_visual_indexes": [0],
    }


def _vision_model_config():
    return {
        "model_type": "qwen3_vl_vision",
        "vision_config": _vision_config(),
    }


def _manual_merger(merger, hidden_states):
    if merger.use_postshuffle_norm:
        hidden_states = F.layer_norm(
            hidden_states.reshape(-1, 32),
            (32,),
            merger.norm.weight,
            merger.norm.bias,
            merger.norm.eps,
        )
    else:
        hidden_states = F.layer_norm(
            hidden_states,
            (8,),
            merger.norm.weight,
            merger.norm.bias,
            merger.norm.eps,
        ).reshape(-1, 32)
    hidden_states = F.gelu(
        F.linear(
            hidden_states,
            merger.linear_fc1.weight,
            merger.linear_fc1.bias,
        )
    )
    return F.linear(
        hidden_states,
        merger.linear_fc2.weight,
        merger.linear_fc2.bias,
    )


def _manual_vision_forward(model, pixel_values):
    visual = model.visual
    hidden_states = F.conv3d(
        pixel_values.reshape(-1, 1, 2, 2, 2),
        visual.patch_embed.proj.weight,
        visual.patch_embed.proj.bias,
        stride=(2, 2, 2),
    ).reshape(-1, 8)
    position_indexes = torch.tensor([0, 3, 12, 15] * 2)
    hidden_states = hidden_states + visual.pos_embed(position_indexes)

    block = visual.blocks[0]
    normed = F.layer_norm(
        hidden_states,
        (8,),
        block.norm1.weight,
        block.norm1.bias,
        block.norm1.eps,
    )
    qkv = F.linear(normed, block.attn.qkv.weight, block.attn.qkv.bias)
    query, key, value = qkv.reshape(8, 3, 2, 4).permute(1, 0, 2, 3).unbind(0)
    positions = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]] * 2)
    cos = positions.cos().unsqueeze(1).repeat(1, 1, 2)
    sin = positions.sin().unsqueeze(1).repeat(1, 1, 2)

    def rotate_half(tensor):
        first, second = tensor.chunk(2, dim=-1)
        return torch.cat((-second, first), dim=-1)

    query = query * cos + rotate_half(query) * sin
    key = key * cos + rotate_half(key) * sin
    attention_outputs = []
    for start in (0, 4):
        end = start + 4
        attention = F.scaled_dot_product_attention(
            query[start:end].transpose(0, 1).unsqueeze(0),
            key[start:end].transpose(0, 1).unsqueeze(0),
            value[start:end].transpose(0, 1).unsqueeze(0),
            dropout_p=0.0,
        )
        attention_outputs.append(attention.squeeze(0).transpose(0, 1))
    attention = torch.cat(attention_outputs).reshape(8, 8)
    hidden_states = hidden_states + F.linear(
        attention,
        block.attn.proj.weight,
        block.attn.proj.bias,
    )

    normed = F.layer_norm(
        hidden_states,
        (8,),
        block.norm2.weight,
        block.norm2.bias,
        block.norm2.eps,
    )
    mlp = F.gelu(
        F.linear(normed, block.mlp.linear_fc1.weight, block.mlp.linear_fc1.bias),
        approximate="tanh",
    )
    hidden_states = hidden_states + F.linear(
        mlp,
        block.mlp.linear_fc2.weight,
        block.mlp.linear_fc2.bias,
    )
    return (
        _manual_merger(visual.merger, hidden_states),
        [_manual_merger(visual.deepstack_merger_list[0], hidden_states)],
    )


def _vision_checkpoint(model):
    return {
        f"model.{name}": tensor.detach().clone()
        for name, tensor in model.state_dict().items()
    }


class Qwen3VLNewLoaderTest(unittest.TestCase):
    def test_package_export_resolves_language_model_lazily(self):
        self.assertIs(ExportedQwen3VLForCausalLM, Qwen3VLForCausalLM)

    def test_language_loader_filters_visual_tensors_before_dispatch(self):
        weights = _language_weights()
        weights["model.visual.not_a_language_weight"] = torch.ones(3)
        with tempfile.TemporaryDirectory() as model_path:
            save_file(weights, f"{model_path}/model.safetensors")
            model = NewModelLoader(
                model_config=_language_config(),
                load_config=_load_config(),
                model_path=model_path,
            ).load()

        self.assertIsInstance(model, Qwen3VLForCausalLM)
        torch.testing.assert_close(model.lm_head.weight, model.embed_tokens.weight)
        self.assertNotIn("visual", dict(model.named_modules()))

    def test_unknown_language_tensor_is_not_silently_ignored(self):
        weights = _language_weights()
        weights["model.language_model.layers.0.typo.weight"] = torch.ones(1)
        with tempfile.TemporaryDirectory() as model_path:
            save_file(weights, f"{model_path}/model.safetensors")
            with self.assertRaisesRegex(RuntimeError, "typo"):
                NewModelLoader(
                    model_config=_language_config(),
                    load_config=_load_config(),
                    model_path=model_path,
                ).load()

    def test_missing_language_tensor_is_not_silently_ignored(self):
        weights = _language_weights()
        del weights["model.language_model.layers.0.self_attn.q_proj.weight"]
        with tempfile.TemporaryDirectory() as model_path:
            save_file(weights, f"{model_path}/model.safetensors")
            with self.assertRaisesRegex(RuntimeError, r"qkv_proj.*weight"):
                NewModelLoader(
                    model_config=_language_config(),
                    load_config=_load_config(),
                    model_path=model_path,
                ).load()

    def test_dense_language_ignores_runtime_ep_topology(self):
        with tempfile.TemporaryDirectory() as model_path:
            save_file(_language_weights(), f"{model_path}/model.safetensors")
            with self.assertLogs(
                "rtp_llm.models_py.model_loader", level="WARNING"
            ) as logs:
                model = NewModelLoader(
                    model_config=_language_config(),
                    load_config=_load_config(ep_size=2),
                    model_path=model_path,
                ).load()

        self.assertIsInstance(model, Qwen3VLForCausalLM)
        torch.testing.assert_close(model.lm_head.weight, model.embed_tokens.weight)
        self.assertTrue(
            any("treating the model as dense" in message for message in logs.output)
        )

    def test_language_forward_handles_text_only_input(self):
        class IdentityLayer(torch.nn.Module):
            def forward(self, hidden_states, fmha_impl, kv_cache=None):
                return hidden_states

        model = Qwen3VLForCausalLM.__new__(Qwen3VLForCausalLM)
        torch.nn.Module.__init__(model)
        model.embed_tokens = torch.nn.Embedding(8, 4)
        model.layers = torch.nn.ModuleList([IdentityLayer()])
        model.norm = torch.nn.Identity()
        model.kv_cache = None
        model.multimodal_embedding_injector = (
            lambda hidden_states, features, locations: hidden_states
        )
        model.multimodal_deepstack_injector = (
            lambda hidden_states, deepstack, locations, layer_id: hidden_states
        )
        input_ids = torch.tensor([1, 2, 3])
        inputs = types.SimpleNamespace(
            input_ids=input_ids,
            embedding_inputs=types.SimpleNamespace(text_tokens_mask=None),
            multimodal_inputs=types.SimpleNamespace(
                multimodal_features=[],
                mm_features_locs=torch.empty(0, dtype=torch.long),
                mm_extra_input=[],
            ),
            attention_inputs=object(),
        )
        fmha_impl = types.SimpleNamespace(fmha_params=None)

        with mock.patch(
            "rtp_llm.models_py.new_models.qwen3_vl.model." "select_block_map_for_layer"
        ):
            outputs = model(inputs, fmha_impl)

        torch.testing.assert_close(outputs.hidden_states, model.embed_tokens(input_ids))

    def test_language_forward_injects_multimodal_features_and_deepstack(self):
        class IdentityLayer(torch.nn.Module):
            def forward(self, hidden_states, fmha_impl, kv_cache=None):
                return hidden_states

        class EmbeddingInjector:
            def __init__(self):
                self.input_at_mm_location = None

            def __call__(self, hidden_states, features, locations):
                loc = int(locations.item())
                self.input_at_mm_location = hidden_states[loc].clone()
                output = hidden_states.clone()
                output[loc] = features[0]
                return output

        class DeepstackInjector:
            def __init__(self):
                self.calls = []

            def __call__(self, hidden_states, deepstack, locations, layer_id):
                self.calls.append((tuple(locations), layer_id))
                output = hidden_states.clone()
                output[locations[0]] += deepstack[layer_id]
                return output

        model = Qwen3VLForCausalLM.__new__(Qwen3VLForCausalLM)
        torch.nn.Module.__init__(model)
        model.embed_tokens = torch.nn.Embedding(8, 4)
        with torch.no_grad():
            model.embed_tokens.weight.copy_(
                torch.arange(32, dtype=torch.float32).reshape(8, 4)
            )
        model.layers = torch.nn.ModuleList([IdentityLayer(), IdentityLayer()])
        model.norm = torch.nn.Identity()
        model.kv_cache = None
        embedding_injector = EmbeddingInjector()
        deepstack_injector = DeepstackInjector()
        model.multimodal_embedding_injector = embedding_injector
        model.multimodal_deepstack_injector = deepstack_injector

        input_ids = torch.tensor([1, 99, 2])
        text_mask = torch.tensor([True, False, True])
        mm_feature = torch.tensor([10.0, 20.0, 30.0, 40.0])
        mm_feature_locs = torch.tensor([1])
        deepstack = [
            torch.tensor([1.0, 2.0, 3.0, 4.0]),
            torch.tensor([5.0, 6.0, 7.0, 8.0]),
        ]
        inputs = types.SimpleNamespace(
            input_ids=input_ids,
            embedding_inputs=types.SimpleNamespace(text_tokens_mask=text_mask),
            multimodal_inputs=types.SimpleNamespace(
                multimodal_features=[mm_feature],
                mm_features_locs=mm_feature_locs,
                mm_extra_input=[object()],
            ),
            attention_inputs=object(),
        )
        fmha_impl = types.SimpleNamespace(fmha_params=None)

        with mock.patch(
            "rtp_llm.models_py.new_models.qwen3_vl.model."
            "reshape_extra_input_to_deepstack",
            return_value=deepstack,
        ), mock.patch(
            "rtp_llm.models_py.new_models.qwen3_vl.model." "select_block_map_for_layer"
        ) as select_block_map:
            outputs = model(inputs, fmha_impl)

        torch.testing.assert_close(
            embedding_injector.input_at_mm_location, torch.zeros(4)
        )
        expected = model.embed_tokens(torch.tensor([1, 0, 2])).detach()
        expected[1] = mm_feature + deepstack[0] + deepstack[1]
        torch.testing.assert_close(outputs.hidden_states, expected)
        self.assertEqual(deepstack_injector.calls, [((1,), 0), ((1,), 1)])
        self.assertEqual(select_block_map.call_count, 2)

    def test_vision_loader_matches_independent_cpu_reference(self):
        torch.manual_seed(7)
        source = Qwen3VLForVisionEmbedding(
            _vision_model_config(), _load_config()
        ).eval()
        weights = _vision_checkpoint(source)
        weights["model.language_model.unused.weight"] = torch.ones(3)
        with tempfile.TemporaryDirectory() as model_path:
            save_file(weights, f"{model_path}/model.safetensors")
            loaded = NewModelLoader(
                model_config=_vision_model_config(),
                load_config=_load_config(),
                model_path=model_path,
            ).load()

        pixel_values = torch.arange(64, dtype=torch.float32).reshape(8, 8) / 63
        grid_thw = torch.tensor([[2, 2, 2]], dtype=torch.int64)
        with torch.inference_mode():
            actual_output, actual_deepstack = loaded(pixel_values, grid_thw)
            expected_output, expected_deepstack = _manual_vision_forward(
                loaded, pixel_values
            )
        torch.testing.assert_close(actual_output, expected_output, atol=1e-6, rtol=1e-5)
        self.assertEqual(len(actual_deepstack), 1)
        torch.testing.assert_close(
            actual_deepstack[0], expected_deepstack[0], atol=1e-6, rtol=1e-5
        )

    def test_vision_loader_rejects_unknown_and_missing_tensors(self):
        source = Qwen3VLForVisionEmbedding(
            _vision_model_config(), _load_config()
        ).eval()
        weights = _vision_checkpoint(source)
        with tempfile.TemporaryDirectory() as model_path:
            unexpected = dict(weights)
            unexpected["model.visual.blocks.0.attn.typo.weight"] = torch.ones(1)
            save_file(unexpected, f"{model_path}/model.safetensors")
            with self.assertRaisesRegex(RuntimeError, "typo"):
                NewModelLoader(
                    model_config=_vision_model_config(),
                    load_config=_load_config(),
                    model_path=model_path,
                ).load()

        with tempfile.TemporaryDirectory() as model_path:
            missing = dict(weights)
            del missing["model.visual.blocks.0.attn.qkv.weight"]
            save_file(missing, f"{model_path}/model.safetensors")
            with self.assertRaisesRegex(RuntimeError, "attn.qkv.weight"):
                NewModelLoader(
                    model_config=_vision_model_config(),
                    load_config=_load_config(),
                    model_path=model_path,
                ).load()

    def test_vision_forward_handles_video_and_heterogeneous_grids(self):
        torch.manual_seed(13)
        model = Qwen3VLForVisionEmbedding(_vision_model_config(), _load_config()).eval()
        pixel_values = torch.linspace(-1.0, 1.0, 128).reshape(16, 8)
        grid_thw = torch.tensor([[2, 2, 2], [1, 4, 2]], dtype=torch.int64)
        with torch.inference_mode():
            combined, combined_deepstack = model(pixel_values, grid_thw)
            first, first_deepstack = model(pixel_values[:8], grid_thw[:1])
            second, second_deepstack = model(pixel_values[8:], grid_thw[1:])

        torch.testing.assert_close(
            combined, torch.cat((first, second)), atol=1e-6, rtol=1e-5
        )
        self.assertEqual(len(combined_deepstack), 1)
        torch.testing.assert_close(
            combined_deepstack[0],
            torch.cat((first_deepstack[0], second_deepstack[0])),
            atol=1e-6,
            rtol=1e-5,
        )

    def test_vision_forward_rejects_invalid_grid_and_patch_count(self):
        model = Qwen3VLForVisionEmbedding(_vision_model_config(), _load_config()).eval()
        with self.assertRaisesRegex(ValueError, "at least one"):
            model(torch.zeros(0, 8), torch.empty((0, 3), dtype=torch.int64))
        with self.assertRaisesRegex(ValueError, "align"):
            model(torch.zeros(6, 8), torch.tensor([[1, 2, 3]]))
        with self.assertRaisesRegex(ValueError, "describes 4"):
            model(torch.zeros(3, 8), torch.tensor([[1, 2, 2]]))
        with self.assertRaisesRegex(TypeError, "integer dtype"):
            model(torch.zeros(4, 8), torch.tensor([[1.0, 2.0, 2.0]]))

    def test_vision_config_rejects_empty_deepstack_indexes(self):
        config = _vision_model_config()
        config["vision_config"] = dict(config["vision_config"])
        config["vision_config"]["deepstack_visual_indexes"] = []
        with self.assertRaisesRegex(ValueError, "at least one layer"):
            Qwen3VLForVisionEmbedding(config, _load_config())

    def test_vision_loader_matches_legacy_fp16_staging_for_runtime_dtypes(self):
        torch.manual_seed(19)
        source = Qwen3VLForVisionEmbedding(
            _vision_model_config(), _load_config()
        ).eval()
        weights = _vision_checkpoint(source)
        key = "model.visual.blocks.0.attn.qkv.weight"
        weights[key] = torch.linspace(
            -0.0001,
            0.0001,
            weights[key].numel(),
            dtype=torch.float32,
        ).reshape_as(weights[key])
        with tempfile.TemporaryDirectory() as model_path:
            save_file(weights, f"{model_path}/model.safetensors")
            for runtime_dtype in (
                torch.float16,
                torch.bfloat16,
                torch.float32,
            ):
                with self.subTest(runtime_dtype=runtime_dtype):
                    loaded = NewModelLoader(
                        model_config=_vision_model_config(),
                        load_config=_load_config(runtime_dtype),
                        model_path=model_path,
                    ).load()
                    expected = weights[key].to(torch.float16).to(runtime_dtype)
                    torch.testing.assert_close(
                        loaded.visual.blocks[0].attn.qkv.weight,
                        expected,
                    )
                    if runtime_dtype == torch.bfloat16:
                        self.assertFalse(
                            torch.equal(expected, weights[key].to(torch.bfloat16))
                        )
                    before = loaded.visual.blocks[0].attn.qkv.weight.detach().clone()
                    loaded.visual.process_weights_after_loading()
                    torch.testing.assert_close(
                        loaded.visual.blocks[0].attn.qkv.weight,
                        before,
                    )

    def test_vision_loader_rejects_fp16_staging_overflow(self):
        source = Qwen3VLForVisionEmbedding(
            _vision_model_config(), _load_config()
        ).eval()
        weights = _vision_checkpoint(source)
        key = "model.visual.blocks.0.attn.qkv.weight"
        weights[key] = torch.full_like(weights[key], 70000, dtype=torch.bfloat16)
        with tempfile.TemporaryDirectory() as model_path:
            save_file(weights, f"{model_path}/model.safetensors")
            with self.assertRaisesRegex(ValueError, "FP16-compatible range"):
                NewModelLoader(
                    model_config=_vision_model_config(),
                    load_config=_load_config(torch.bfloat16),
                    model_path=model_path,
                ).load()


if __name__ == "__main__":
    unittest.main()
