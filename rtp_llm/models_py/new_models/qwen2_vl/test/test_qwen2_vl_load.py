import tempfile
import types
import unittest

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.new_models.qwen2_vl.model import Qwen2VLForCausalLM
from rtp_llm.models_py.new_models.qwen2_vl.vision import Qwen2VLForVisionEmbedding
from rtp_llm.models_py.quant_methods import QuantizationConfig


def _parallelism():
    return types.SimpleNamespace(
        tp_size=1,
        tp_rank=0,
        ep_size=1,
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


def _load_config() -> NewLoaderConfig:
    return NewLoaderConfig(
        compute_dtype=torch.float32,
        device="cpu",
        quant_config=QuantizationConfig("none"),
        parallelism_config=_parallelism(),
    )


def _language_config():
    return types.SimpleNamespace(
        model_type="qwen2_vl",
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
    )


def _language_weights():
    return {
        "model.embed_tokens.weight": torch.arange(32, dtype=torch.float32).reshape(
            8, 4
        ),
        "model.layers.0.input_layernorm.weight": torch.ones(4),
        "model.layers.0.self_attn.q_proj.weight": torch.arange(
            16, dtype=torch.float32
        ).reshape(4, 4),
        "model.layers.0.self_attn.k_proj.weight": torch.arange(
            8, dtype=torch.float32
        ).reshape(2, 4),
        "model.layers.0.self_attn.v_proj.weight": torch.arange(
            8, 16, dtype=torch.float32
        ).reshape(2, 4),
        "model.layers.0.self_attn.q_proj.bias": torch.full((4,), 0.5),
        "model.layers.0.self_attn.k_proj.bias": torch.full((2,), -0.25),
        "model.layers.0.self_attn.v_proj.bias": torch.full((2,), 0.75),
        "model.layers.0.self_attn.o_proj.weight": torch.eye(4),
        "model.layers.0.post_attention_layernorm.weight": torch.ones(4),
        "model.layers.0.mlp.gate_proj.weight": torch.ones(4, 4),
        "model.layers.0.mlp.up_proj.weight": torch.full((4, 4), 2.0),
        "model.layers.0.mlp.down_proj.weight": torch.ones(4, 4),
        "model.norm.weight": torch.ones(4),
    }


def _vision_config():
    return {
        "depth": 1,
        "embed_dim": 8,
        "hidden_size": 6,
        "hidden_act": "quick_gelu",
        "mlp_ratio": 2.0,
        "num_heads": 2,
        "in_channels": 1,
        "patch_size": 2,
        "spatial_merge_size": 2,
        "temporal_patch_size": 1,
    }


def _vision_model_config():
    return {
        "model_type": "qwen2_vl_vision",
        "vision_config": _vision_config(),
    }


def _manual_vision_forward(model, pixel_values):
    visual = model.visual
    patch = visual.patch_embed
    hidden = F.conv3d(
        pixel_values.reshape(-1, 1, 1, 2, 2),
        patch.proj.weight,
        stride=(1, 2, 2),
    ).reshape(-1, 8)

    block = visual.blocks[0]
    normed = F.layer_norm(
        hidden,
        (8,),
        block.norm1.weight,
        block.norm1.bias,
        block.norm1.eps,
    )
    qkv = F.linear(normed, block.attn.qkv.weight, block.attn.qkv.bias)
    q, k, v = qkv.reshape(8, 3, 2, 4).permute(1, 0, 2, 3).unbind(0)
    positions = torch.tensor(
        [[0, 0], [0, 1], [1, 0], [1, 1]] * 2,
        dtype=torch.float32,
    )
    cos = positions.cos().unsqueeze(1).repeat(1, 1, 2)
    sin = positions.sin().unsqueeze(1).repeat(1, 1, 2)

    def rotate_half(tensor):
        first, second = tensor.chunk(2, dim=-1)
        return torch.cat((-second, first), dim=-1)

    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    attention_outputs = []
    for start in (0, 4):
        end = start + 4
        attention = F.scaled_dot_product_attention(
            q[start:end].transpose(0, 1).unsqueeze(0),
            k[start:end].transpose(0, 1).unsqueeze(0),
            v[start:end].transpose(0, 1).unsqueeze(0),
            dropout_p=0.0,
        )
        attention_outputs.append(attention.squeeze(0).transpose(0, 1))
    attention = torch.cat(attention_outputs).reshape(8, 8)
    hidden = hidden + F.linear(attention, block.attn.proj.weight, block.attn.proj.bias)
    normed = F.layer_norm(
        hidden,
        (8,),
        block.norm2.weight,
        block.norm2.bias,
        block.norm2.eps,
    )
    mlp_hidden = F.linear(normed, block.mlp.fc1.weight, block.mlp.fc1.bias)
    mlp_hidden = mlp_hidden * torch.sigmoid(1.702 * mlp_hidden)
    hidden = hidden + F.linear(mlp_hidden, block.mlp.fc2.weight, block.mlp.fc2.bias)

    merger = visual.merger
    merged = F.layer_norm(
        hidden,
        (8,),
        merger.ln_q.weight,
        merger.ln_q.bias,
        merger.ln_q.eps,
    ).reshape(-1, 32)
    merged = F.gelu(F.linear(merged, merger.mlp[0].weight, merger.mlp[0].bias))
    return F.linear(merged, merger.mlp[2].weight, merger.mlp[2].bias)


class Qwen2VLNewLoaderTest(unittest.TestCase):
    def test_language_loader_filters_visual_tensors_before_dispatch(self):
        weights = _language_weights()
        weights["visual.not_a_language_weight"] = torch.ones(3)
        with tempfile.TemporaryDirectory() as model_path:
            save_file(weights, f"{model_path}/model.safetensors")
            model = NewModelLoader(
                model_config=_language_config(),
                load_config=_load_config(),
                model_path=model_path,
            ).load()

        self.assertIsInstance(model, Qwen2VLForCausalLM)
        torch.testing.assert_close(model.lm_head.weight, model.embed_tokens.weight)
        self.assertNotIn("visual", dict(model.named_modules()))

    def test_vision_loader_streams_only_visual_weights_and_matches_reference(self):
        torch.manual_seed(7)
        source = Qwen2VLForVisionEmbedding(
            _vision_model_config(), _load_config()
        ).eval()
        weights = {
            name: tensor.detach().clone()
            for name, tensor in source.state_dict().items()
        }
        weights["model.unused_language_weight"] = torch.ones(3)

        with tempfile.TemporaryDirectory() as model_path:
            save_file(weights, f"{model_path}/model.safetensors")
            loaded = NewModelLoader(
                model_config=_vision_model_config(),
                load_config=_load_config(),
                model_path=model_path,
            ).load()

        for name, expected in source.state_dict().items():
            torch.testing.assert_close(loaded.state_dict()[name], expected)
        pixel_values = torch.arange(32, dtype=torch.float32).reshape(8, 4) / 31
        grid_thw = torch.tensor([[1, 2, 2], [1, 2, 2]], dtype=torch.int64)
        with torch.inference_mode():
            actual = loaded(pixel_values, grid_thw)
            expected = _manual_vision_forward(loaded, pixel_values)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    def test_vision_loader_rejects_unknown_visual_tensor(self):
        source = Qwen2VLForVisionEmbedding(
            _vision_model_config(), _load_config()
        ).eval()
        weights = {
            name: tensor.detach().clone()
            for name, tensor in source.state_dict().items()
        }
        weights["visual.blocks.0.attn.typo.weight"] = torch.ones(1)
        with tempfile.TemporaryDirectory() as model_path:
            save_file(weights, f"{model_path}/model.safetensors")
            with self.assertRaisesRegex(RuntimeError, "typo"):
                NewModelLoader(
                    model_config=_vision_model_config(),
                    load_config=_load_config(),
                    model_path=model_path,
                ).load()

    def test_vision_loader_rejects_missing_required_tensor(self):
        source = Qwen2VLForVisionEmbedding(
            _vision_model_config(), _load_config()
        ).eval()
        weights = {
            name: tensor.detach().clone()
            for name, tensor in source.state_dict().items()
            if name != "visual.blocks.0.attn.qkv.weight"
        }
        with tempfile.TemporaryDirectory() as model_path:
            save_file(weights, f"{model_path}/model.safetensors")
            with self.assertRaisesRegex(RuntimeError, "attn.qkv.weight"):
                NewModelLoader(
                    model_config=_vision_model_config(),
                    load_config=_load_config(),
                    model_path=model_path,
                ).load()

    def test_vision_forward_rejects_inconsistent_grid(self):
        model = Qwen2VLForVisionEmbedding(_vision_model_config(), _load_config())
        with self.assertRaisesRegex(ValueError, "at least one"):
            model(torch.zeros(0, 4), torch.empty((0, 3), dtype=torch.int64))
        with self.assertRaisesRegex(ValueError, "describes 4"):
            model(torch.zeros(3, 4), torch.tensor([[1, 2, 2]]))
        with self.assertRaisesRegex(ValueError, "spatial_merge_size"):
            model(torch.zeros(6, 4), torch.tensor([[1, 2, 3]]))
        with self.assertRaisesRegex(TypeError, "integer dtype"):
            model(torch.zeros(4, 4), torch.tensor([[1.0, 2.0, 2.0]]))

    def test_vision_config_rejects_invalid_rotary_head_dimension(self):
        config = _vision_config()
        config["embed_dim"] = 12
        with self.assertRaisesRegex(ValueError, "head_dim=6 must be divisible by 4"):
            Qwen2VLForVisionEmbedding(
                {"model_type": "qwen2_vl_vision", "vision_config": config},
                _load_config(),
            )


if __name__ == "__main__":
    unittest.main()
