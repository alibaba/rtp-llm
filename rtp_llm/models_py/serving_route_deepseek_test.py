"""Direct serving publication under main's actual DeepSeek/GLM ModelConfig ABI."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from rtp_llm.models.deepseek_v2 import DeepSeekV2
from rtp_llm.models.direct_checkpoint import load_owned_model
from rtp_llm.models.glm4_moe_lite import Glm4MoeLite
from rtp_llm.models_py.checkpoint_reader_test_support import checkpoint_readers
from rtp_llm.models_py.model_loader import NewLoaderLoadMethod
from rtp_llm.models_py.new_models.deepseek_v3.language import DeepSeekV32ForCausalLM
from rtp_llm.models_py.serving_route_checkpoint_test import service
from rtp_llm.utils.model_weight import W


class DeepSeekServiceRouteTest(unittest.TestCase):
    def test_main_config_to_direct_runtime_layout(self):
        for model_type, parser, q_rank in (
            ("deepseek2", DeepSeekV2, None),
            ("glm4_moe_lite", Glm4MoeLite, 4),
        ):
            for method in checkpoint_readers("scratch", "fastsafetensors"):
                with self.subTest(model=model_type, method=method), tempfile.TemporaryDirectory() as root:
                    cfg = dict(
                        model_type=model_type, vocab_size=16, hidden_size=8,
                        num_attention_heads=2, num_hidden_layers=1, intermediate_size=12,
                        q_lora_rank=q_rank, kv_lora_rank=4, qk_nope_head_dim=2,
                        qk_rope_head_dim=2, v_head_dim=2, rope_theta=10000,
                        torch_dtype="bfloat16", n_shared_experts=1,
                        moe_intermediate_size=4, n_routed_experts=4, num_experts_per_tok=2,
                        routed_scaling_factor=1.0, first_k_dense_replace=1,
                    )
                    (Path(root) / "config.json").write_text(json.dumps(cfg))
                    shapes = {
                        "model.embed_tokens.weight": (16, 8),
                        "lm_head.weight": (16, 8),
                        "model.norm.weight": (8,),
                    }
                    prefix = "model.layers.0."
                    for name, shape in {
                        "input_layernorm.weight": (8,),
                        "post_attention_layernorm.weight": (8,),
                        "self_attn.kv_a_proj_with_mqa.weight": (6, 8),
                        "self_attn.kv_a_layernorm.weight": (4,),
                        "self_attn.kv_b_proj.weight": (8, 4),
                        "self_attn.o_proj.weight": (8, 4),
                        "mlp.gate_proj.weight": (12, 8),
                        "mlp.up_proj.weight": (12, 8),
                        "mlp.down_proj.weight": (8, 12),
                    }.items():
                        shapes[prefix + name] = shape
                    prefix += "self_attn."
                    if q_rank:
                        shapes[prefix + "q_a_proj.weight"] = (4, 8)
                        shapes[prefix + "q_b_proj.weight"] = (8, 4)
                        shapes[prefix + "q_a_layernorm.weight"] = (4,)
                    else:
                        shapes[prefix + "q_proj.weight"] = (8, 8)
                    generator = torch.Generator().manual_seed(733)
                    weights = {
                        k: torch.randn(shape, generator=generator).bfloat16() for k, shape in shapes.items()
                    }
                    save_file(weights, str(Path(root) / "model.safetensors"))
                    config = parser._create_config(root)
                    config.ckpt_path = root
                    config.model_type = model_type
                    # Match the engine configuration step after parsing the checkpoint.
                    config.attn_config.tokens_per_block = config.attn_config.kernel_tokens_per_block = 64
                    config.max_seq_len = 16
                    config.data_type = "bf16"
                    config.lora_infos = {}
                    fixture = service(root, method, model_type=model_type)
                    fixture.model_config = config
                    load_owned_model(fixture)
                    model = fixture.py_model
                    self.assertIsInstance(model, DeepSeekV32ForCausalLM)
                    self.assertEqual(fixture.model_weights_loader.loaded_method, NewLoaderLoadMethod(method))
                    self.assertIs(fixture.weight.get_global_weight("embedding"), model.embed_tokens.weight)
                    self.assertTrue(model.initialize(SimpleNamespace(kv_cache=None)))
                    layout = model.weight.weights[0]
                    k, v = weights[prefix + "kv_b_proj.weight"].reshape(2, 4, 4).split(2, dim=1)
                    torch.testing.assert_close(layout[W.mla_kc], k)
                    torch.testing.assert_close(layout[W.mla_vc], v.transpose(1, 2))
                    self.assertIs(layout[W.mla_kc], model.layers[0].self_attn._kc_w)


if __name__ == "__main__":
    unittest.main()
