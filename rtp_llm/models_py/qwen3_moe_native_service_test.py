"""Public from_config regression requiring the real main native runtime ABI."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from rtp_llm.models.qwen_v3_moe import Qwen3Moe
from rtp_llm.models_py.checkpoint_reader_test_support import checkpoint_readers
from rtp_llm.models_py.model_loader import NewLoaderLoadMethod
from rtp_llm.models_py.new_models.qwen3_moe.language import Qwen3MoeForCausalLM
from rtp_llm.models_py.qwen3_moe_checkpoint_test import weights
from rtp_llm.models_py.serving_route_checkpoint_test import service


class _Server(Qwen3Moe):
    def _get_device_str(self):
        return "cpu"

    def load_tokenizer(self):
        pass

    def _init_custom_module(self):
        return None


class Qwen3MoeServingTest(unittest.TestCase):
    def test_real_registered_owner(self):
        for method in checkpoint_readers("scratch", "fastsafetensors"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as root:
                path = Path(root)
                (path / "config.json").write_text(
                    json.dumps(
                        {
                            "model_type": "qwen3_moe",
                            "num_hidden_layers": 1,
                            "hidden_size": 4,
                            "intermediate_size": 4,
                            "num_attention_heads": 2,
                            "num_key_value_heads": 1,
                            "vocab_size": 8,
                            "moe_intermediate_size": 4,
                            "num_experts": 4,
                            "num_experts_per_tok": 2,
                            "decoder_sparse_step": 1,
                            "norm_topk_prob": True,
                        }
                    )
                )
                data = weights()
                save_file(data, str(path / "model.safetensors"))
                config = Qwen3Moe._create_config(root)
                config.model_type = "qwen_3_moe"
                config.data_type = "fp32"
                fixture = service(root, method)
                fixture.parallelism_config.local_world_size = 1
                fixture.parallelism_config.pp_stage_layer_counts = []
                model = _Server.from_config(
                    model_config=config,
                    parallelism_config=fixture.parallelism_config,
                    kv_cache_config=SimpleNamespace(
                        multi_task_prompt="", multi_task_prompt_str=""
                    ),
                    fmha_config=None,
                    moe_config=None,
                    max_generate_batch_size=7,
                    hw_kernel_config=SimpleNamespace(
                        enable_cuda_graph=False, use_swizzleA=False
                    ),
                    load_method=method,
                    merge_lora=False,
                    vit_config=None,
                )
                self.assertIsInstance(model.py_model, Qwen3MoeForCausalLM)
                self.assertEqual(
                    model.model_weights_loader.loaded_method,
                    NewLoaderLoadMethod(method),
                )
                self.assertIs(
                    model.py_model.lm_head.weight,
                    model.weight.get_global_weight("lm_head"),
                )
                self.assertTrue(
                    torch.equal(model.py_model.lm_head.weight, data["lm_head.weight"])
                )
                self.assertTrue(
                    torch.equal(
                        model.py_model.layers[0].mlp.experts.w2[3],
                        data["model.layers.0.mlp.experts.3.down_proj.weight"],
                    )
                )


if __name__ == "__main__":
    unittest.main()
