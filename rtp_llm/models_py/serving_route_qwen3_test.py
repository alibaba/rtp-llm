"""Public BaseModel.from_config -> Qwen3 owner, on a real main native ABI."""

import os
import tempfile
import unittest
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from rtp_llm.models.downstream_modules.custom_module import CustomHandler, CustomModule
from rtp_llm.models.downstream_modules.reranker.qwen3_reranker import (
    Qwen3RerankerHandler,
)
from rtp_llm.models.qwen_v3 import QwenV3
from rtp_llm.models_py.model_loader import NewLoaderLoadMethod
from rtp_llm.models_py.new_models.qwen3.language import Qwen3ForCausalLM
from rtp_llm.models_py.new_models.qwen3.main_pp_checkpoint_test import (
    checkpoint,
    model_config,
)
from rtp_llm.models_py.serving_route_checkpoint_test import service


class CheckpointOnlyQwen3(QwenV3):
    """No tokenizer or GPU kernel needed for this weight-loading regression."""

    def load_tokenizer(self):
        pass

    def _get_device_str(self):
        return "cpu"

    def _init_custom_module(self):
        return None


class Qwen3ServiceRouteTest(unittest.TestCase):
    def _load(self, root, method, cls=CheckpointOnlyQwen3):
        fixture = service(root, method)
        config = model_config()
        config.model_type = "qwen_3"
        config.ckpt_path = root
        config.quant_config = None
        config.lora_infos = {}
        config.generate_env_config = None
        config.compute_dtype = torch.float32
        config.data_type = torch.float32
        config.eplb_config = fixture.model_config.eplb_config
        config.enable_output_vocab_pruning = False
        return cls.from_config(
            model_config=config,
            kv_cache_config=SimpleNamespace(
                multi_task_prompt="", multi_task_prompt_str=""
            ),
            parallelism_config=fixture.parallelism_config,
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

    def test_explicit_projection_chain_loads_from_checkpoint(self):
        class Processor(CustomHandler):
            def checkpoint_projection_layers(self):
                return (
                    ("business.input", "business.offset"),
                    ("business.output", None),
                )

            def init(self, tensors):
                self.tensors = tensors

            def score(self, hidden):
                params = self.tensors
                return torch.nn.functional.linear(
                    torch.nn.functional.linear(
                        hidden, params["business.input"], params["business.offset"]
                    ).tanh(),
                    params["business.output"],
                )

        class ScoredQwen3(CheckpointOnlyQwen3):
            def _init_custom_module(self):
                custom = CustomModule(self.model_config, self.tokenizer)
                custom.handler = Processor(self.model_config)
                return custom

        for method in ("scratch", "fastsafetensors"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as root:
                path = os.path.join(root, "model.safetensors")
                data = checkpoint()
                weights = {
                    "business.output": torch.tensor([[1.0, -2.0]]),
                    "business.offset": torch.tensor([1.0, -1.0]),
                    "business.input": torch.tensor(
                        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
                    ),
                }
                save_file({**data, **weights}, path)
                model = self._load(root, method, ScoredQwen3)
                self.assertEqual(
                    model.model_weights_loader.loaded_method,
                    NewLoaderLoadMethod(method),
                )
                head = model.custom_module.handler
                self.assertIs(
                    head.tensors["business.output"],
                    model.weight.get_global_weight("__custom__.business.output"),
                )
                value = torch.tensor([[2.0, 3.0, 4.0, 5.0]])
                torch.testing.assert_close(
                    head.score(value),
                    torch.tanh(value[:, :1] + 1) - 2 * torch.tanh(value[:, 1:2] - 1),
                )
                for key, wrong, message in (
                    ("business.input", torch.ones(2, 5), "input width 4"),
                    ("business.output", torch.ones(1, 4), "input width 2"),
                    ("business.offset", torch.ones(3), "expected 2 outputs"),
                    (
                        "business.output",
                        torch.ones(1, 2, dtype=torch.int32),
                        "Invalid downstream tensor",
                    ),
                ):
                    with self.subTest(key=key, message=message):
                        save_file({**data, **weights, key: wrong}, path)
                        with self.assertRaisesRegex(ValueError, message):
                            self._load(root, method, ScoredQwen3)

                class Collision(ScoredQwen3):
                    def _init_custom_module(self):
                        custom = super()._init_custom_module()
                        custom.handler.checkpoint_projection_layers = lambda: (
                            ("model.layers.0.self_attn.q_proj.weight", None),
                        )
                        return custom

                with self.assertRaisesRegex(ValueError, "Ambiguous handler projection"):
                    self._load(root, method, Collision)

    def test_shared_reranker_source_reaches_both_consumers(self):
        class RerankerQwen3(CheckpointOnlyQwen3):
            def _init_custom_module(self):
                custom = CustomModule(self.model_config, self.tokenizer)
                custom.handler = Qwen3RerankerHandler(self.model_config, 0, 1)
                return custom

        for method in ("scratch", "fastsafetensors"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as root:
                data = checkpoint()
                save_file(data, os.path.join(root, "model.safetensors"))
                model = self._load(root, method, RerankerQwen3)
                weight = data["model.embed_tokens.weight"]
                torch.testing.assert_close(
                    model.py_model.embed_tokens.weight.cpu(), weight
                )
                torch.testing.assert_close(
                    model.custom_module.handler.linear.weight.cpu(), weight
                )

    def test_scratch_and_native_fst(self):
        for method in ("scratch", "fastsafetensors"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as root:
                data = checkpoint()
                save_file(data, os.path.join(root, "model.safetensors"))
                model = self._load(root, method)
                self.assertIsInstance(model.py_model, Qwen3ForCausalLM)
                self.assertEqual(
                    model.model_weights_loader.loaded_method,
                    NewLoaderLoadMethod(method),
                )
                self.assertIs(
                    model.py_model.embed_tokens.weight,
                    model.weight.get_global_weight("embedding"),
                )
                torch.testing.assert_close(
                    model.py_model.embed_tokens.weight,
                    data["model.embed_tokens.weight"],
                )
                self.assertTrue(callable(model.weight_manager.update))
                self.assertIsNotNone(model.rank_local_lora)


if __name__ == "__main__":
    unittest.main()
