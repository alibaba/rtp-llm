"""Public VisionBERT discovery and trained visual-weight loading contracts."""

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from safetensors.torch import save_file

from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.model_factory import ModelFactory
from rtp_llm.model_loader.load_config import LoadMethod
from rtp_llm.model_loader.multimodal_mixin_loader import MultimodalMixinLoader
from rtp_llm.multimodal.multimodal_mixin_factory import MultimodalMixinFactory


class VisionBertTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.checkpoint_path = Path(self.directory.name)
        self.checkpoint_config = {
            "architectures": ["VisionBert"],
            "hidden_size": 6,
            "num_attention_heads": 2,
            "num_hidden_layers": 1,
            "intermediate_size": 10,
            "max_position_embeddings": 17,
            "vocab_size": 128,
            "type_vocab_size": 2,
            "layer_norm_eps": 0.03,
            "vision_seq_len": 3,
            "vision_hidden_size": 5,
            "vision_embedding_opt": True,
        }
        (self.checkpoint_path / "config.json").write_text(
            json.dumps(self.checkpoint_config)
        )

    def config(self):
        model_cls = ModelFactory.get_model_cls("vision_bert")
        config = model_cls._create_config(str(self.checkpoint_path))
        config.model_type = "vision_bert"
        return config

    def test_public_factory_and_tokenizer_without_internal_models(self):
        # A fresh interpreter proves discovery works without importing the model
        # early or allowing legacy internal registration to provide it.
        script = textwrap.dedent(
            """
            import importlib.abc
            import sys
            from unittest import mock

            import rtp_llm.utils.import_util as import_util

            class RejectInternalImports(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname == "internal_source" or fullname.startswith("internal_source."):
                        raise AssertionError("public VisionBERT imported " + fullname)

            sys.meta_path.insert(0, RejectInternalImports())
            with mock.patch.object(import_util, "has_internal_source", return_value=False):
                from rtp_llm.model_factory_register import ModelDict

                assert "rtp_llm.models.vision_bert.vision_bert" not in sys.modules
                assert ModelDict.get_ft_model_type_by_config(
                    {"architectures": ["VisionBert"]}
                ) == "vision_bert"
                assert "rtp_llm.models.vision_bert.vision_bert" not in sys.modules

                from rtp_llm.model_factory import ModelFactory
                model_cls = ModelFactory.get_model_cls("vision_bert")
                assert model_cls.__module__ == "rtp_llm.models.vision_bert.vision_bert"

                from rtp_llm.frontend.tokenizer_factory.tokenizer_factory_register import (
                    _tokenizer_factory,
                    ensure_tokenizer_registered,
                )
                assert ensure_tokenizer_registered("vision_bert")
                assert _tokenizer_factory["vision_bert"].__module__ == (
                    "rtp_llm.frontend.tokenizer_factory.tokenizers.bert_tokenizer"
                )
                from rtp_llm.multimodal.multimodal_mixin_register import get_multimodal_mixin_cls
                assert get_multimodal_mixin_cls("vision_bert").__module__ == (
                    "rtp_llm.multimodal.multimodal_mixins.vision_bert_mixin"
                )
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_factory_preserves_checkpoint_and_multimodal_config(self):
        from rtp_llm.models.bert_weight import BertWeightInfo

        model_cls = ModelFactory.get_model_cls("vision_bert")
        self.assertIs(model_cls.get_weight_cls(), BertWeightInfo)
        config = self.config()
        self.assertEqual(config.hidden_size, 6)
        self.assertEqual(config.max_seq_len, 17)
        self.assertAlmostEqual(config.layernorm_eps, 0.03)
        self.assertFalse(config.attn_config.is_causal)
        self.assertTrue(config.mm_model_config.is_multimodal)
        self.assertTrue(config.mm_model_config.include_sep_tokens)
        self.assertEqual(config.mm_model_config.mm_sep_tokens, [[-1, -2]])
        self.assertEqual(config.mm_related_params.preprocess_batch_size, -1)
        self.assertEqual(config.mm_related_params.config["vision_seq_len"], 3)
        self.assertEqual(config.mm_related_params.config["vision_hidden_size"], 5)
        self.assertEqual(
            config.mm_related_params.config["ckpt_path"], str(self.checkpoint_path)
        )

    def test_real_visual_checkpoint_load_and_projection(self):
        # Non-square marker weights catch prefix, transpose and random-init
        # regressions while the public factory drives the production loader.
        weight = torch.arange(30, dtype=torch.float32).reshape(6, 5) / 13 - 0.6
        bias = torch.arange(6, dtype=torch.float32) / 7 - 0.2
        gamma = 1 + torch.arange(6, dtype=torch.float32) / 17
        beta = torch.arange(6, dtype=torch.float32) / 19 - 0.5
        save_file(
            {
                "vision_embedding.vision_embedding.weight": weight,
                "vision_embedding.vision_embedding.bias": bias,
                "vision_embedding.LayerNorm.weight": gamma,
                "vision_embedding.LayerNorm.bias": beta,
                "bert.embeddings.word_embeddings.weight": torch.zeros(128, 6),
            },
            str(self.checkpoint_path / "model.safetensors"),
        )
        config = self.config()
        engine_config = SimpleNamespace(
            load_config=SimpleNamespace(load_method=LoadMethod.AUTO)
        )
        # The checkpoint path stays real; CPU loading does not need CUDA
        # allocator cleanup even when the test runner has a visible GPU.
        with mock.patch.object(MultimodalMixinLoader, "force_clean_cuda_memory"):
            mixin = MultimodalMixinFactory._create_multimodal_mixin(
                config, engine_config, VitConfig(), device="cpu"
            )
        part = mixin.mm_part
        dtype = config.compute_dtype
        self.assertEqual(part.vision_embedding.weight.dtype, dtype)
        for actual, expected in (
            (part.vision_embedding.weight, weight),
            (part.vision_embedding.bias, bias),
            (part.LayerNorm.weight, gamma),
            (part.LayerNorm.bias, beta),
        ):
            torch.testing.assert_close(actual, expected.to(dtype), rtol=0, atol=0)
        self.assertEqual(part.LayerNorm.eps, self.checkpoint_config["layer_norm_eps"])
        features = [
            torch.arange(15, dtype=torch.float32).reshape(3, 5) / 9,
            torch.arange(15, 30, dtype=torch.float32).reshape(3, 5) / 11,
        ]
        output, extra = part.embedding(features)
        self.assertIsNone(extra)
        self.assertEqual([tuple(value.shape) for value in output], [(3, 6), (3, 6)])
        # Evaluate the training projection and LayerNorm formula independently
        # using the loaded dtype's rounded inputs and weights.
        projected = (
            (
                torch.cat(features).to(dtype).float() @ weight.to(dtype).float().T
                + bias.to(dtype).float()
            )
            .to(dtype)
            .float()
        )
        mean = projected.mean(-1, keepdim=True)
        variance = (projected - mean).square().mean(-1, keepdim=True)
        expected = (
            (projected - mean)
            * torch.rsqrt(variance + self.checkpoint_config["layer_norm_eps"])
            * gamma.to(dtype).float()
            + beta.to(dtype).float()
        ).to(dtype)
        torch.testing.assert_close(torch.cat(output), expected, rtol=2e-3, atol=2e-3)


if __name__ == "__main__":
    unittest.main()
