"""Execute both real transports on the deployed sampler layout (not a full draft)."""

import gc
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from rtp_llm.models_py.checkpoint_reader_test_support import checkpoint_readers
from rtp_llm.models_py.model_loader import (
    NewLoaderConfig,
    NewLoaderLoadMethod,
    NewModelLoader,
)
from rtp_llm.models_py.new_models.qwen3_dspark_weights import Qwen3DSparkMarkovHead
from rtp_llm.models_py.registry import register_model


@register_model("_qwen3_dspark_sampler_test")
class _Sampler(Qwen3DSparkMarkovHead):
    fastsafetensors_load_isolated = True

    def __init__(self, config, load_config):
        super().__init__(config.vocab_size, config.draft_vocab, config.rank)


class SamplerTest(unittest.TestCase):
    def test_sampler_padding_is_not_a_logical_vocabulary_row(self):
        a = torch.arange(20, dtype=torch.float32).reshape(10, 2)
        b = torch.arange(256, dtype=torch.float32).reshape(128, 2)
        model = Qwen3DSparkMarkovHead(8, 5, 2)
        model.load_weights({"markov_head.markov_w1.weight": a, "markov_head.markov_w2.weight": b})
        torch.testing.assert_close(model.markov_w1.weight, a[:8].bfloat16())
        torch.testing.assert_close(model.markov_w2.weight, b[:5].bfloat16())
        for rows in (4, 129):
            with self.assertRaisesRegex(ValueError, "Shape mismatch"):
                Qwen3DSparkMarkovHead(8, 5, 2).load_weights(
                    {"markov_head.markov_w2.weight": torch.ones(rows, 2)}
                )

    def load(self, root, vocab, draft, rank, method):
        cfg = SimpleNamespace(
            model_type="_qwen3_dspark_sampler_test",
            vocab_size=vocab,
            draft_vocab=draft,
            rank=rank,
        )
        loader = NewModelLoader(
            cfg, NewLoaderConfig(device="cpu", load_method=method), model_path=str(root)
        )
        model = loader.load()
        self.assertEqual(loader._selected_load_method, NewLoaderLoadMethod(method))
        return model

    def test_real_scratch_and_fst_distinct_vocabulary(self):
        a = torch.arange(32, dtype=torch.bfloat16).reshape(8, 4)
        b = torch.arange(20, dtype=torch.bfloat16).reshape(5, 4) + 91
        with tempfile.TemporaryDirectory() as root:
            save_file(
                {
                    "markov_head.markov_w1.weight": a,
                    "markov_head.markov_w2.weight": b,
                    "fc.weight": torch.ones(2, 3),
                },
                str(Path(root) / "model.safetensors"),
            )
            models = []
            for method in checkpoint_readers("scratch", "fastsafetensors"):
                if method == "fastsafetensors":
                    import fastsafetensors.cpp as native

                    self.assertTrue(native.__file__.endswith(".so"))
                models.append(self.load(root, 8, 5, 4, method))
        gc.collect()
        _ = torch.empty(32 * 1024)
        for model in models:
            self.assertTrue(torch.equal(model.markov_w1.weight, a))
            self.assertTrue(torch.equal(model.markov_w2.weight, b))
            self.assertIs(
                model.runtime_weight_view()["dspark_markov_w1.weight"],
                model.markov_w1.weight,
            )
            self.assertIs(
                model.runtime_weight_view()["dspark_markov_w2.weight"],
                model.markov_w2.weight,
            )

    @unittest.skipUnless(
        os.environ.get("QWEN3_DSPARK_RELEASE"),
        "set release path to load real Markov tables",
    )
    def test_release_two_real_tables(self):
        root = Path(os.environ["QWEN3_DSPARK_RELEASE"])
        cfg = json.loads((root / "config.json").read_text())
        model = self.load(
            root,
            cfg["vocab_size"],
            cfg.get("draft_vocab_size", cfg["vocab_size"]),
            cfg["markov_rank"],
            "scratch",
        )
        with safe_open(
            str(root / "model.safetensors"), framework="pt", device="cpu"
        ) as ckpt:
            for name, param in model.named_parameters():
                self.assertTrue(
                    torch.equal(param, ckpt.get_tensor("markov_head." + name)), name
                )


if __name__ == "__main__":
    unittest.main()
