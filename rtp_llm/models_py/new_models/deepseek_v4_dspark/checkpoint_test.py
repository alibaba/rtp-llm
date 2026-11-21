"""V4 DSpARK release routing and sampler parameter loading (not a GPU smoke)."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from rtp_llm.models_py.model_loader import (
    NewLoaderConfig,
    NewLoaderLoadMethod,
    NewModelLoader,
)
from rtp_llm.models_py.new_models.deepseek_v4_dspark.layout import (
    DSparkV4CheckpointLayout,
)
from rtp_llm.models_py.new_models.deepseek_v4_dspark.markov_head import DSparkMarkovHead
from rtp_llm.models_py.registry import register_model


@register_model("_test_v4_flash_dspark_markov")
class _Sampler(DSparkMarkovHead):
    def __init__(self, config, load_config):
        super().__init__(
            config.vocab_size, config.vocab_size, config.dspark_markov_rank
        )
        self.proposal_layout = config.proposal_layout

    def checkpoint_weight_name_filter(self):
        if self.proposal_layout:
            return DSparkV4CheckpointLayout(False).includes
        return super().checkpoint_weight_name_filter()


def _load(folder, vocab, rank, method, *, proposal_layout=False):
    loader = NewModelLoader(
        SimpleNamespace(
            model_type="_test_v4_flash_dspark_markov",
            vocab_size=vocab,
            dspark_markov_rank=rank,
            proposal_layout=proposal_layout,
        ),
        NewLoaderConfig(device="cpu", load_method=method),
        model_path=str(folder),
    )
    model = loader.load()
    assert loader._selected_load_method == NewLoaderLoadMethod(method)
    return model


_TABLES = tuple(f"mtp.2.markov_head.markov_w{i}.weight" for i in (1, 2))
_COMMIT = {
    "mtp.0.main_proj.weight",
    "mtp.0.main_proj.scale",
    "mtp.0.main_norm.weight",
    *(
        f"mtp.{i}.attn.{kind}"
        for i in range(3)
        for kind in ("wkv.weight", "wkv.scale", "kv_norm.weight")
    ),
}
_GLOBALS = {"head.weight", "embed.weight"}


class V4DSparkCheckpointTest(unittest.TestCase):
    def test_prefill_exact_live_checkpoint_weights_and_no_target(self):
        inventory = (
            _GLOBALS
            | _COMMIT
            | set(_TABLES)
            | {
                "layers.42.attn.wkv.weight",
                "mtp.3.attn.wkv.weight",
                "mtp.0.attn.wq_a.weight",
                "mtp.2.norm.weight",
                "mtp.1.ffn.experts.12.w2.scale",
                "mtp.2.confidence_head.proj.weight",
                "mtp.2.confidence_head.unknown.weight",
                "norm.weight",
            }
        )
        prefill = DSparkV4CheckpointLayout(True)
        decode = DSparkV4CheckpointLayout(False)
        self.assertEqual(set(filter(prefill.includes, inventory)), _GLOBALS | _COMMIT)
        self.assertEqual(
            set(filter(decode.includes, inventory)),
            inventory
            - {
                "layers.42.attn.wkv.weight",
                "mtp.3.attn.wkv.weight",
                "norm.weight",
                "mtp.2.confidence_head.proj.weight",
            },
        )
        self.assertNotIn(_TABLES[0], set(filter(prefill.auxiliary, inventory)))
        self.assertTrue(all(map(decode.auxiliary, _TABLES)))
        for bad in (None, 0, "PREFILL"):
            with self.assertRaises(TypeError):
                DSparkV4CheckpointLayout(bad)

    def test_real_sampler_scratch(self):
        self._check_sampler("scratch")

    def test_proposal_load_ignores_only_the_unused_release_head(self):
        tensors = {name: torch.ones(16, 3).bfloat16() for name in _TABLES}
        tensors["mtp.2.confidence_head.proj.weight"] = torch.ones(1, 8)
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = str(Path(folder) / "model.safetensors")
            save_file(tensors, checkpoint)
            model = _load(folder, 16, 3, "scratch", proposal_layout=True)
            torch.testing.assert_close(model.markov_w2.weight, tensors[_TABLES[1]])
            tensors["mtp.2.confidence_head.unknown.weight"] = torch.ones(1)
            save_file(tensors, checkpoint)
            with self.assertRaisesRegex(RuntimeError, "confidence_head.unknown.weight"):
                _load(folder, 16, 3, "scratch", proposal_layout=True)

    @unittest.skipUnless(
        os.getenv("NEWLOADER_TEST_NATIVE_FST") == "1", "requires real FST"
    )
    def test_real_sampler_fst(self):
        import fastsafetensors.cpp as native

        self.assertTrue(native.__file__.endswith(".so"))
        self._check_sampler("fastsafetensors")

    def _check_sampler(self, method):
        tensors = {
            name: (torch.arange(48).reshape(16, 3) + 3 * i).bfloat16()
            for i, name in enumerate(_TABLES)
        }
        tensors["mtp.0.main_norm.weight"] = torch.zeros(19)
        with tempfile.TemporaryDirectory() as folder:
            save_file(tensors, str(Path(folder) / "model.safetensors"))
            model = _load(folder, 16, 3, method)
        for i, source in enumerate(_TABLES, 1):
            actual = model.runtime_weight_view()[f"dspark_markov_w{i}.weight"]
            self.assertIs(actual, model.get_parameter(f"markov_w{i}.weight"))
            torch.testing.assert_close(actual, tensors[source], rtol=0, atol=0)

    @unittest.skipUnless(
        os.getenv("V4_DSPARK_RELEASE_PATH"), "requires real release checkpoint"
    )
    def test_release_commit_file_selection_and_actual_sampler_weights(self):
        release = Path(os.environ["V4_DSPARK_RELEASE_PATH"])
        config = json.loads((release / "config.json").read_text())
        index = json.loads((release / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        # Exclude the two shared globals which a production target lends.
        prefill = set(filter(DSparkV4CheckpointLayout(True).includes, index)) - _GLOBALS
        self.assertEqual(prefill, _COMMIT)
        self.assertNotIn(
            "mtp.2.confidence_head.proj.weight",
            set(filter(DSparkV4CheckpointLayout(False).includes, index)),
        )
        self.assertEqual(len({index[name] for name in prefill}), 3)
        self.assertEqual(
            len(config["compress_ratios"]) - config["num_hidden_layers"], 3
        )
        self.assertTrue(
            set(_TABLES).issubset(
                set(filter(DSparkV4CheckpointLayout(False).includes, index))
            )
        )
        model = _load(
            release,
            config["vocab_size"],
            config["dspark_markov_rank"],
            os.getenv("V4_DSPARK_RELEASE_METHOD", "scratch"),
        )
        for i, source in enumerate(_TABLES, 1):
            with safe_open(
                str(release / index[source]), framework="pt", device="cpu"
            ) as opened:
                torch.testing.assert_close(
                    model.get_parameter(f"markov_w{i}.weight"),
                    opened.get_tensor(source),
                    rtol=0,
                    atol=0,
                )


if __name__ == "__main__":
    unittest.main()
