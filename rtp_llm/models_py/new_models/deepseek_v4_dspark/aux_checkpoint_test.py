"""Real 128x128 UE8M0 draft auxiliary staging; CPU staging is not inference."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.new_models.deepseek_v4_dspark.auxiliary import (
    DSparkAuxiliaryWeights,
)
from rtp_llm.models_py.weight_mapper import get_all_weights


def _materialize(folder, files, cfg, prefill, method):
    owner = DSparkAuxiliaryWeights(
        cfg["hidden_size"],
        cfg["dspark_target_layer_ids"],
        cfg["vocab_size"],
        cfg["vocab_size"],
        cfg["dspark_markov_rank"],
        prefill_commit_only=prefill,
    )
    loader = NewModelLoader(
        SimpleNamespace(model_type="deepseek_v4_dspark"),
        NewLoaderConfig(device="cpu", load_method=method),
        model_path=str(folder),
    )
    include = owner.checkpoint_weight_name_filter()
    resolve = loader._checkpoint_read_spec_resolver(owner, include, None)
    tensors = (
        loader._fastsafetensors_weights_iterator(files, "cpu", include, None, resolve)
        if method == "fastsafetensors"
        else get_all_weights(files, name_filter=include, read_spec_resolver=resolve)
    )
    owner.load_standalone_checkpoint(tensors)
    owner.validate_weights_loaded()
    return owner


class V4DraftAuxiliaryTest(unittest.TestCase):
    def test_nested_draft_projection_still_exposes_leaf_read_spec(self):
        root = RtpModule()
        root.auxiliary = DSparkAuxiliaryWeights(128, [1, 2], 16, 16, 3)
        target, plan = root._resolve_checkpoint_load_plan(
            root, "auxiliary.main_proj.scale", (1, 2)
        )
        self.assertIs(target.module, root.auxiliary.main_proj)
        self.assertEqual(plan.source_shape, (1, 2))
        with self.assertRaisesRegex(ValueError, "checkpoint shape"):
            root._resolve_checkpoint_load_plan(
                root, "auxiliary.main_proj.scale", (2, 1)
            )

    def test_scratch_role_io_and_exact_scale(self):
        self._check_role_io("scratch")

    @unittest.skipUnless(
        os.environ.get("NEWLOADER_TEST_NATIVE_FST") == "1",
        "requires explicitly provisioned native FST",
    )
    def test_native_fst_role_io_and_exact_scale(self):
        import fastsafetensors.cpp as native

        self.assertTrue(native.__file__.endswith(".so"))
        self._check_role_io("fastsafetensors")

    def _check_role_io(self, method):
        cfg = dict(
            hidden_size=128,
            dspark_target_layer_ids=[1, 2],
            vocab_size=16,
            dspark_markov_rank=3,
        )
        tensors = {
            "mtp.0.main_proj.weight": (
                torch.arange(128 * 256).reshape(128, 256) % 11
            ).to(torch.float8_e4m3fn),
            "mtp.0.main_proj.scale": torch.tensor([[126, 130]], dtype=torch.uint8).view(
                torch.float8_e8m0fnu
            ),
            "mtp.0.main_norm.weight": torch.arange(128).bfloat16(),
            "mtp.2.markov_head.markov_w1.weight": torch.arange(48)
            .reshape(16, 3)
            .bfloat16(),
            "mtp.2.markov_head.markov_w2.weight": (
                torch.arange(48).reshape(16, 3) + 5
            ).bfloat16(),
            "layers.0.attn.wkv.weight": torch.zeros(3),
        }
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "model.safetensors")
            save_file(tensors, path)
            owners = [
                (prefill, _materialize(folder, [path], cfg, prefill, method))
                for prefill in (False, True)
            ]
        for prefill, owner in owners:
            for name in filter(owner.checkpoint_weight_name_filter(), tensors):
                exact = owner.get_parameter(owner.WEIGHTS_MAPPER.map_name(name))
                self.assertTrue(
                    torch.equal(
                        exact.view(torch.uint8), tensors[name].view(torch.uint8)
                    ),
                    name,
                )
            self.assertEqual(len(list(owner.parameters())), 3 if prefill else 5)
            self.assertEqual(owner.markov_head is None, prefill)
            with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
                owner.main_proj.validate_runtime_device("cpu")

    @unittest.skipUnless(
        os.environ.get("V4_DSPARK_RELEASE_PATH"), "requires real smoke checkpoint"
    )
    def test_actual_release_ue8m0_norm_and_sampler(self):
        folder = Path(os.environ["V4_DSPARK_RELEASE_PATH"])
        cfg = json.loads((folder / "config.json").read_text())
        index = json.loads((folder / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        names = (
            "mtp.0.main_proj.weight",
            "mtp.0.main_proj.scale",
            "mtp.0.main_norm.weight",
            "mtp.2.markov_head.markov_w1.weight",
            "mtp.2.markov_head.markov_w2.weight",
        )
        paths = sorted({str(folder / index[name]) for name in names})
        method = os.environ.get("V4_DSPARK_RELEASE_METHOD", "scratch")
        self.assertIn(method, ("scratch", "fastsafetensors"))
        owner = _materialize(folder, paths, cfg, False, method)
        for name in names:
            actual = owner.get_parameter(owner.WEIGHTS_MAPPER.map_name(name))
            with safe_open(
                str(folder / index[name]), framework="pt", device="cpu"
            ) as sf:
                expected = sf.get_tensor(name)
                self.assertEqual(actual.dtype, expected.dtype)
                self.assertTrue(
                    torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)),
                    name,
                )


if __name__ == "__main__":
    unittest.main()
