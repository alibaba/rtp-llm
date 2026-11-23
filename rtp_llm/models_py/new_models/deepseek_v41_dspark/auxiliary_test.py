"""Real V4.1 draft auxiliary checkpoint protocol, not CUDA execution."""

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.new_models.deepseek_v41_dspark.auxiliary import (
    V41DSparkAuxiliaryWeights,
)
from rtp_llm.models_py.weight_mapper import get_all_weights


def payload():
    return {
        "mtp.0.main_proj.weight": torch.ones(128, 384, dtype=torch.float8_e4m3fn),
        "mtp.0.main_proj.scale": torch.full((4, 12), 127, dtype=torch.uint8).view(
            torch.float8_e8m0fnu
        ),
        "mtp.0.main_norm.weight": torch.ones(128, dtype=torch.bfloat16),
        "mtp.2.markov_head.embed.weight": torch.arange(64).reshape(16, 4).bfloat16(),
        "mtp.2.markov_head.head.weight": torch.arange(64, 128)
        .reshape(16, 4)
        .bfloat16(),
    }


@unittest.skipUnless(
    hasattr(torch, "float8_e8m0fnu"), "the draft projection requires the UE8M0 dtype"
)
class V41DSparkAuxiliaryTest(unittest.TestCase):
    def test_cuda_drafts_bind_the_runtime_norm_to_the_checkpoint_parameter(self):
        from unittest.mock import patch

        from safetensors.torch import save_file

        from rtp_llm.models_py.modules.base.cuda.norm import RMSNorm as RuntimeNorm
        from rtp_llm.models_py.new_models.deepseek_v4_dspark.auxiliary import (
            DSparkAuxiliaryWeights,
        )

        key = "mtp.0.main_norm.weight"
        weight = torch.linspace(0.25, 1.75, 128).bfloat16()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "norm.safetensors"
            save_file({key: weight}, str(path))
            # Select the production CUDA binding; checkpoint reads stay on CPU.
            with patch.object(torch.version, "cuda", "13.0"):
                v4 = DSparkAuxiliaryWeights(128, (1,), 16, 16, 4, device="cuda")
                v41 = V41DSparkAuxiliaryWeights(128, (1,), 16, 4, device="cuda")
            for model, dispatch in (
                (v4, v4.load_standalone_checkpoint),
                (v41, v41.load_weights),
            ):
                with self.subTest(owner=type(model).__name__):
                    self.assertIsInstance(model.main_norm, RuntimeNorm)
                    self.assertIs(type(model.main_norm).forward, RuntimeNorm.forward)
                    param = model.main_norm.weight
                    method = (
                        "scratch"
                        if os.environ.get("NEWLOADER_TEST_NATIVE_FST") == "0"
                        else "fastsafetensors"
                    )
                    loader = NewModelLoader(
                        SimpleNamespace(model_type="deepseek_v41_dspark"),
                        NewLoaderConfig(device="cpu", load_method=method),
                        model_path=folder,
                    )
                    include = model.checkpoint_weight_name_filter()
                    resolver = loader._checkpoint_read_spec_resolver(
                        model, include, None
                    )
                    reader = (
                        loader._fastsafetensors_weights_iterator
                        if method == "fastsafetensors"
                        else get_all_weights
                    )
                    dispatch(reader([str(path)], "cpu", include, None, resolver))
                    self.assertIs(model.main_norm.weight, param)
                    self.assertTrue(torch.equal(param, weight))
                    self.assertEqual(model.main_norm.variance_epsilon, 1e-6)
                    self.assertTrue(model.main_norm.supports_independent_weight_updates)
                    updates = model.main_norm.prepare_weight_updates(
                        {"weight": torch.zeros_like(weight)}
                    )
                    self.assertEqual(len(updates), 1)
                    self.assertTrue(torch.equal(param, weight))

    def _read(self, model, values):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pytorch_model.bin"
            torch.save(values, path)
            loader = NewModelLoader(
                SimpleNamespace(model_type="deepseek_v41_dspark"),
                NewLoaderConfig(device="cpu", load_method="scratch"),
                model_path=directory,
            )
            include = model.checkpoint_weight_name_filter()
            resolver = loader._checkpoint_read_spec_resolver(model, include, None)
            model.load_weights(
                get_all_weights(
                    [str(path)], name_filter=include, read_spec_resolver=resolver
                )
            )
            NewModelLoader._validate_loaded_weights(model)

    def test_actual_32block_projection_and_release_markov_names(self):
        model = V41DSparkAuxiliaryWeights(128, (0, 4, 9), 16, 4)
        values = payload()
        self._read(model, values)
        self.assertEqual(tuple(model.main_proj.scale.shape), (4, 12))
        self.assertEqual(model.main_proj.scale.dtype, torch.float8_e8m0fnu)
        torch.testing.assert_close(
            model.main_proj.scale.view(torch.uint8),
            values["mtp.0.main_proj.scale"].view(torch.uint8),
        )
        self.assertIsNone(model.main_proj._executor)
        self.assertIs(
            model.runtime_weight_view()["dspark_markov_w1.weight"],
            model.markov_head.markov_w1.weight,
        )
        self.assertEqual(len(list(model.named_parameters())), 5)
        with self.assertRaisesRegex(RuntimeError, "CUDA|GPU"):
            model.main_proj.validate_runtime_device("cpu")

    def test_prefill_role_owns_no_sampler_or_proposal_global(self):
        model = V41DSparkAuxiliaryWeights(
            128, (0, 4, 9), 16, 4, prefill_commit_only=True
        )
        values = payload()
        values["mtp.2.markov_head.embed.weight"] = torch.ones(1)
        values["head.weight"] = torch.ones(1)
        values["mtp.2.hc_head_fn"] = torch.ones(1)
        self._read(model, values)
        self.assertIsNone(model.markov_head)
        self.assertEqual(model.runtime_weight_view(), {})
        self.assertEqual(
            set(dict(model.named_parameters())),
            {"main_proj.weight", "main_proj.scale", "main_norm.weight"},
        )

    def test_v4_scale_geometry_or_missing_scale_is_not_accepted(self):
        for scale in (None, torch.ones(1, 3).to(torch.float8_e8m0fnu)):
            values = payload()
            if scale is None:
                del values["mtp.0.main_proj.scale"]
            else:
                values["mtp.0.main_proj.scale"] = scale
            model = V41DSparkAuxiliaryWeights(128, (0, 4, 9), 16, 4)
            with self.assertRaises((ValueError, RuntimeError)):
                self._read(model, values)


if __name__ == "__main__":
    unittest.main()
