"""Exercise the native mHC runtime parameter tree via complete checkpoint loads."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from rtp_llm.models_py.layers.norm import RMSNorm
from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.new_models.deepseek_v4.hc import OwnedHCHead, OwnedHCUnit
from rtp_llm.models_py.registry import register_model
from rtp_llm.models_py.weight_mapper import WeightsMapper


@register_model("_v4_flash_hc_test")
class _HCModel(RtpModule):
    fastsafetensors_load_isolated = True
    WEIGHTS_MAPPER = WeightsMapper(
        exact_mapping={
            "layers.0.attn_norm.weight": "norm.weight",
            **{f"layers.0.hc_attn_{n}": f"unit.{n}" for n in ("base", "scale", "fn")},
            **{f"hc_head_{n}": f"head.{n}" for n in ("base", "scale", "fn")},
        }
    )

    def __init__(self, config, load_config):
        super().__init__()
        kw = dict(dim=config.dim, hc_mult=4, hc_eps=1e-6, norm_eps=1e-6)
        self.unit = OwnedHCUnit(hc_sinkhorn_iters=20, **kw)
        self.head = OwnedHCHead(**kw)
        self.norm = RMSNorm(config.dim, eps=0.0025, params_dtype=torch.bfloat16)

    def checkpoint_weight_name_filter(self):
        return self.WEIGHTS_MAPPER.exact_mapping.__contains__

    def load_weights(self, items):
        super().load_weights(self.WEIGHTS_MAPPER.apply(items))


class FlashHCTest(unittest.TestCase):
    def load(self, folder, dim, method):
        loader = NewModelLoader(
            SimpleNamespace(model_type="_v4_flash_hc_test", dim=dim),
            NewLoaderConfig(
                device="cpu", compute_dtype=torch.bfloat16, load_method=method
            ),
            model_path=str(folder),
        )
        model = loader.load()
        self.assertEqual(loader._selected_load_method.value, method)
        return model

    def weights(self):
        return {"layers.0.attn_norm.weight": torch.linspace(0.5, 1.5, 8).bfloat16()} | {
            name: torch.arange(n, dtype=torch.float32).reshape(shape) + 0.00123
            for name, n, shape in (
                ("layers.0.hc_attn_fn", 768, (24, 32)),
                ("layers.0.hc_attn_scale", 3, (3,)),
                ("layers.0.hc_attn_base", 24, (24,)),
                ("hc_head_fn", 128, (4, 32)),
                ("hc_head_scale", 1, (1,)),
                ("hc_head_base", 4, (4,)),
            )
        }

    def check_params(self, model, weights):
        params = dict(model.named_parameters())
        self.assertEqual(len(params), len(weights))
        for key, ref in weights.items():
            value = params[model.WEIGHTS_MAPPER.map_name(key)]
            self.assertEqual(value.dtype, ref.dtype)
            self.assertFalse(value.requires_grad)
            torch.testing.assert_close(value, ref, rtol=0, atol=0)
        # Fast FP8 prefill fuses this loaded norm and accesses its epsilon by
        # the runtime name. The eager and fused consumers must agree at a
        # non-default epsilon as well.
        norm = model.norm
        self.assertEqual(norm.variance_epsilon, 0.0025)
        x = torch.linspace(-0.04, 0.04, norm.hidden_size).bfloat16()
        fp = x.float()
        ref = (
            fp
            * (fp.square().mean() + norm.variance_epsilon).rsqrt()
            * weights["layers.0.attn_norm.weight"]
        ).to(x.dtype)
        torch.testing.assert_close(norm(x), ref, rtol=0, atol=0)

    def test_scratch_ownership(self):
        weights = self.weights()
        with tempfile.TemporaryDirectory() as folder:
            save_file(weights, str(Path(folder) / "model.safetensors"))
            model = self.load(folder, 8, "scratch")
        self.check_params(model, weights)

    @unittest.skipUnless(
        os.environ.get("NEWLOADER_TEST_NATIVE_FST"), "requires native FST"
    )
    def test_fst_ownership_after_close(self):
        weights = self.weights()
        with tempfile.TemporaryDirectory() as folder:
            save_file(weights, str(Path(folder) / "model.safetensors"))
            model = self.load(folder, 8, "fastsafetensors")
        self.check_params(model, weights)

    def test_missing_fn_not_published(self):
        weights = self.weights()
        del weights["layers.0.hc_attn_fn"]
        with tempfile.TemporaryDirectory() as folder:
            save_file(weights, str(Path(folder) / "model.safetensors"))
            with self.assertRaisesRegex(RuntimeError, "fn"):
                self.load(folder, 8, "scratch")

    @unittest.skipUnless(
        os.environ.get("DSV4_FLASH_RELEASE"), "requires Flash checkpoint"
    )
    def test_real_release_hc(self):
        folder = Path(os.environ["DSV4_FLASH_RELEASE"])
        index = json.loads((folder / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        dim = json.loads((folder / "config.json").read_text())["hidden_size"]
        weights = {}
        for name in self.weights():
            with safe_open(str(folder / index[name]), framework="pt") as source:
                weights[name] = source.get_tensor(name)
        # Shards also carry FP8 expert tensors. Restrict the reader to these
        # mHC and attention norm records; do not read a model-sized payload.
        model = self.load(folder, dim, "scratch")
        self.check_params(model, weights)


if __name__ == "__main__":
    unittest.main()
