"""Opt-in CPU component check against an indexed release (not golden smoke)."""

import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open

from rtp_llm.models_py.model_loader import (
    NewLoaderConfig,
    NewLoaderLoadMethod,
    NewModelLoader,
)
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.new_models.qwen4_exp_mtp.input_owner import Qwen4MTPInputOwner
from rtp_llm.models_py.registry import register_model


@register_model("_q4_mtp_release_projection_only")
class _ProjectionOnly(RtpModule):
    def __init__(self, cfg, load):
        super().__init__()
        self.mtp = Qwen4MTPInputOwner(cfg.hidden, cfg.hc, norm_eps=cfg.eps)

    def checkpoint_weight_name_filter(self):
        names = set(dict(self.named_parameters()))
        return names.__contains__


@unittest.skipUnless(
    os.environ.get("QWEN4_MTP_INDEXED_SOURCE"), "requires indexed release checkpoint"
)
class ReleaseProjectionTest(unittest.TestCase):
    def test_real_source_to_existing_fusion(self):
        directory = Path(os.environ["QWEN4_MTP_INDEXED_SOURCE"])
        conf = json.loads((directory / "config.json").read_text())["text_config"]
        h, m, eps = conf["hidden_size"], conf["hc_count"], conf["rms_norm_eps"]
        self.assertEqual(conf["mtp"]["layer_types"], ["full_attention"])
        loader = NewModelLoader(
            SimpleNamespace(
                model_type="_q4_mtp_release_projection_only", hidden=h, hc=m, eps=eps
            ),
            NewLoaderConfig(
                device="cpu", compute_dtype=torch.bfloat16, load_method="scratch"
            ),
            model_path=str(directory),
        )
        result = loader.load()
        self.assertEqual(loader._selected_load_method, NewLoaderLoadMethod.SCRATCH)
        index = json.loads((directory / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        raw = {}
        for name, p in result.named_parameters():
            with safe_open(
                str(directory / index[name]), framework="pt", device="cpu"
            ) as file:
                expected = file.get_tensor(name)
            self.assertEqual(expected.dtype, torch.bfloat16)
            torch.testing.assert_close(p, expected, rtol=0, atol=0)
            raw[name.removeprefix("mtp.").removesuffix(".weight")] = expected
        self.assertEqual(
            set(raw),
            {
                "pre_fc_norm_embedding",
                "pre_fc_norm_hidden",
                "fc_embedding",
                "fc_hidden",
            },
        )
        rng = torch.Generator().manual_seed(381)
        x = torch.randn((2, h), generator=rng).bfloat16()
        y = torch.randn((2, h * m), generator=rng).bfloat16()

        def norm(x, gamma):
            v = x.float()
            return (
                (v * torch.rsqrt(v.square().mean(-1, keepdim=True) + eps))
                * (1 + gamma.float())
            ).bfloat16()

        pe = norm(x, raw["pre_fc_norm_embedding"]) @ raw["fc_embedding"].t()
        ph = norm(y, raw["pre_fc_norm_hidden"]).view(-1, h) @ raw["fc_hidden"].t()
        torch.testing.assert_close(
            result.mtp(x, y),
            (pe[:, None] + ph.view(2, m, h)).flatten(-2),
            rtol=0,
            atol=0,
        )
        print(
            f"release={directory} H={h} branches={m} source_bytes={sum(v.numel()*v.element_size() for v in raw.values())}",
            flush=True,
        )


if __name__ == "__main__":
    unittest.main()
