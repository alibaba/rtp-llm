"""Check GDN's serialized integer layout on both physical attention TP ranks."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from rtp_llm.models_py.new_models.qwen3_next.gdn_parameters import Qwen3NextGdnParameters
from rtp_llm.models_py.quant_methods.base import QuantizationConfig


WIDTHS = dict(q=64, k=64, v=128, z=128)


def owner(method, bits, tp, rank, ignored=()):
    cfg = SimpleNamespace(linear_key_head_dim=32, linear_value_head_dim=32,
                          linear_num_key_heads=2, linear_num_value_heads=4,
                          linear_conv_kernel_dim=3)
    quant = QuantizationConfig(method, SimpleNamespace(bits=bits, group_size=32),
                               ignored_layers=ignored)
    return Qwen3NextGdnParameters(cfg, 128, 1e-6, tp, rank, torch.float16, quant, "linear_attn")


def checkpoint(method, bits, split):
    gen = torch.Generator().manual_seed(91)
    cols = {}
    for part, n in {**WIDTHS, "out": 128}.items():
        pack = 32 // bits
        cols[part] = {
            "qweight": torch.randint(-2**31, 2**31-1, (128 if method == "awq" else 128 // pack,
                                                     n // pack if method == "awq" else n),
                                     dtype=torch.int32, generator=gen),
            "qzeros": torch.randint(-2**31, 2**31-1, (4, n // pack), dtype=torch.int32, generator=gen),
            "scales": torch.rand(4, n, generator=gen).half(),
        }
    records = {}
    for key in cols["q"]:
        vals = [cols[n][key] for n in WIDTHS]
        if split:
            records["in_proj_qkv." + key] = torch.cat(vals[:3], -1)
            records["in_proj_z." + key] = vals[-1]
        else:
            # The physical order is [q0,k0,v0,z0, q1,k1,v1,z1].
            records["in_proj_qkvz." + key] = torch.cat(
                [v.chunk(2, dim=-1)[head] for head in range(2) for v in vals], -1
            )
        records["out_proj." + key] = cols["out"][key]
    if method == "gptq":
        for key in (("in_proj_qkv", "in_proj_z") if split else ("in_proj_qkvz",)) + ("out_proj",):
            records[key + ".g_idx"] = torch.arange(128, dtype=torch.int32) // 32
            n = sum(WIDTHS[k] for k in key.removeprefix("in_proj_")) if key != "out_proj" else 128
            records[key + ".bias"] = torch.zeros(n, dtype=torch.float16)
    return cols, records


class GdnGroupwiseTest(unittest.TestCase):
    def test_fused_and_split_checkpoint_match_local_packed_bytes(self):
        for method in ("awq", "gptq"):
            for bits in (4, 8):
                for split in (False, True):
                    cols, records = checkpoint(method, bits, split)
                    with tempfile.TemporaryDirectory() as directory:
                        path = str(Path(directory) / "model.safetensors")
                        save_file(records, path)
                        for tp in (1, 2):
                            for rank in range(tp):
                                with self.subTest(method=method, bits=bits, split=split, tp=tp, rank=rank):
                                    m = owner(method, bits, tp, rank)
                                    with safe_open(path, framework="pt") as f:
                                        for key in reversed(f.keys()):
                                            m.load_weights({key: f.get_tensor(key)})
                                    for key in cols["q"]:
                                        expected = torch.cat([cols[n][key].chunk(tp, dim=1)[rank] for n in WIDTHS], 1)
                                        self.assertTrue(torch.equal(getattr(m.in_proj_qkvz, key), expected), key)
                                        self.assertTrue(torch.equal(getattr(m.out_proj, key),
                                                                    cols["out"][key].chunk(tp, dim=0)[rank]), key)
                                    m.in_proj_qkvz.validate_weights_loaded()
                                    m.out_proj.validate_weights_loaded()

    def test_invalid_metadata_and_missing_quantized_component(self):
        m = owner("gptq", 4, 2, 1)
        _, rows = checkpoint("gptq", 4, True)
        for key in ("out_proj.g_idx", "in_proj_qkv.g_idx"):
            with self.assertRaisesRegex(ValueError, "activation-order"):
                m.load_weights({key: rows[key].roll(1)})
        for name, value in rows.items():
            if name != "in_proj_z.qzeros":
                m.load_weights({name: value})
        with self.assertRaisesRegex(RuntimeError, "qzeros"):
            m.in_proj_qkvz.validate_weights_loaded()
        with self.assertRaisesRegex(RuntimeError, "Duplicate"):
            m.load_weights({"in_proj_qkv.scales": rows["in_proj_qkv.scales"]})

    def test_source_name_exclusions(self):
        m = owner("awq", 4, 2, 1, ("in_proj_z", "in_proj_qkv"))
        self.assertTrue(hasattr(m.in_proj_qkvz, "weight"))
        self.assertTrue(hasattr(m.out_proj, "qweight"))
        with self.assertRaisesRegex(ValueError, "partially match fused layer.*in_proj_qkvz"):
            owner("awq", 4, 2, 1, ("in_proj_z",))


if __name__ == "__main__":
    unittest.main()
