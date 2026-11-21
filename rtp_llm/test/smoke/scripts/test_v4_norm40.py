"""CPU protocol/rounding tests. Optional native FlashInfer on an explicitly selected idle GPU."""

import os
import tempfile
import unittest
from types import SimpleNamespace

import torch
from compare_v4_ops import compare_norm

from rtp_llm.models_py.modules.dsv4.prefill.v4_norm40 import (
    NormProbe,
    rms_references,
    support40,
)


class Norm40Test(unittest.TestCase):
    def test_bf16_tie_and_ulp(self):
        real = torch.tensor(
            [[1 + 2**-8, -1.0 - 2**-8, +0.0, -0.0]], dtype=torch.float64
        )
        nearest = real.bfloat16()
        shifted = nearest.clone()
        shifted[0, 0] = torch.nextafter(
            shifted[0, 0], torch.tensor(10.0, dtype=torch.bfloat16)
        )
        shifted[0, 1] = torch.nextafter(
            shifted[0, 1], torch.tensor(-10.0, dtype=torch.bfloat16)
        )
        result = support40(shifted, real, 0)
        self.assertEqual(result["delta"], [[0, 1], [1, -1]])
        self.assertEqual(result["ties"], 2)

    def test_one_vs_two_roundings(self):
        g = torch.Generator().manual_seed(33)
        x = torch.randn(28, 4096, generator=g).bfloat16()
        w = torch.randn(4096, generator=g).bfloat16()
        keep = x.clone()
        variants, hi = rms_references(x, w, 1e-6)
        self.assertTrue(torch.equal(x, keep))
        self.assertTrue(torch.equal(variants["ref.f64_once"], hi.bfloat16()))
        self.assertFalse(
            torch.equal(variants["ref.bf16_before_weight"], variants["ref.f32_once"])
        )

    def test_snapshot_precedes_inplace_and_partial_evidence(self):
        trace = SimpleNamespace(pos=[39, 40], rows=torch.tensor([0, 1]))
        p = NormProbe(trace)
        x = torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16)
        p.save("before", x)
        x.zero_()
        p.record("live.norm_input", p.saved["before"])
        p.record("live.norm_output", x)

        def doc(role, rank):
            return dict(
                role=role,
                rank=rank,
                weights={"ffn_norm": {"sha": "a" * 64}},
                ops={"input": {"pos": trace.pos}},
                norm40={
                    "schema": 1,
                    "events": dict(p.docs),
                    "meta": {"rms_eps": 1e-6},
                    "path": {"rms": ["rmsnorm"]},
                },
            )

        docs = [doc(s, r) for s in ("oldfst", "newfst") for r in (0, 1)]
        docs[2].pop("norm40")
        docs[3]["norm40"]["events"]["live.norm_output"] = dict(
            p.docs["live.norm_output"], rows=["a" * 64, "b" * 64]
        )
        out = compare_norm(docs)
        self.assertEqual(out["per_role"]["newfst0"]["status"], "missing")
        self.assertTrue(out["per_rank"]["1"]["same_rms_operands"])
        self.assertEqual(
            out["per_rank"]["1"]["first_observed_live"], "live.norm_output"
        )
        self.assertIn("live.gate_input", out["per_rank"]["1"]["missing_either"])

    @unittest.skipUnless(
        os.getenv("V4_NORM_PROBE_CUDA") == "1", "explicit idle GPU only"
    )
    def test_native_flashinfer_original_shape_and_release_weight(self):
        import json
        from pathlib import Path

        import flashinfer.norm as fi
        from safetensors import safe_open

        root = Path("/mnt/nas1/hf/DeepSeek-V4-Flash-DSpark")
        key = "layers.1.ffn_norm.weight"
        name = json.loads((root / "model.safetensors.index.json").read_text())[
            "weight_map"
        ][key]
        with safe_open(str(root / name), framework="pt", device="cpu") as f:
            w = f.get_tensor(key).cuda()
        for n in (27, 28):
            x = torch.randn(n, 4096, device="cuda", dtype=torch.bfloat16)
            y = fi.rmsnorm(x, w, eps=1e-6)
            clone = x.clone()
            z = fi.rmsnorm(clone, w, eps=1e-6, out=clone)
            refs, high = rms_references(x, w, 1e-6)
            self.assertTrue(torch.equal(y, z))
            self.assertTrue(support40(y, high, 0)["valid"])


if __name__ == "__main__":
    unittest.main()
