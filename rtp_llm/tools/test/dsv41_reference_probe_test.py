"""Direct reference loader sharding and conversion, independent of NAS/GPU."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from rtp_llm.tools.dsv41_reference_probe import (
    checkpoint_headers,
    copy_parameter,
    load_checkpoint,
    shard_spec,
    text_name,
)


class ReferenceLoaderTest(unittest.TestCase):
    def copy(self, tensor, name, spec, dtype=None, scale=None):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.safetensors"
            values = {"source": tensor}
            if scale is not None:
                values["scale"] = scale
            save_file(values, path)
            out = torch.empty(spec[0], dtype=dtype or tensor.dtype)
            out.view(torch.uint8).fill_(255)
            with safe_open(path, framework="pt", device="cpu") as handle:
                copy_parameter(
                    out,
                    handle.get_slice("source"),
                    (name, *spec),
                    scale=handle.get_slice("scale") if scale is not None else None,
                    chunk_bytes=64,
                )
            return out

    def test_names_and_expert_ownership(self):
        self.assertEqual(
            text_name("model.layers.1.self_attn.wq_b.weight_scale_inv"),
            "layers.1.attn.wq_b.scale",
        )
        for name in (
            "model.vision.a",
            "model.mtp.a",
            "model.aligner.a",
            "model.image_start",
            "model.layers.1.mlp.gate.bias_vl",
        ):
            self.assertIsNone(text_name(name))
        for expert in (0, 47, 48, 335, 336, 383):
            owners = [
                rank
                for rank in range(8)
                if shard_spec(
                    f"layers.0.ffn.experts.{expert}.w1.weight", (8, 4), rank, 8, 384, {}
                )
                is not None
            ]
            self.assertEqual(owners, [expert // 48])

    def test_row_and_column_shards(self):
        tensor = torch.arange(64, dtype=torch.bfloat16).reshape(8, 8)
        mapping = {"wq_b": ("wq_b", 0), "wo_b": ("wo_b", 1)}
        for key, axis in (("wq_b", 0), ("wo_b", 1)):
            name = f"layers.0.attn.{key}.weight"
            for rank in (0, 7):
                spec = shard_spec(name, tensor.shape, rank, 8, 384, mapping)
                out = self.copy(tensor, name, spec)
                torch.testing.assert_close(out, tensor.narrow(axis, rank, 1))

    def test_engram_partial_and_empty_last_shards(self):
        for rows in (3, 17, 23):
            for rank in (0, 7):
                for kind in ("weight", "weight_scale_inv", "scale"):
                    name = f"model.layers.1.engram.embed.{kind}"
                    dtype = (
                        torch.float8_e8m0fnu
                        if kind != "weight"
                        else torch.float8_e4m3fn
                    )
                    tensor = torch.full((rows, 4), 2.0).to(dtype)
                    spec = shard_spec(text_name(name), tensor.shape, rank, 8, 384, {})
                    out = self.copy(tensor, name, spec)
                    expected = torch.full(spec[0], 1.0 if kind != "weight" else 0.0)
                    expected[: spec[3] - spec[2]] = 2
                    torch.testing.assert_close(out.float(), expected)

    def test_fp4_bytes_and_fp32_cast(self):
        tensor = torch.arange(64, dtype=torch.int8).reshape(8, 8)
        spec = (tuple(tensor.shape), None, 0, 0)
        out = self.copy(tensor, "expert.weight", spec, torch.float4_e2m1fn_x2)
        self.assertTrue(torch.equal(out.view(torch.int8), tensor))
        value = tensor.bfloat16()
        out = self.copy(value, "head.weight", spec, torch.float32)
        torch.testing.assert_close(out, value.float())

    def test_wo_a_dequantization_after_sharding(self):
        tensor = (
            torch.arange(256 * 64)
            .reshape(256, 64)
            .remainder(8)
            .float()
            .to(torch.float8_e4m3fn)
        )
        scale = (
            torch.arange(16)
            .reshape(8, 2)
            .remainder(3)
            .float()
            .exp2()
            .to(torch.float8_e8m0fnu)
        )
        name = "layers.0.attn.wo_a.weight"
        expected = tensor.float() * scale.float().repeat_interleave(
            32, 0
        ).repeat_interleave(32, 1)
        for rank in (0, 7):
            spec = shard_spec(name, tensor.shape, rank, 8, 384, {"wo_a": ("wo_a", 0)})
            out = self.copy(tensor, name, spec, torch.bfloat16, scale)
            torch.testing.assert_close(
                out, expected[rank * 32 : (rank + 1) * 32].bfloat16()
            )

    def test_file_load_and_identity_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            name = "layers.0.attn.wo_a.weight"
            scale_name = "layers.0.attn.wo_a.scale"
            values = {
                name: torch.full((256, 64), 2.0).to(torch.float8_e4m3fn),
                scale_name: torch.full((8, 2), 4.0).to(torch.float8_e8m0fnu),
            }
            path = root / "one.safetensors"
            save_file(values, path)
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {key: path.name for key in values}})
            )
            index, _, identities = checkpoint_headers(root)
            spec = shard_spec(name, (256, 64), 7, 8, 384, {"wo_a": ("wo_a", 0)})
            out = torch.zeros(spec[0], dtype=torch.bfloat16)
            reference = SimpleNamespace(named_parameters=lambda: [(name, out)])
            admitted = ({name: (name, *spec)}, index, identities, {name: scale_name})
            load_checkpoint(reference, root, admitted)
            torch.testing.assert_close(out, torch.full_like(out, 8))
            save_file(values, path)
            out.zero_()
            with self.assertRaisesRegex(RuntimeError, "changed before loading"):
                load_checkpoint(reference, root, admitted)
            self.assertEqual(out.count_nonzero(), 0)

    def test_index_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_file({"present": torch.ones(1)}, root / "one.safetensors")
            index = root / "model.safetensors.index.json"
            index.write_text(json.dumps({"weight_map": {"missing": "one.safetensors"}}))
            with self.assertRaisesRegex(ValueError, "index/header mismatch"):
                checkpoint_headers(root)


if __name__ == "__main__":
    unittest.main()
