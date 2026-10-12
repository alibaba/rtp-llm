"""Direct reference loader sharding and conversion, independent of NAS/GPU."""

import json
import tempfile
import unittest
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from rtp_llm.tools.dsv41_reference_probe import (
    checkpoint_headers,
    construct_meta_reference,
    copy_parameter,
    load_checkpoint,
    shard_spec,
    text_name,
)
from rtp_llm.tools.dsv41_trace import capture_model_boundaries, save_model_trace


class ReferenceLoaderTest(unittest.TestCase):
    def test_trace_snapshots_survive_aliases_and_hooks_are_removed(self):
        layer = SimpleNamespace(layer_id=2, ffn=torch.nn.ReLU(inplace=True))
        model = SimpleNamespace(layers=[layer], norm=torch.nn.Identity())
        value = torch.tensor([[-1.0, 2.0]])
        with capture_model_boundaries(model) as tensors:
            model.norm(layer.ffn(value))
        value.zero_()
        self.assertEqual(tensors["layer02.ffn.input"].tolist(), [[-1, 2]])
        self.assertEqual(tensors["layer02.ffn.output"].tolist(), [[0, 2]])
        self.assertEqual(tensors["final_norm"].tolist(), [[0, 2]])
        with self.assertRaisesRegex(RuntimeError, "injected"):
            with capture_model_boundaries(model):
                raise RuntimeError("injected")
        self.assertFalse(layer.ffn._forward_pre_hooks)
        self.assertFalse(layer.ffn._forward_hooks)
        self.assertFalse(model.norm._forward_hooks)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.pt"
            save_model_trace(path, tensors, {"start_pos": 0})
            result = torch.load(path, weights_only=True)
            self.assertEqual(result["metadata"], {"start_pos": 0})
            self.assertTrue(
                torch.equal(result["tensors"]["final_norm"], tensors["final_norm"])
            )
            with self.assertRaises(FileExistsError):
                save_model_trace(path, tensors, {})

    def test_meta_audit_does_not_reuse_or_pollute_device_cache(self):
        @lru_cache(2)
        def frequencies(size):
            return torch.arange(size)

        module = SimpleNamespace(
            precompute_freqs_cis=frequencies, dist=torch.distributed
        )

        class Reference(torch.nn.Module):
            def __init__(self, size, tokenizer):
                super().__init__()
                self.register_buffer("freqs_cis", module.precompute_freqs_cis(size))

        module.Transformer = Reference
        for populated in (False, True):
            frequencies.cache_clear()
            original = frequencies(4) if populated else None
            before = frequencies.cache_info()
            reference = construct_meta_reference(module, 4, None, 0, 8)
            self.assertEqual(reference.freqs_cis.device.type, "meta")
            self.assertEqual(frequencies.cache_info(), before)
            self.assertIs(module.precompute_freqs_cis, frequencies)
            actual = Reference(4, None).freqs_cis
            self.assertEqual(actual.device.type, "cpu")
            if populated:
                self.assertIs(actual, original)

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

    def test_cpu_slices_ignore_ambient_default_device(self):
        value = torch.arange(64, dtype=torch.bfloat16).reshape(8, 8)
        output = torch.empty((2, 8), dtype=torch.bfloat16)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.safetensors"
            save_file({"weight": value}, path)
            with safe_open(path, framework="pt", device="cpu") as handle:
                source = handle.get_slice("weight")
                with torch.device("meta"):
                    copy_parameter(
                        output,
                        source,
                        ("weight", (2, 8), 0, 2, 4),
                        chunk_bytes=16,
                    )
        torch.testing.assert_close(output, value[2:4], rtol=0, atol=0)

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
