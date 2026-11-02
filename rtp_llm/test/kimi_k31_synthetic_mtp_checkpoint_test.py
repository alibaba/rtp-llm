"""Toy CPU fixtures check exact-copy semantics, bridge RNG and write guards."""
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

from rtp_llm.utils.kimi_k31_fake_checkpoint import fixture_config, _tensor_chunks
from rtp_llm.utils.kimi_k3_mtp_checkpoint import expected_tensors, IGNORED_SUFFIXES
from rtp_llm.utils.kimi_k31_synthetic_mtp_checkpoint import checkpoint_plan, write_checkpoint, sha256


def payloads(root):
    result = {}
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    for name in set(index.values()):
        with (root / name).open("rb") as reader:
            size = struct.unpack("<Q", reader.read(8))[0]
            header = json.loads(reader.read(size))
            for key, entry in header.items():
                if key == "__metadata__":
                    continue
                begin, end = entry["data_offsets"]
                reader.seek(8 + size + begin)
                result[key] = reader.read(end - begin)
    return result


def make_source(root):
    text = dict(num_hidden_layers=93, num_nextn_predict_layers=0, hidden_size=32,
                vocab_size=64, num_attention_heads=2, q_lora_rank=16, kv_lora_rank=32,
                qk_nope_head_dim=16, qk_rope_head_dim=8, v_head_dim=8,
                routed_expert_hidden_size=32, moe_intermediate_size=32,
                num_shared_experts=1, num_experts=2, first_k_dense_replace=1,
                linear_attn_config=dict(kda_layers=[1, 2, 3], full_attn_layers=[4, 92, 93]),
                dsa_config=dict(dsa_layers=[4, 92, 93], index_num_heads=32, index_head_dim=128),
                quantization_config=dict(format="mxfp4-pack-quantized",
                    config_groups={"experts": {"weights": {"group_size": 32, "num_bits": 4}}}))
    config = dict(model_type="kimi_k31", text_config=text)
    expected = expected_tensors(fixture_config(config, 93, 3101, mtp=True)["text_config"])
    global_names = {"embed_tokens.weight": "language_model.model.embed_tokens.weight",
                    "shared_head.head.weight": "language_model.lm_head.weight",
                    "shared_head.norm.weight": "language_model.model.norm.weight"}
    tensors = {}
    for key, (shape, dtype) in expected.items():
        suffix = key[len("language_model.model.layers.93."):]
        if suffix in ("enorm.weight", "hnorm.weight", "eh_proj.weight"):
            continue
        name = global_names.get(suffix, "language_model.model.layers.92." + suffix)
        tensors[name] = dict(shape=list(shape), dtype=dtype)
    for suffix in IGNORED_SUFFIXES:
        tensors["language_model.model.layers.92." + suffix] = dict(shape=[32], dtype="BF16")
    root.mkdir()
    header, data, offset = {}, [], 0
    for name, spec in sorted(tensors.items()):
        raw = b"".join(_tensor_chunks(name, spec, 3101))
        header[name] = dict(spec, data_offsets=[offset, offset + len(raw)])
        offset += len(raw)
        data.append(raw)
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    (root / "source.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(data))
    (root / "config.json").write_text(json.dumps(config))
    (root / "model.safetensors.index.json").write_text(json.dumps(
        dict(weight_map={name: "source.safetensors" for name in tensors})))
    return config


class SyntheticMtpCopyTest(unittest.TestCase):
    def test_exact_real_payload_copy_bridge_rule_and_both_layer_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "real"
            make_source(source)
            before = {p.name: sha256(p) for p in source.iterdir()}
            original = payloads(source)
            for layers in (4, 93):
                plan = checkpoint_plan(source, layers)
                self.assertEqual(sum(x.get("generated", False) for x in plan["items"]), 3)
                self.assertFalse(plan["config"]["rtp_llm_fixture"]["trained"])
                out = root / f"draft{layers}"
                receipt = write_checkpoint(source, out, layers, fixture_root=root,
                                           shard_bytes=65536, copy_bytes=37)
                actual = payloads(out)
                self.assertFalse((out / "INCOMPLETE").exists())
                self.assertTrue(receipt["complete"])
                self.assertEqual(receipt["source_decoder_layer"], 92)
                self.assertEqual(receipt["runtime_draft_layer"], 0)
                for item in plan["items"]:
                    expected = (b"".join(_tensor_chunks(item["name"], item, 3101))
                                if item.get("generated") else original[item["source_name"]])
                    self.assertEqual(actual[item["name"]], expected)
                for name, expected in receipt["shards"].items():
                    self.assertEqual(sha256(out / name), expected)
            self.assertEqual(before, {p.name: sha256(p) for p in source.iterdir()})

    def test_missing_indexer_fails_before_any_output_is_created(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "real"
            make_source(source)
            p = source / "model.safetensors.index.json"
            index = json.loads(p.read_text())
            del index["weight_map"]["language_model.model.layers.92.self_attn.indexer.wk.weight"]
            p.write_text(json.dumps(index))
            with self.assertRaisesRegex(ValueError, "missing real checkpoint parameter"):
                write_checkpoint(source, root / "draft", 4, fixture_root=root)
            self.assertFalse((root / "draft").exists())

    def test_truncated_real_payload_fails_before_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "real"
            make_source(source)
            p = source / "source.safetensors"
            p.write_bytes(p.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError, "truncated real parameter"):
                write_checkpoint(source, root / "draft", 4, fixture_root=root)
            self.assertFalse((root / "draft").exists())

    def test_existing_destination_and_root_escape_do_not_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "real"
            make_source(source)
            out = root / "draft"
            out.mkdir()
            (out / "existing").write_text("preserve")
            with self.assertRaises(FileExistsError):
                write_checkpoint(source, out, 4, fixture_root=root)
            self.assertEqual((out / "existing").read_text(), "preserve")
            with self.assertRaisesRegex(ValueError, "explicit permitted fixture root"):
                write_checkpoint(source, root.parent / "outside", 4, fixture_root=root)


if __name__ == "__main__":
    unittest.main()
