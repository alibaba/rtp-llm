import json
import math
import struct
import tempfile
import unittest
from pathlib import Path

from rtp_llm.utils.kimi_k3_mtp_checkpoint import expected_tensors, validate_checkpoint


class KimiK3MtpCheckpointTest(unittest.TestCase):
    def make_checkpoint(self, root, corrupt=None, *, layers=93, shards=9):
        text = dict(
            hidden_size=32,
            vocab_size=64,
            num_attention_heads=2,
            q_lora_rank=32,
            kv_lora_rank=32,
            qk_nope_head_dim=16,
            qk_rope_head_dim=8,
            v_head_dim=16,
            routed_expert_hidden_size=32,
            moe_intermediate_size=32,
            num_shared_experts=2,
            num_experts=8,
            num_hidden_layers=layers,
            num_nextn_predict_layers=1,
            quantization_config={
                "format": "mxfp4-pack-quantized",
                "config_groups": {
                    "group_0": {"weights": {"group_size": 32, "num_bits": 4}}
                },
            },
        )
        (root / "config.json").write_text(json.dumps({"text_config": text}))
        weight_map, headers = {}, {str(i) + ".safetensors": {} for i in range(shards)}
        offsets = {key: 0 for key in headers}
        for name, (shape, dtype) in expected_tensors(text).items():
            shard_id = (
                int(name.split(".experts.")[1].split(".")[0]) + 1
                if ".experts." in name else 0
            ) % shards
            shard = str(shard_id) + ".safetensors"
            weight_map[name] = shard
            size = math.prod(shape) * {"BF16": 2, "U8": 1, "F32": 4}[dtype]
            headers[shard][name] = dict(
                shape=shape,
                dtype=dtype,
                data_offsets=[offsets[shard], offsets[shard] + size],
            )
            offsets[shard] += size
        if corrupt:
            corrupt(weight_map, headers)
        (root / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weight_map})
        )
        for shard, header in headers.items():
            encoded = json.dumps(header).encode()
            encoded += b" " * (-len(encoded) % 8)
            (root / shard).write_bytes(
                struct.pack("<Q", len(encoded)) + encoded + bytes(offsets[shard])
            )

    def test_valid_nine_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_checkpoint(root)
            result = validate_checkpoint(root)
            self.assertEqual(result["shards"], 9)
            self.assertEqual(result["required_tensors"], 24 + 8 * 6)

    def test_source_layer_and_shard_count_follow_checkpoint(self):
        for layers in (4, 47, 93):
            for shards in (1, 3, 9):
                with self.subTest(layers=layers, shards=shards), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    self.make_checkpoint(root, layers=layers, shards=shards)
                    result = validate_checkpoint(root)
                    self.assertEqual(result["shards"], shards)
                    self.assertEqual(result["required_tensors"], 72)

    def test_wrong_source_layer_is_not_silently_remapped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_checkpoint(root, layers=4)
            config = json.loads((root / "config.json").read_text())
            config["text_config"]["num_hidden_layers"] = 47
            with self.assertRaisesRegex(ValueError, "manifest mismatch"):
                validate_checkpoint(root, config)

    def test_bad_shape_rejected_before_loading(self):
        def corrupt(_, headers):
            key = next(
                k for k in headers["0.safetensors"] if k.endswith("eh_proj.weight")
            )
            headers["0.safetensors"][key]["shape"] = [32, 63]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_checkpoint(root, corrupt)
            with self.assertRaisesRegex(ValueError, "tensor mismatch.*eh_proj"):
                validate_checkpoint(root)

    def test_missing_global_expert_rejected(self):
        def corrupt(weight_map, headers):
            key = next(k for k in weight_map if ".experts.7.w2.weight_scale" in k)
            del weight_map[key]
            del headers["8.safetensors"][key]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_checkpoint(root, corrupt)
            with self.assertRaisesRegex(ValueError, "manifest mismatch.*experts.7.w2"):
                validate_checkpoint(root)

    def test_truncated_payload_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_checkpoint(root)
            shard = root / "8.safetensors"
            with shard.open("r+b") as writer:
                writer.truncate(shard.stat().st_size - 1)
            with self.assertRaisesRegex(
                ValueError, "truncated or invalid tensor payload"
            ):
                validate_checkpoint(root)


if __name__ == "__main__":
    unittest.main()
