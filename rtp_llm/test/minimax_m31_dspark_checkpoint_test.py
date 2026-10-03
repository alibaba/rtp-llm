"""CPU-only checkpoint schema regressions; no torch or installed RTP required."""

import copy
import importlib.util
import json
import struct
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

_spec = importlib.util.spec_from_file_location(
    "dspark_checkpoint",
    Path(__file__).parents[1] / "models/minimax_m31_dspark_checkpoint.py",
)
checkpoint = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(checkpoint)


def fixture():
    text = dict(
        num_hidden_layers=5,
        hidden_size=6144,
        vocab_size=200064,
        num_attention_heads=64,
        num_key_value_heads=4,
        head_dim=128,
        dense_intermediate_size=12288,
        dspark_ffn_hidden_size=12288,
        dspark_markov_rank=256,
        dspark_block_size=7,
        dspark_noise_token_id=200058,
        sliding_window=4096,
        dspark_markov_head_type="vanilla",
        dspark_use_dense_ffn=True,
        dspark_hybrid_context_fusion=False,
        enable_confidence_head=True,
        confidence_head_with_markov=True,
        use_gemma_norm=True,
        dspark_target_layer_ids=[3, 17, 31, 45, 59],
        layer_types=["sliding_attention"] * 5,
        moe_layer_freq=[0] * 5,
        sparse_attention_config=None,
        dspark_config={"hybrid_context_fusion": False},
    )
    config = dict(
        architectures=["DSparkMiniMaxDraftModel"],
        text_config=text,
        quantization_config={"quant_method": "mxfp8", "weight_block_size": [1, 32]},
    )
    tensors = {}

    def add(name, dtype, shape):
        tensors[checkpoint.PREFIX + name] = dict(dtype=dtype, shape=shape)

    for name, shape in {
        "fc.weight": [6144, 30720],
        "hidden_norm.weight": [6144],
        "final_norm.weight": [6144],
        "markov_head.markov_w1.weight": [200064, 256],
        "markov_head.markov_w2.weight": [200064, 256],
        "confidence_head.proj.weight": [1, 6400],
        "confidence_head.proj.bias": [1],
    }.items():
        add(name, "BF16", shape)
    for i in range(5):
        root = f"layers.{i}.decoder_layer."
        for name, width in (
            ("input_layernorm", 6144),
            ("post_attention_layernorm", 6144),
            ("self_attn.q_norm", 128),
            ("self_attn.k_norm", 128),
        ):
            add(root + name + ".weight", "BF16", [width])
        for name, shape, scale in (
            ("self_attn.q_proj", [8192, 6144], [8192, 192]),
            ("self_attn.k_proj", [512, 6144], [512, 192]),
            ("self_attn.v_proj", [512, 6144], [512, 192]),
            ("self_attn.o_proj", [6144, 8192], [6144, 256]),
            ("mlp.gate_proj", [12288, 6144], [12288, 192]),
            ("mlp.up_proj", [12288, 6144], [12288, 192]),
            ("mlp.down_proj", [6144, 12288], [6144, 384]),
        ):
            add(root + name + ".weight", "F8_E4M3", shape)
            add(root + name + ".weight_scale_inv", "U8", scale)
    return config, tensors


class CheckpointSchemaTest(unittest.TestCase):
    def test_released_shape_and_absent_shared_weights(self):
        report = checkpoint.validate_schema(*fixture())
        self.assertEqual(report["tensor_count"], 97)
        self.assertTrue(report["shared_embedding_and_lm_head_absent"])
        self.assertFalse(report["runtime_ready"])
        self.assertTrue(report["unverified_math"])

    def test_wrong_shape_dtype_and_missing_norm(self):
        config, tensors = fixture()
        for suffix, field, bad in (
            ("fc.weight", "shape", [6144, 6144]),
            ("markov_head.markov_w2.weight", "shape", [256, 200064]),
            (
                "layers.4.decoder_layer.mlp.down_proj.weight_scale_inv",
                "shape",
                [6144, 768],
            ),
            (
                "layers.0.decoder_layer.self_attn.q_proj.weight_scale_inv",
                "dtype",
                "F32",
            ),
            ("confidence_head.proj.weight", "shape", [1, 6144]),
        ):
            with self.subTest(tensor=suffix):
                changed = copy.deepcopy(tensors)
                changed[checkpoint.PREFIX + suffix][field] = bad
                with self.assertRaises(ValueError):
                    checkpoint.validate_schema(config, changed)
        del tensors[checkpoint.PREFIX + "hidden_norm.weight"]
        with self.assertRaisesRegex(ValueError, "missing="):
            checkpoint.validate_schema(config, tensors)

    def test_mock_and_mixed_architecture_rejected(self):
        for field, value in (
            ("num_hidden_layers", 1),
            ("layer_types", ["full_attention"] * 5),
            ("moe_layer_freq", [1] * 5),
            ("dspark_hybrid_context_fusion", True),
        ):
            config, tensors = fixture()
            config["text_config"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                checkpoint.validate_schema(config, tensors)

    def test_extra_target_embedding_rejected(self):
        config, tensors = fixture()
        tensors["language_model.model.embed_tokens.weight"] = dict(
            dtype="BF16", shape=[200064, 6144]
        )
        with self.assertRaisesRegex(ValueError, "unexpected="):
            checkpoint.validate_schema(config, tensors)

    def test_header_extent_and_truncation_checks(self):
        header = {"x": dict(dtype="BF16", shape=[2], data_offsets=[0, 4])}
        encoded = json.dumps(header).encode()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "small.safetensors"
            path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"\0" * 4)
            self.assertEqual(checkpoint.read_safetensors_header(path), header)
            path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"\0" * 3)
            with self.assertRaisesRegex(ValueError, "payload"):
                checkpoint.read_safetensors_header(path)
            path.write_bytes(b"bad")
            with self.assertRaisesRegex(ValueError, "truncated"):
                checkpoint.read_safetensors_header(path)

    def test_target_root_reports_nested_dspark_checkpoint(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(
                json.dumps({"architectures": ["MiniMaxM3ForCausalLM"]})
            )
            (root / "dspark").mkdir()
            (root / "dspark/config.json").write_text(
                json.dumps({"architectures": ["DSparkMiniMaxDraftModel"]})
            )
            with self.assertRaisesRegex(ValueError, "use the nested DSpARK checkpoint"):
                checkpoint.inspect_checkpoint(str(root))


if __name__ == "__main__":
    unittest.main()
