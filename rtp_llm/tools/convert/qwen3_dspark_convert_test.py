"""DCP-to-safetensors tests for the Qwen3 DSpark converter."""

import json
import tempfile
import unittest
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from safetensors import safe_open

from rtp_llm.tools.convert.qwen3_dspark_convert import convert, expected_tensor_shapes


def _config():
    return {
        "architectures": ["DSparkDraftModel"],
        "model_type": "dspark",
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "vocab_size": 17,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000000,
        "target_layer_ids": [3, 7],
        "target_hidden_size": 8,
        "target_num_hidden_layers": 8,
        "num_target_layers": 8,
        "block_size": 5,
        "mask_token_id": 16,
        "markov_rank": 3,
        "enable_confidence_head": True,
        "confidence_head_with_markov": True,
        "layer_types": ["full_attention"] * 2,
        "use_sliding_window": False,
        "sliding_window": None,
        "hidden_act": "silu",
        "attention_bias": False,
        "markov_head_type": "vanilla",
        "tie_word_embeddings": False,
    }


def _write_dcp(
    path,
    config,
    *,
    omit=(),
    dtype=torch.bfloat16,
    shape_overrides=None,
    add_unknown=False
):
    state = {"model_state": {"model": {"draft_model": {}}}}
    shape_overrides = shape_overrides or {}
    source = {}
    for index, (key, shape) in enumerate(expected_tensor_shapes(config).items()):
        if key in omit:
            continue
        shape = shape_overrides.get(key, shape)
        value = torch.arange(
            int(torch.tensor(shape).prod().item()), dtype=torch.float32
        ).reshape(shape)
        source[key] = (value + index).to(dtype)
    if add_unknown:
        source["unexpected.weight"] = torch.ones((1,), dtype=dtype)
    state["model_state"]["model"]["draft_model"].update(source)
    dcp.save(state_dict=state, storage_writer=dcp.FileSystemWriter(str(path)))
    return source


class Qwen3DSparkConvertTest(unittest.TestCase):
    def test_real_dcp_roundtrip_and_export_contract(self):
        config = _config()
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            dcp_dir, output = root / "dcp", root / "output"
            source = _write_dcp(dcp_dir, config)
            config_path = root / "draft_config.json"
            config_path.write_text(json.dumps(config))
            convert(str(dcp_dir), str(config_path), str(output))
            exported = json.loads((output / "config.json").read_text())
            self.assertEqual(exported["architectures"], ["Qwen3DSparkForCausalLM"])
            self.assertEqual(exported["model_type"], "qwen_3_dspark")
            self.assertEqual(exported["aux_hidden_state_layer_ids"], [3, 7])
            self.assertTrue(exported["sample_from_anchor"])
            self.assertEqual(exported["lm_head_source"], "target")
            self.assertEqual(exported["dtype"], "bfloat16")
            self.assertEqual(exported["block_size"], 5)
            with safe_open(
                str(output / "model.safetensors"), framework="pt", device="cpu"
            ) as saved:
                self.assertNotIn("lm_head.weight", saved.keys())
                for key, tensor in source.items():
                    mapped = {
                        "context_proj.weight": "fc.weight",
                        "context_norm.weight": "hidden_norm.weight",
                        "final_norm.weight": "model.norm.weight",
                    }.get(
                        key,
                        (
                            "model." + key
                            if key == "embed_tokens.weight" or key.startswith("layers.")
                            else key
                        ),
                    )
                    self.assertTrue(torch.equal(saved.get_tensor(mapped), tensor))
            report = json.loads((output / "conversion-report.json").read_text())
            self.assertTrue(report["verification"]["bitwise"])
            self.assertEqual(
                set(report["unused_by_current_runtime"]["confidence_keys"]),
                {"confidence_head.proj.weight", "confidence_head.proj.bias"},
            )

    def test_rejects_missing_unknown_shape_dtype_and_existing_output(self):
        cases = [
            ("missing", {"omit": ("context_proj.weight",)}, "keys mismatch"),
            (
                "missing_markov",
                {"omit": ("markov_head.markov_w2.weight",)},
                "keys mismatch",
            ),
            ("unknown", {"add_unknown": True}, "keys mismatch"),
            (
                "shape",
                {"shape_overrides": {"context_norm.weight": (9,)}},
                "shape mismatch",
            ),
            ("dtype", {"dtype": torch.float32}, "must be bfloat16"),
        ]
        for name, kwargs, message in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as root:
                root = Path(root)
                config = _config()
                config_path = root / "config.json"
                config_path.write_text(json.dumps(config))
                _write_dcp(root / "dcp", config, **kwargs)
                with self.assertRaisesRegex(ValueError, message):
                    convert(str(root / "dcp"), str(config_path), str(root / "output"))
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            config = _config()
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config))
            _write_dcp(root / "dcp", config)
            (root / "output").mkdir()
            with self.assertRaises(FileExistsError):
                convert(str(root / "dcp"), str(config_path), str(root / "output"))

    def test_preserves_training_block_and_rejects_invalid_semantics(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            config = _config()
            config["block_size"] = 7
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config))
            _write_dcp(root / "dcp", config)
            convert(str(root / "dcp"), str(config_path), str(root / "output"))
            self.assertEqual(
                json.loads((root / "output/config.json").read_text())["block_size"], 7
            )
        for field, value in (
            ("target_layer_ids", []),
            ("target_layer_ids", [7, 3]),
            ("target_layer_ids", [3, 3]),
            ("target_layer_ids", [3, 8]),
            ("mask_token_id", 17),
            ("mask_token_id", -1),
            ("markov_rank", 0),
            ("hidden_size", 0),
            ("target_hidden_size", 9),
            ("layer_types", ["sliding_attention", "full_attention"]),
            ("attention_bias", True),
        ):
            with self.subTest(field=field):
                config = _config()
                config[field] = value
                with self.assertRaises(ValueError):
                    expected_tensor_shapes(config)

    def test_rejects_non_anchor_sampling_before_writing_output(self):
        # The conversion contract requires anchor sampling: a draft trained
        # without it must be rejected instead of silently flipped, and before
        # any output artifact is written.
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            config = _config()
            config["sample_from_anchor"] = False
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config))
            _write_dcp(root / "dcp", config)
            with self.assertRaisesRegex(ValueError, "anchor sampling"):
                convert(str(root / "dcp"), str(config_path), str(root / "output"))
            self.assertFalse((root / "output").exists())


if __name__ == "__main__":
    unittest.main()
