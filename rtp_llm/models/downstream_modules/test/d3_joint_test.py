"""Joint inline-readout sidecar and handler tests (decision_config version 2)."""

import json
import string
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from rtp_llm.models.downstream_modules.d3 import D3Handler, D3Module
from rtp_llm.models.downstream_modules.d3_config import (
    JOINT_MARKER,
    JOINT_TEMPLATE,
    D3Config,
    load_d3_config,
)

MARKER_ID = 777


def answer_codes():
    return (
        list(string.ascii_uppercase)
        + [a + b for a in string.ascii_uppercase for b in string.ascii_uppercase]
    )[:255]


def joint_decision_config(**overrides):
    data = {
        "format_version": 2,
        "base_model": "Qwen/Qwen3.8-27B",
        "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        "attention_mode": "noncausal_full_attention",
        "pooling": "last",
        "temperature": 1.3823494677922652,
        "max_pixels": 1638400,
        "codes": answer_codes(),
        "token_ids": list(range(1000, 1255)),
    }
    data.update(overrides)
    return data


def joint_protocol(**overrides):
    data = {
        "version": 3,
        "template": JOINT_TEMPLATE,
        "marker": JOINT_MARKER,
        "readout_offset": 0,
        "max_questions": None,
        "max_length": None,
        "usage": "nonpadding_joint_sequence_including_template_and_image_positions",
    }
    data.update(overrides)
    return data


def backbone_config(hidden_size=4, max_position_embeddings=262144):
    layer_types = ["linear_attention"] * 32
    for index in range(3, 32, 4):
        layer_types[index] = "full_attention"
    return {
        "architectures": ["Qwen3_5Model"],
        "model_type": "qwen3_5",
        "text_config": {
            "num_hidden_layers": 32,
            "hidden_size": hidden_size,
            "max_position_embeddings": max_position_embeddings,
            "layer_types": layer_types,
        },
    }


class JointTokenizer:
    """Resolves the marker and every answer code as single tokens.

    Answer codes map to the sidecar's token ids (1000 + index), matching the
    joint decision_config fixture.
    """

    def __init__(self, marker_id=MARKER_ID):
        self.marker_id = marker_id
        self.mapping = {}

    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        if text == JOINT_MARKER:
            return [self.marker_id]
        if text not in self.mapping:
            self.mapping[text] = 1000 + len(self.mapping)
        return [self.mapping[text]]

    def convert_ids_to_tokens(self, token_id):
        return JOINT_MARKER if token_id == self.marker_id else None


class D3JointConfigTest(unittest.TestCase):
    def checkpoint(self, sidecar=True, protocol=True, backbone=True, **overrides):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        if backbone:
            Path(directory.name, "config.json").write_text(
                json.dumps(backbone_config())
            )
        if sidecar:
            Path(directory.name, "decision_config.json").write_text(
                json.dumps(joint_decision_config(**overrides))
            )
        if protocol:
            Path(directory.name, "joint_config.json").write_text(json.dumps(joint_protocol()))
        return directory.name

    def test_loads_joint_sidecar(self):
        config = D3Config.load(self.checkpoint())
        self.assertIsNotNone(config.joint)
        self.assertEqual(config.format_version, 2)
        self.assertEqual(config.format_id, JOINT_TEMPLATE)
        self.assertEqual(config.attention_mode, "noncausal_full_attention")
        self.assertEqual(config.pooling, "last")
        self.assertAlmostEqual(config.temperature, 1.3823494677922652)
        self.assertEqual(config.max_pixels, 1638400)
        self.assertEqual(config.max_length, 262144)
        self.assertEqual(config.readout_dtype, "float32")
        self.assertEqual(config.hidden_size, 4)
        self.assertEqual(config.joint.marker, JOINT_MARKER)
        self.assertEqual(config.joint.readout_offset, 0)
        self.assertIsNone(config.joint.max_questions)

    def test_length_limit_follows_the_backbone(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        Path(directory.name, "config.json").write_text(
            json.dumps(backbone_config(max_position_embeddings=8192))
        )
        Path(directory.name, "decision_config.json").write_text(
            json.dumps(joint_decision_config())
        )
        Path(directory.name, "joint_config.json").write_text(json.dumps(joint_protocol()))
        self.assertEqual(D3Config.load(directory.name).max_length, 8192)

    def test_version_one_sidecar_unchanged(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        Path(directory.name, "config.json").write_text(json.dumps(backbone_config()))
        Path(directory.name, "decision_config.json").write_text(
            json.dumps(
                {
                    "format_version": 1,
                    "format_id": "d3-code-readout-v1",
                    "prompt": "d3",
                    "codes": answer_codes(),
                    "token_ids": list(range(1000, 1255)),
                    "temperature": 1.0,
                    "attention_mode": "noncausal_full_attention",
                    "pooling": "last",
                    "max_length": None,
                    "readout_dtype": "float32",
                }
            )
        )
        config = D3Config.load(directory.name)
        self.assertIsNone(config.joint)
        self.assertEqual(config.format_version, 1)

    def test_rejects_invalid_joint_sidecars(self):
        with self.assertRaises(ValueError):
            D3Config.load(self.checkpoint(protocol=False))
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        Path(directory.name, "config.json").write_text(json.dumps(backbone_config()))
        Path(directory.name, "decision_config.json").write_text(
            json.dumps(joint_decision_config())
        )
        for overrides in (
            {"version": 2},
            {"template": "joint-question-inline-readout-v2"},
            {"marker": "<|fim_prefix|>"},
            {"readout_offset": 1},
            {"max_questions": 4},
            {"max_length": 4096},
            {"usage": "tokens"},
        ):
            with self.subTest(overrides=overrides):
                Path(directory.name, "joint_config.json").write_text(
                    json.dumps(joint_protocol(**overrides))
                )
                with self.assertRaises(ValueError):
                    D3Config.load(directory.name)
        Path(directory.name, "joint_config.json").write_text(json.dumps(joint_protocol()))
        for overrides in (
            {"attention_mode": "causal"},
            {"pooling": "mean"},
            {"temperature": 0},
            {"max_length": 512},
            {"codes": answer_codes()[:254]},
            {"token_ids": [1000] * 255},
            {"format_version": 3},
        ):
            with self.subTest(overrides=overrides):
                Path(directory.name, "decision_config.json").write_text(
                    json.dumps(joint_decision_config(**overrides))
                )
                with self.assertRaises(ValueError):
                    D3Config.load(directory.name)


class D3JointHandlerTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        Path(self.directory.name, "config.json").write_text(
            json.dumps(backbone_config())
        )
        Path(self.directory.name, "decision_config.json").write_text(
            json.dumps(joint_decision_config())
        )
        Path(self.directory.name, "joint_config.json").write_text(
            json.dumps(joint_protocol())
        )
        self.config = D3Config.load(self.directory.name)
        self.weight = torch.arange(255 * 4, dtype=torch.float32).reshape(255, 4) / 100

    def handler(self, dtype=torch.float32, marker_id=MARKER_ID):
        handler = D3Handler(self.config, marker_id=marker_id)
        weight = self.weight.to(dtype)
        handler.init({"weight": weight})
        return handler

    def test_accepts_bf16_readout_and_upcasts(self):
        handler = self.handler(dtype=torch.bfloat16)
        self.assertEqual(handler._weight.dtype, torch.float32)
        # The upcast keeps the checkpoint's bf16 values exactly.
        torch.testing.assert_close(
            handler._weight,
            self.weight.to(torch.bfloat16).to(torch.float32),
            rtol=0,
            atol=0,
        )

    def test_v1_handler_rejects_marker_and_bf16(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        Path(directory.name, "config.json").write_text(json.dumps(backbone_config()))
        Path(directory.name, "decision_config.json").write_text(
            json.dumps(
                {
                    "format_version": 1,
                    "format_id": "d3-code-readout-v1",
                    "prompt": "d3",
                    "codes": answer_codes(),
                    "token_ids": list(range(1000, 1255)),
                    "temperature": 1.0,
                    "attention_mode": "causal",
                    "pooling": "last",
                    "max_length": None,
                    "readout_dtype": "float32",
                }
            )
        )
        v1_config = D3Config.load(directory.name)
        plain = D3Handler(v1_config)
        with self.assertRaises(ValueError):
            plain.init({"weight": self.weight.to(torch.bfloat16)})
        with self.assertRaises(ValueError):
            D3Handler(v1_config, marker_id=MARKER_ID)

    def test_select_token_position_returns_last_marker(self):
        handler = self.handler()
        ids = torch.tensor([5, MARKER_ID, 6, MARKER_ID, 7], dtype=torch.int32)
        mask = torch.ones(5, dtype=torch.bool)
        self.assertEqual(handler.select_token_position(ids, mask), 3)
        with self.assertRaises(ValueError):
            handler.select_token_position(torch.tensor([5, 6, 7], dtype=torch.int32))

    def test_joint_readout_packs_one_block_per_question(self):
        handler = self.handler()
        # Row 0: five tokens with markers at 1 and 3; row 1: three tokens with
        # one marker at 6. Positions are global over the expanded batch.
        input_ids = torch.tensor(
            [5, MARKER_ID, 6, MARKER_ID, 7, 8, MARKER_ID, 9], dtype=torch.int32
        )
        input_lengths = torch.tensor([5, 3], dtype=torch.int32)
        hidden = torch.arange(8 * 4, dtype=torch.float32).reshape(8, 4) / 10
        packed = handler.extend_forward(
            hidden_states=hidden,
            input_ids=input_ids,
            input_lengths=input_lengths,
        )
        self.assertEqual(tuple(packed.shape), (2, 510))
        expected = torch.nn.functional.linear(
            hidden[[1, 3, 6]].float(), self.weight
        )
        torch.testing.assert_close(packed[0, :255], expected[0], rtol=0, atol=0)
        torch.testing.assert_close(packed[0, 255:], expected[1], rtol=0, atol=0)
        torch.testing.assert_close(packed[1, :255], expected[2], rtol=0, atol=0)
        self.assertTrue(
            torch.equal(packed[1, 255:], torch.zeros(255))
        )

    def test_joint_readout_single_question_row(self):
        handler = self.handler()
        input_ids = torch.tensor([3, MARKER_ID], dtype=torch.int32)
        input_lengths = torch.tensor([2], dtype=torch.int32)
        hidden = torch.arange(2 * 4, dtype=torch.float32).reshape(2, 4)
        packed = handler.extend_forward(
            hidden_states=hidden,
            input_ids=input_ids,
            input_lengths=input_lengths,
        )
        self.assertEqual(tuple(packed.shape), (1, 255))
        torch.testing.assert_close(
            packed[0],
            torch.nn.functional.linear(hidden[1].float(), self.weight),
            rtol=0,
            atol=0,
        )

    def test_joint_readout_rejects_markerless_rows(self):
        handler = self.handler()
        input_ids = torch.tensor(
            [5, MARKER_ID, 6, 4, 5], dtype=torch.int32
        )
        input_lengths = torch.tensor([3, 2], dtype=torch.int32)
        hidden = torch.zeros(5, 4)
        with self.assertRaises(ValueError):
            handler.extend_forward(
                hidden_states=hidden,
                input_ids=input_ids,
                input_lengths=input_lengths,
            )


class D3JointModuleTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        Path(self.directory.name, "config.json").write_text(
            json.dumps(backbone_config())
        )
        Path(self.directory.name, "decision_config.json").write_text(
            json.dumps(joint_decision_config())
        )
        Path(self.directory.name, "joint_config.json").write_text(
            json.dumps(joint_protocol())
        )
        self.tokenizer = JointTokenizer()
        save_file(
            {"weight": torch.zeros(255, 4, dtype=torch.bfloat16)},
            str(Path(self.directory.name, "readout.safetensors")),
        )

    def config(self, **overrides):
        config = dict(
            ckpt_path=self.directory.name,
            hidden_size=4,
            compute_dtype=torch.bfloat16,
            quant_algo=SimpleNamespace(isQuant=lambda: False),
            quantization=None,
            quant_config=None,
            attn_config=SimpleNamespace(is_causal=False),
        )
        config.update(overrides)
        return SimpleNamespace(**config)

    def test_module_wires_the_marker(self):
        module = D3Module(self.config(), self.tokenizer)
        self.assertTrue(module.handler.joint)
        self.assertEqual(module.handler.marker_id, MARKER_ID)
        module.init(None)
        self.assertEqual(module.handler._weight.dtype, torch.float32)

    def test_module_requires_noncausal_wiring(self):
        with self.assertRaises(ValueError):
            D3Module(self.config(attn_config=SimpleNamespace(is_causal=True)), self.tokenizer)

    def test_module_rejects_multitoken_marker(self):
        class SplitMarker:
            def encode(self, text, *, add_special_tokens):
                return [1, 2]

            def convert_ids_to_tokens(self, token_id):
                return None

        with self.assertRaises(ValueError):
            D3Module(self.config(), SplitMarker())


if __name__ == "__main__":
    unittest.main()
