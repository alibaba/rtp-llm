import copy
import json
import string
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from safetensors.torch import save_file

from rtp_llm.models.downstream_modules.jev import (
    JevConfig,
    JevHandler,
    JevModule,
    load_jev_config,
)


def manifest(choice_count=16):
    choice = (
        list(string.ascii_uppercase)
        + [a + b for a in string.ascii_uppercase for b in string.ascii_uppercase]
    )[:choice_count]
    labels = ["false", "true"] + [str(i) for i in range(6)] + choice
    ids = list(range(100, 100 + len(labels)))
    kinds = {}
    for kind, start, count in (
        ("noul", 0, 2),
        ("score", 2, 6),
        ("choice", 8, choice_count),
    ):
        kinds[kind] = {
            "row_indices": list(range(start, start + count)),
            "label_names": labels[start : start + count],
            "token_ids": ids[start : start + count],
            "bias": [float(i + 1) / 10 for i in range(min(count, 16))]
            + [0.0] * max(0, count - 16),
            "temperature": 2.0,
        }
    kinds["choice"].update(native_slots=16, max_options=choice_count)
    return {
        "schema_version": 1,
        "model_type": "jev",
        "mode": "system1",
        "hidden_size": 4,
        "num_labels": len(labels),
        "labels": labels,
        "token_ids": ids,
        "kinds": kinds,
        "head_file": "jev_head.safetensors",
        "head_weight_key": "projection.weight",
        "protocol": "jev27-bare-v1",
        "template_version": "bare-v1",
        "prefix": "",
        "default_strategy": "single",
        "max_context_tokens": 1024,
        "backbone_architecture": "Qwen3_5ForConditionalGeneration",
        "source_revision": "test-provenance",
    }


class FakeTokenizer:
    def __init__(self, data):
        self.mapping = dict(zip(data["labels"], data["token_ids"]))

    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return [self.mapping[text]]


class JevConfigTest(unittest.TestCase):
    def test_manifest_preserves_provenance_and_all_extended_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(load_jev_config(directory))
            data = manifest(256)
            Path(directory, "jev_config.json").write_text(json.dumps(data))
            self.assertEqual(load_jev_config(directory), data)
            config = JevConfig.load(directory)
        self.assertEqual(config.num_labels, 264)
        self.assertEqual(config.kinds["choice"].max_options, 256)
        self.assertEqual(config.kinds["choice"].native_slots, 16)
        self.assertEqual(config.kinds["score"].label_names, tuple("012345"))
        self.assertEqual(config.kinds["choice"].bias[16:], (0.0,) * 240)

    def test_rejects_invalid_top_level_metadata(self):
        cases = [
            ("schema_version", True),
            ("schema_version", 2),
            ("mode", "system2"),
            ("model_type", "kev"),
            ("head_file", "../head.safetensors"),
            ("head_weight_key", "weight"),
            ("head_dtype", "bfloat16"),
            ("head_includes_bias", True),
            ("head_includes_bias", 0),
            ("thinking", "on"),
            ("template_version", "chat"),
            ("protocol", "other"),
            ("hidden_size", 0),
            ("hidden_size", True),
            ("num_labels", 25),
            ("num_labels", 265),
            ("token_ids", [1] * 24),
            ("labels", ["x"] * 24),
            ("max_context_tokens", 0),
            ("backbone_architecture", ""),
            ("prefix", []),
            ("default_strategy", "auto"),
        ]
        for key, value in cases:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                JevConfig.from_dict(dict(manifest(), **{key: value}))
        with self.assertRaises(ValueError):
            JevConfig.from_dict([])

    def test_rejects_invalid_kind_mapping_and_calibration(self):
        cases = [
            ("choice", "row_indices", list(range(7, 23))),
            ("choice", "row_indices", list(range(8, 23)) + [999]),
            ("choice", "label_names", list("ABCDEFGHIJKLMNOP")[::-1]),
            ("choice", "token_ids", list(range(108, 123)) + [999]),
            ("choice", "native_slots", 32),
            ("choice", "max_options", 256),
            ("noul", "label_names", ["true", "false"]),
            ("score", "label_names", list("123456")),
            ("score", "bias", [0.0]),
            ("noul", "bias", [0, float("nan")]),
            ("noul", "bias", [True, 0]),
            ("noul", "temperature", 0),
            ("score", "temperature", float("inf")),
            ("choice", "temperature", True),
            ("score", "native_slots", 5),
        ]
        for kind, key, value in cases:
            data = manifest()
            data["kinds"][kind][key] = value
            with self.subTest(kind=kind, key=key), self.assertRaises(ValueError):
                JevConfig.from_dict(data)
        data = manifest(17)
        data["kinds"]["choice"]["bias"][-1] = 1.0
        with self.assertRaisesRegex(ValueError, "untrained"):
            JevConfig.from_dict(data)
        for kinds in ({}, {**manifest()["kinds"], "extra": {}}, []):
            with self.assertRaisesRegex(ValueError, "exactly"):
                JevConfig.from_dict(dict(manifest(), kinds=kinds))


class JevHeadTest(unittest.TestCase):
    def setUp(self):
        self.data = manifest(256)
        self.config = JevConfig.from_dict(self.data)
        self.handler = JevHandler(self.config)
        generator = torch.Generator().manual_seed(40)
        self.weight = torch.randn((264, 4), generator=generator) * 0.25
        self.handler.init({"projection.weight": self.weight})
        self.hidden = torch.randn((15, 4), generator=generator).bfloat16()
        self.ids = torch.arange(15)
        self.lengths = torch.tensor([4, 8, 3], dtype=torch.int32)

    def forward(self, **overrides):
        args = dict(
            hidden_states=self.hidden, input_ids=self.ids, input_lengths=self.lengths
        )
        args.update(overrides)
        return self.handler.extend_forward(**args)

    def test_all_raw_logits_match_independent_fp64_projection(self):
        actual = self.forward()
        expected = self.hidden[[3, 11, 14]].double() @ self.weight.double().T
        torch.testing.assert_close(actual, expected.float(), rtol=1e-5, atol=1e-7)
        self.assertEqual(actual.shape, (3, 264))
        self.assertEqual(actual.dtype, torch.float32)
        # No bias, temperature or softmax belongs in the engine output.
        self.assertTrue((actual < 0).any())
        self.assertFalse(torch.allclose(actual.sum(-1), torch.ones(3)))
        data = copy.deepcopy(self.data)
        for kind in data["kinds"].values():
            kind["temperature"] = 0.123
            kind["bias"][:2] = [100.0, -100.0]
        other = JevHandler(JevConfig.from_dict(data))
        other.init({"projection.weight": self.weight})
        torch.testing.assert_close(
            actual,
            other.extend_forward(
                hidden_states=self.hidden,
                input_ids=self.ids,
                input_lengths=self.lengths,
            ),
            rtol=0,
            atol=0,
        )

    def test_prefix_cached_suffix_and_partition_preserve_readout(self):
        expected = self.forward()
        for prefixes in ([0, 0, 0], [1, 6, 2], [3, 7, 2]):
            suffixes = []
            offset = 0
            lengths = []
            for length, prefix in zip(self.lengths.tolist(), prefixes):
                suffixes.append(self.hidden[offset + prefix : offset + length])
                lengths.append(length - prefix)
                offset += length
            hidden = torch.cat(suffixes)
            actual = self.forward(
                hidden_states=hidden,
                input_ids=torch.arange(len(hidden)),
                input_lengths=torch.tensor(lengths),
                prefix_lengths=torch.tensor(prefixes),
            )
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            split = [
                self.forward(
                    hidden_states=suffix,
                    input_ids=torch.arange(len(suffix)),
                    input_lengths=torch.tensor([len(suffix)]),
                    prefix_lengths=torch.tensor([prefix]),
                )
                for suffix, prefix in zip(suffixes, prefixes)
            ]
            torch.testing.assert_close(torch.cat(split), expected, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(
            self.forward(prefix_lengths=torch.tensor([], dtype=torch.int32)),
            expected,
            rtol=0,
            atol=0,
        )

    def test_autocast_does_not_round_fp32_head_or_apply_final_norm_again(self):
        expected = self.forward()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = self.forward()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            self.forward(hidden_states=self.hidden.float()), expected, rtol=0, atol=0
        )
        torch.testing.assert_close(
            self.forward(hidden_states=self.hidden * 2), expected * 2, rtol=0, atol=0
        )
        self.assertEqual(self.handler._weight.dtype, torch.float32)

    def test_multimodal_admission_uses_expanded_last_text_position(self):
        # Two image spans have expanded into three and four feature tokens.
        ids = torch.tensor([10, 20, 0, 0, 0, 21, 30, 20, 0, 0, 0, 0, 21, 40])
        mask = torch.tensor(
            [1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1], dtype=torch.int32
        )
        self.assertEqual(self.handler.select_token_position(ids, mask), 13)
        self.assertEqual(self.handler.cache_prefix_length(ids), 13)
        self.assertEqual(self.handler.cache_prefix_length(torch.tensor([10])), 0)
        self.assertTrue(self.handler.supports_prefix_cache)
        self.assertTrue(self.handler.prefill_only)
        # Readout uses expanded row lengths, including image features, rather
        # than selecting the last text position from a compressed prompt.
        hidden = torch.zeros((len(ids), 4))
        hidden[7] = 10
        hidden[-1] = torch.tensor([1.0, 2.0, 3.0, 4.0])
        result = self.forward(
            hidden_states=hidden, input_ids=ids, input_lengths=torch.tensor([len(ids)])
        )
        torch.testing.assert_close(result[0], self.weight @ hidden[-1])

    def test_admission_rejects_missing_readout_or_bad_expansion_metadata(self):
        tokens = torch.tensor([1, 2, 3])
        for ids in (
            torch.tensor([], dtype=torch.int64),
            tokens.float(),
            tokens[None],
            torch.empty(3, dtype=torch.int64, device="meta"),
        ):
            with self.subTest(ids=ids), self.assertRaisesRegex(
                ValueError, "CPU token row"
            ):
                self.handler.select_token_position(ids)
        for mask in (
            torch.tensor([1, 0, 0]),
            torch.ones(2),
            torch.ones((1, 3)),
            torch.ones(3),
            torch.empty(3, dtype=torch.int64, device="meta"),
        ):
            with self.subTest(mask=mask), self.assertRaisesRegex(
                ValueError, "text_tokens_mask"
            ):
                self.handler.select_token_position(tokens, mask)
        with self.assertRaisesRegex(ValueError, "max_context_tokens"):
            self.handler.select_token_position(torch.arange(1025))

    def test_forward_rejects_incomplete_and_invalid_rows(self):
        for overrides in (
            {"hidden_states": self.hidden[:, :3]},
            {"hidden_states": self.hidden.long()},
            {"input_ids": self.ids.float()},
            {"input_ids": self.ids[:-1]},
            {"input_lengths": torch.tensor([4.0, 8.0, 3.0])},
            {"input_lengths": torch.tensor([4, 8, 4])},
            {"input_lengths": torch.tensor([4, 11, 0])},
            {"input_lengths": torch.tensor([4, 12, -1])},
            {"input_lengths": torch.tensor([], dtype=torch.int32)},
            {"input_lengths": self.lengths[None]},
            {"prefix_lengths": [-1, 0, 0]},
            {"prefix_lengths": torch.tensor([-1, 0, 0])},
            {"prefix_lengths": torch.tensor([0, 0])},
            {"prefix_lengths": torch.tensor([[0, 0, 0]])},
            {"prefix_lengths": torch.tensor([0.0, 0.0, 0.0])},
            {"prefix_lengths": torch.tensor([1024, 0, 0])},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.forward(**overrides)

    def test_projection_load_rejects_bad_shape_dtype_nonfinite_or_extra_weights(self):
        for tensors in (
            {},
            {"weight": self.weight},
            {"projection.weight": self.weight[:, :3]},
            {"projection.weight": self.weight.half()},
            {"projection.weight": torch.full_like(self.weight, float("nan"))},
            {"projection.weight": self.weight, "bias": torch.zeros(264)},
        ):
            with self.subTest(keys=tensors.keys()), self.assertRaises(ValueError):
                self.handler.init(tensors)
        self.handler = JevHandler(self.config)
        with self.assertRaisesRegex(RuntimeError, "initialized"):
            self.forward()

    def test_forward_never_materializes_token_ids_or_lengths_as_python_lists(self):
        with mock.patch.object(
            torch.Tensor, "tolist", side_effect=AssertionError("unexpected host copy")
        ):
            result = self.forward()
        self.assertEqual(result.shape, (3, 264))


class JevModuleTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.data = manifest()
        Path(self.directory.name, "jev_config.json").write_text(json.dumps(self.data))
        self.tokenizer = FakeTokenizer(self.data)
        self.weight = torch.arange(96, dtype=torch.float32).reshape(24, 4) / 10
        save_file(
            {"projection.weight": self.weight},
            str(Path(self.directory.name, "jev_head.safetensors")),
        )

    def config(self, **overrides):
        config = dict(
            ckpt_path=self.directory.name,
            hidden_size=4,
            compute_dtype=torch.bfloat16,
            quant_algo=SimpleNamespace(isQuant=lambda: False),
            quantization=None,
            quant_config=None,
        )
        config.update(overrides)
        return SimpleNamespace(**config)

    def fp8(self, **overrides):
        config = dict(
            quantization="FP8_PER_BLOCK",
            quant_config=SimpleNamespace(get_method=lambda: "FP8_PER_BLOCK"),
            quant_algo=SimpleNamespace(
                isQuant=lambda: True,
                isFp8=lambda: True,
                getWeightBits=lambda: 8,
                getGroupSize=lambda: 128,
            ),
        )
        config.update(overrides)
        return self.config(**config)

    def test_separate_head_load_preserves_fp32_for_bf16_and_resolved_fp8(self):
        for config in (self.config(), self.fp8(), self.fp8(quantization=None)):
            module = JevModule(config, self.tokenizer)
            self.assertEqual(module.get_custom_weight_info(), [])
            module.init(None)
            handler = module.get_handler()
            self.assertEqual(
                handler.extend_forward_args(),
                ["hidden_states", "input_ids", "input_lengths", "prefix_lengths"],
            )
            self.assertEqual(handler._weight.dtype, torch.float32)
            torch.testing.assert_close(handler._weight, self.weight, rtol=0, atol=0)

    def test_tokenizer_wrapper_and_backbone_contract(self):
        JevModule(
            self.config(), SimpleNamespace(get_real_tokenizer=lambda: self.tokenizer)
        )
        with self.assertRaisesRegex(ValueError, "hidden_size"):
            JevModule(self.config(hidden_size=8), self.tokenizer)
        self.tokenizer.mapping["A"] = 999
        with self.assertRaisesRegex(ValueError, "tokenizer"):
            JevModule(self.config(), self.tokenizer)
        for dtype in (torch.float16, torch.float32):
            with self.assertRaisesRegex(ValueError, "act_type bf16"):
                JevModule(self.config(compute_dtype=dtype), None)

    def test_unsupported_or_unresolved_quantization_is_rejected(self):
        configs = (
            self.config(quantization="FP8_PER_BLOCK"),
            self.config(quant_algo=SimpleNamespace(isQuant=lambda: True)),
            self.config(enable_w4a16_sm120_dense_ffn=True),
            self.fp8(enable_nvfp4_dense_ffn=True),
            self.fp8(quant_config=SimpleNamespace(get_method=lambda: "FP8")),
            self.fp8(
                quant_algo=SimpleNamespace(
                    isQuant=lambda: True,
                    isFp8=lambda: True,
                    getWeightBits=lambda: 8,
                    getGroupSize=lambda: 64,
                )
            ),
        )
        for config in configs:
            with self.subTest(config=config), self.assertRaisesRegex(
                ValueError, "only BF16 or FP8_PER_BLOCK"
            ):
                JevModule(config, self.tokenizer)


if __name__ == "__main__":
    unittest.main()
