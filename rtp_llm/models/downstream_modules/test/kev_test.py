import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from rtp_llm.models.downstream_modules.kev import (
    SPECIAL,
    KevConfig,
    KevHandler,
    KevModule,
    encode_kev_rows,
    load_kev_config,
    validate_kev_row,
)

try:
    import torch
except ImportError:
    torch = None


class FakeTokenizer:
    """One token per character, with real delimiter recognition for injection tests."""

    def __init__(self):
        self.calls = []

    def convert_tokens_to_ids(self, token):
        return SPECIAL.index(token) + 1

    def __call__(self, text, *, add_special_tokens):
        if add_special_tokens:
            raise AssertionError("Kev must tokenize each segment without extra tokens")
        self.calls.append(text)
        result = []
        while text:
            for index, special in enumerate(SPECIAL, start=1):
                if text.startswith(special):
                    result.append(index)
                    text = text[len(special) :]
                    break
            else:
                result.append(ord(text[0]) + 1000)
                text = text[1:]
        return SimpleNamespace(input_ids=result)


def manifest():
    return {
        "format_version": 1,
        "hidden_size": 4,
        "head_dim": 3,
        "temperature": 1.32,
        "special_token_ids": [1, 2, 3, 4, 5],
        "max_options": 4,
        "base_model": "Qwen/Qwen3.5-4B",
        "base_revision": "base-sha",
        "source_revision": "kev-sha",
    }


class KevCodecTest(unittest.TestCase):
    def encode(self, state="ab", questions=None, **limits):
        tokenizer = FakeTokenizer()
        if questions is None:
            questions = [{"instr": "x", "options": ["a", "b"]}]
        rows = encode_kev_rows(
            tokenizer,
            state,
            questions,
            max_state_tokens=limits.get("max_state_tokens", 64),
            max_row_tokens=limits.get("max_row_tokens", 128),
            max_options=limits.get("max_options", 4),
        )
        return tokenizer, rows

    def test_official_encoding_row_fixture(self):
        # Golden rows from kev.model encode(strict=True) + rows_of(), revision
        # 1f04cfa28e6a708485b28a9585010b9e436b776e, with FakeTokenizer above.
        tokenizer, rows = self.encode(
            questions=[
                {"instr": "x", "options": ["a", "b"]},
                {"instr": "Y", "options": ["no"]},
            ]
        )
        self.assertEqual(
            rows,
            [
                [1, 1097, 1098, 2, 1120, 3, 1097, 4, 3, 1098, 4, 5],
                [1, 1097, 1098, 2, 1089, 3, 1110, 1111, 4, 5],
            ],
        )
        self.assertEqual(tokenizer.calls, ["ab", "x", "a", "b", "Y", "no"])
        self.assertEqual(validate_kev_row(rows[0], [1, 2, 3, 4, 5], 4), ([7, 10], 11))

    def test_special_tokens_are_escaped_per_segment(self):
        tokenizer, rows = self.encode(
            state="<|fim_middle|>",
            questions=[{"instr": "<|fim_suffix|>", "options": ["<|box_end|>"]}],
        )
        self.assertEqual(
            tokenizer.calls, ["<¦fim_middle¦>", "<¦fim_suffix¦>", "<¦box_end¦>"]
        )
        for delimiter in range(1, 6):
            self.assertEqual(rows[0].count(delimiter), 1)

    def test_exact_limits_and_overflow_rejection(self):
        _, rows = self.encode(max_state_tokens=3, max_row_tokens=12)
        self.assertEqual(len(rows[0]), 12)
        for limits in ({"max_state_tokens": 2}, {"max_row_tokens": 11}):
            with self.subTest(limits=limits), self.assertRaises(ValueError):
                self.encode(**limits)

    def test_bad_request_shapes(self):
        invalid = [
            [],
            "question",
            [{}],
            [{"instr": 1, "options": ["yes"]}],
            [{"instr": "", "options": []}],
            [{"instr": "", "options": "yes"}],
            [{"instr": "", "options": [None]}],
            [{"instr": "", "options": [""]}],
            [{"instr": "", "options": ["x"] * 5}],
        ]
        for questions in invalid:
            with self.subTest(questions=questions), self.assertRaises(ValueError):
                self.encode(questions=questions)
        with self.assertRaises(ValueError):
            self.encode(state=None)

    def test_invalid_protocol_rows(self):
        valid = [1, 101, 2, 102, 3, 103, 4, 5]
        invalid = [
            [],
            valid[1:],
            valid[:-1],
            [1, 2, 5],
            [1, 3, 101, 4, 5],
            [1, 1, 2, 3, 101, 4, 5],
            [1, 2, 2, 3, 101, 4, 5],
            [1, 2, 3, 3, 101, 4, 5],
            [1, 2, 4, 5],
            [1, 2, 3, 101, 5],
            [1, 2, 3, 101, 4, 102, 5],
            [1, 2, 3, 101, 4, 5, 5],
            [1, 2, 3, 101, 4, 3, 102, 4, 5],
        ]
        for row in invalid:
            with self.subTest(row=row), self.assertRaises(ValueError):
                validate_kev_row(row, [1, 2, 3, 4, 5], 1)


class KevConfigTest(unittest.TestCase):
    def test_defaults_and_ordinary_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(load_kev_config(directory))
            data = manifest()
            del data["max_options"]
            Path(directory, "kev_config.json").write_text(json.dumps(data))
            loaded = load_kev_config(directory)
        self.assertEqual(loaded["max_options"], 32)
        self.assertEqual(loaded["max_state_tokens"], 65536)
        self.assertEqual(loaded["max_row_tokens"], 73728)
        self.assertEqual(loaded["special_token_ids"], [1, 2, 3, 4, 5])

    def test_invalid_manifest(self):
        invalid_fields = [
            ("format_version", True),
            ("format_version", 2),
            ("hidden_size", 0),
            ("head_dim", -1),
            ("temperature", float("nan")),
            ("temperature", 0),
            ("temperature", True),
            ("special_token_ids", [1, 2, 3, 4, 4]),
            ("max_options", 0),
            ("max_state_tokens", 0),
            ("max_row_tokens", -1),
            ("source_revision", ""),
            ("option_isolation", True),
        ]
        with tempfile.TemporaryDirectory() as directory:
            for key, value in invalid_fields:
                data = dict(manifest(), **{key: value})
                Path(directory, "kev_config.json").write_text(json.dumps(data))
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    load_kev_config(directory)


class KevFactoryTest(unittest.TestCase):
    def setUp(self):
        # The factory's provider ordering is independent of native config types.
        # Stub only these imports so the regression can run on a CPU-only host.
        self.task_types = SimpleNamespace(LANGUAGE_MODEL=1, DENSE_EMBEDDING=2)
        stubs = {
            "rtp_llm.config.model_config": SimpleNamespace(ModelConfig=object),
            "rtp_llm.frontend.tokenizer_factory.tokenizers": SimpleNamespace(
                BaseTokenizer=object
            ),
            "rtp_llm.ops": SimpleNamespace(TaskType=self.task_types),
        }
        path = Path(__file__).resolve().parents[1] / "utils.py"
        spec = importlib.util.spec_from_file_location("kev_factory_under_test", path)
        self.factory = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, stubs):
            spec.loader.exec_module(self.factory)

    def test_kev_marker_precedes_internal_provider(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "kev_config.json").write_text(json.dumps(manifest()))
            config = SimpleNamespace(
                ckpt_path=directory,
                hidden_size=4,
                task_type=self.task_types.LANGUAGE_MODEL,
            )
            with (
                mock.patch.object(
                    self.factory, "import_optional_internal_source_entrypoint"
                ) as provider,
                mock.patch(
                    "rtp_llm.models.downstream_modules.kev.KevModule"
                ) as module_type,
            ):
                tokenizer = FakeTokenizer()
                module = self.factory.create_custom_module(config, tokenizer)
                self.assertIs(module, module_type.return_value)
                module_type.assert_called_once_with(config, tokenizer)
                provider.assert_not_called()

    def test_ordinary_generation_preserves_absent_provider_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            config = SimpleNamespace(
                ckpt_path=directory, task_type=self.task_types.LANGUAGE_MODEL
            )
            with mock.patch.object(
                self.factory,
                "import_optional_internal_source_entrypoint",
                return_value=False,
            ) as provider:
                self.assertIsNone(self.factory.create_custom_module(config, None))
                provider.assert_called_once()

    def test_marker_rejects_non_language_task(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "kev_config.json").write_text(json.dumps(manifest()))
            config = SimpleNamespace(
                ckpt_path=directory, task_type=self.task_types.DENSE_EMBEDDING
            )
            with self.assertRaisesRegex(ValueError, "LANGUAGE_MODEL"):
                self.factory.create_custom_module(config, FakeTokenizer())


@unittest.skipIf(torch is None, "PyTorch is required for the pointer head tests")
class KevHeadTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        Path(self.directory.name, "kev_config.json").write_text(json.dumps(manifest()))
        self.config = KevConfig.load(self.directory.name)
        self.handler = KevHandler(self.config)
        generator = torch.Generator().manual_seed(17)
        self.weights = {
            name: torch.randn(shape, generator=generator) * 0.2
            for name, shape in {
                "q.weight": (3, 4),
                "q.bias": (3,),
                "k.weight": (3, 4),
                "k.bias": (3,),
            }.items()
        }
        self.handler.init(self.weights)
        self.rows = [
            [1, 101, 2, 102, 3, 103, 4, 3, 104, 4, 5],
            [1, 105, 2, 3, 106, 4, 5],
            [1, 2, 3, 107, 4, 3, 108, 4, 3, 109, 4, 5],
        ]
        self.hidden = torch.randn(
            (sum(map(len, self.rows)), 4), generator=generator
        ).to(torch.bfloat16)

    def run_rows(self, rows, hidden):
        return self.handler.extend_forward(
            hidden_states=hidden,
            input_ids=torch.tensor([token for row in rows for token in row]),
            input_lengths=torch.tensor(list(map(len, rows))),
        )

    def test_biased_fp32_head_matches_scalar_reference(self):
        result = self.run_rows(self.rows, self.hidden)
        self.assertEqual(result.dtype, torch.float32)
        self.assertEqual(result.shape, (3, 4))
        offset = 0
        for index, row in enumerate(self.rows):
            ends, decide = validate_kev_row(row, self.config.special_token_ids, 4)
            reference_hidden = self.hidden[offset : offset + len(row)].double()
            query = (
                self.weights["q.weight"].double() @ reference_hidden[decide]
                + self.weights["q.bias"].double()
            )
            scores = []
            for end in ends:
                key = (
                    self.weights["k.weight"].double() @ reference_hidden[end]
                    + self.weights["k.bias"].double()
                )
                scores.append(torch.dot(key, query) / math.sqrt(3) / 1.32)
            expected = torch.stack(scores).softmax(0).float()
            torch.testing.assert_close(
                result[index, : len(ends)], expected, rtol=1e-6, atol=1e-7
            )
            self.assertEqual(result[index, len(ends) :].count_nonzero().item(), 0)
            offset += len(row)

    def test_batch_partition_and_autocast_do_not_change_scores(self):
        together = self.run_rows(self.rows, self.hidden)
        split = []
        offset = 0
        for row in self.rows:
            split.append(self.run_rows([row], self.hidden[offset : offset + len(row)]))
            offset += len(row)
        torch.testing.assert_close(together, torch.cat(split), rtol=1e-6, atol=1e-7)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            under_autocast = self.run_rows(self.rows, self.hidden)
        torch.testing.assert_close(together, under_autocast, rtol=0, atol=0)
        torch.testing.assert_close(
            together, self.run_rows(self.rows, self.hidden.float()), rtol=0, atol=0
        )

    def test_admission_rejects_multimodal_and_incomplete_rows(self):
        tokens = torch.tensor(self.rows[0])
        self.assertEqual(self.handler.select_token_position(tokens), len(tokens) - 1)
        mask = torch.ones_like(tokens)
        mask[1] = 0
        with self.assertRaisesRegex(ValueError, "text-only"):
            self.handler.select_token_position(tokens, mask)
        with self.assertRaises(ValueError):
            self.handler.select_token_position(tokens[:-1])
        with self.assertRaises(ValueError):
            self.handler.select_token_position(tokens.float())
        with self.assertRaisesRegex(ValueError, "complete"):
            self.run_rows(self.rows, self.hidden[1:])

    def test_head_loading_rejects_missing_or_low_precision_weights(self):
        with self.assertRaises(ValueError):
            self.handler.init(
                {name: value.half() for name, value in self.weights.items()}
            )
        incomplete = dict(self.weights)
        del incomplete["q.bias"]
        with self.assertRaises(ValueError):
            self.handler.init(incomplete)
        nonfinite = dict(self.weights)
        nonfinite["q.bias"] = torch.full((3,), float("nan"))
        with self.assertRaises(ValueError):
            self.handler.init(nonfinite)

    def test_module_loads_separate_safetensors_without_model_weight_casting(self):
        from safetensors.torch import save_file

        save_file(self.weights, str(Path(self.directory.name, "kev_head.safetensors")))
        config = self.module_config()
        module = KevModule(config, FakeTokenizer())
        self.assertEqual(module.get_custom_weight_info(), [])
        module.init(None)
        self.assertTrue(module.get_handler().prefill_only)
        self.assertEqual(
            module.get_handler().extend_forward_args(),
            ["hidden_states", "input_ids", "input_lengths"],
        )
        self.handler = module.get_handler()
        self.assertEqual(self.run_rows(self.rows, self.hidden).dtype, torch.float32)

    def module_config(self, **overrides):
        fields = {
            "ckpt_path": self.directory.name,
            "hidden_size": 4,
            "compute_dtype": torch.bfloat16,
            "quant_algo": SimpleNamespace(isQuant=lambda: False),
            "quantization": None,
            "quant_config": None,
        }
        fields.update(overrides)
        return SimpleNamespace(**fields)

    def test_module_rejects_backbone_dtypes_other_than_bf16(self):
        for dtype in (torch.float16, torch.float32):
            with (
                self.subTest(dtype=dtype),
                self.assertRaisesRegex(ValueError, "--act_type bf16"),
            ):
                KevModule(self.module_config(compute_dtype=dtype), FakeTokenizer())

    def test_module_rejects_quantized_backbone_before_loading_head(self):
        for overrides in (
            {"quant_algo": SimpleNamespace(isQuant=lambda: True)},
            {"quantization": "fp8"},
            {"quant_config": object()},
        ):
            with (
                self.subTest(overrides=overrides),
                self.assertRaisesRegex(ValueError, "quantized backbone"),
            ):
                KevModule(self.module_config(**overrides), FakeTokenizer())


if __name__ == "__main__":
    unittest.main()
