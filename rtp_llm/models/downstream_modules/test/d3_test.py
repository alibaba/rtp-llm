import importlib.util
import json
import string
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from safetensors.torch import save_file

import rtp_llm.model_factory_register as model_registry
from rtp_llm.model_factory_register import ModelDict, _model_factory
from rtp_llm.models.downstream_modules.d3 import D3Handler, D3Module
from rtp_llm.models.downstream_modules.d3_config import D3Config, load_d3_config
from rtp_llm.models.qwen3_next.qwen3_next import Qwen35Dense


def answer_codes():
    return (
        list(string.ascii_uppercase)
        + [a + b for a in string.ascii_uppercase for b in string.ascii_uppercase]
    )[:255]


def decision_config(**overrides):
    data = {
        "format_version": 1,
        "format_id": "d3-code-readout-v1",
        "prompt": "d3",
        "codes": answer_codes(),
        "token_ids": list(range(1000, 1000 + 255)),
        "temperature": 1.0,
        "attention_mode": "noncausal_full_attention",
        "pooling": "last",
        "max_length": None,
        "readout_dtype": "float32",
    }
    data.update(overrides)
    return data


def backbone_config(hidden_size=4):
    layer_types = ["linear_attention"] * 32
    for index in range(3, 32, 4):
        layer_types[index] = "full_attention"
    return {
        "architectures": ["Qwen3_5Model"],
        "model_type": "qwen3_5",
        "vision_start_token_id": 248053,
        "vision_end_token_id": 248054,
        "text_config": {
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "num_hidden_layers": 32,
            "hidden_size": hidden_size,
            "vocab_size": 248320,
            "max_position_embeddings": 262144,
            "rms_norm_eps": 1e-06,
            "intermediate_size": 9216,
            "full_attention_interval": 4,
            "layer_types": layer_types,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_value_head_dim": 128,
            "rope_parameters": {
                "mrope_interleaved": True,
                "mrope_section": [11, 11, 10],
                "partial_rotary_factor": 0.25,
                "rope_theta": 10000000,
                "rope_type": "default",
            },
            "tie_word_embeddings": True,
        },
    }


class FakeTokenizer:
    def __init__(self, data):
        self.mapping = dict(zip(data["codes"], data["token_ids"]))

    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return [self.mapping[text]]


class D3ConfigTest(unittest.TestCase):
    def checkpoint(self, sidecar=True, backbone=True, **overrides):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        if backbone:
            Path(directory.name, "config.json").write_text(
                json.dumps(backbone_config())
            )
        if sidecar:
            Path(directory.name, "decision_config.json").write_text(
                json.dumps(decision_config(**overrides))
            )
        return directory.name

    def test_loads_sidecar_fields_and_backbone_identity(self):
        self.assertIsNone(load_d3_config(self.checkpoint(sidecar=False)))
        directory = self.checkpoint()
        config = D3Config.load(directory)
        self.assertEqual(config, load_d3_config(directory))
        self.assertEqual(config.format_version, 1)
        self.assertEqual(config.format_id, "d3-code-readout-v1")
        self.assertEqual(config.codes, tuple(answer_codes()))
        self.assertEqual(config.token_ids, tuple(range(1000, 1255)))
        self.assertEqual(config.temperature, 1.0)
        self.assertEqual(config.attention_mode, "noncausal_full_attention")
        self.assertEqual(config.pooling, "last")
        self.assertIsNone(config.max_length)
        self.assertEqual(config.readout_dtype, "float32")
        self.assertEqual(config.hidden_size, 4)
        self.assertEqual(config.readout_file, "readout.safetensors")
        self.assertEqual(config.readout_key, "weight")
        limited = D3Config.load(self.checkpoint(max_length=512, temperature=2))
        self.assertEqual(limited.max_length, 512)
        self.assertEqual(limited.temperature, 2.0)

    def test_rejects_invalid_sidecar_metadata(self):
        cases = [
            ("format_version", True),
            ("format_version", 2),
            ("format_id", "d3-code-readout-v2"),
            ("prompt", "other"),
            ("codes", answer_codes()[:254]),
            ("codes", ["A"] * 255),
            ("codes", "ABC"),
            ("token_ids", list(range(1000, 1254))),
            ("token_ids", [1000] * 255),
            ("token_ids", [True] + list(range(1001, 1255))),
            ("token_ids", [-1] + list(range(1001, 1255))),
            ("temperature", 0),
            ("temperature", -1.5),
            ("temperature", float("inf")),
            ("temperature", True),
            ("attention_mode", "bidirectional"),
            ("pooling", "mean"),
            ("max_length", 0),
            ("max_length", True),
            ("readout_dtype", "float16"),
        ]
        for key, value in cases:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                D3Config.load(self.checkpoint(**{key: value}))
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "decision_config.json").write_text("{not json")
            with self.assertRaises(ValueError):
                D3Config.load(directory)
            Path(directory, "decision_config.json").write_text("[]")
            with self.assertRaises(ValueError):
                D3Config.load(directory)

    def test_rejects_incompatible_backbone_config(self):
        def mutate(key):
            return {
                "drop_model_type": lambda b: b.pop("model_type"),
                "vl_backbone": lambda b: b.update(model_type="qwen3_vl"),
                "text_only_backbone": lambda b: b.update(model_type="qwen3_5_text"),
                "drop_hidden": lambda b: b["text_config"].pop("hidden_size"),
                "zero_hidden": lambda b: b["text_config"].update(hidden_size=0),
                "bad_layer_types": lambda b: b["text_config"].update(
                    layer_types=["dense"] * 32
                ),
                "no_full_attention": lambda b: b["text_config"].update(
                    layer_types=["linear_attention"] * 32
                ),
                "flat_backbone": lambda b: b.pop("text_config"),
            }[key]

        for key in (
            "drop_model_type",
            "vl_backbone",
            "text_only_backbone",
            "drop_hidden",
            "zero_hidden",
            "bad_layer_types",
            "no_full_attention",
            "flat_backbone",
        ):
            with self.subTest(key=key):
                backbone = backbone_config()
                mutate(key)(backbone)
                with tempfile.TemporaryDirectory() as directory:
                    Path(directory, "config.json").write_text(json.dumps(backbone))
                    Path(directory, "decision_config.json").write_text(
                        json.dumps(decision_config())
                    )
                    with self.assertRaises(ValueError):
                        D3Config.load(directory)
        with self.assertRaises(ValueError):
            D3Config.load(self.checkpoint(backbone=False))

    def test_sidecar_module_imports_without_torch_or_native_runtime(self):
        code = (
            "import sys; import rtp_llm.models.downstream_modules.d3_config; "
            "assert 'torch' not in sys.modules; "
            "assert not any(m == 'rtp_llm.ops' or m.startswith('rtp_llm.ops.') "
            "for m in sys.modules)"
        )
        subprocess.run([sys.executable, "-c", code], check=True)


class D3HandlerTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        Path(self.directory.name, "decision_config.json").write_text(
            json.dumps(decision_config())
        )
        Path(self.directory.name, "config.json").write_text(
            json.dumps(backbone_config())
        )
        self.config = D3Config.load(self.directory.name)
        self.handler = D3Handler(self.config)
        generator = torch.Generator().manual_seed(41)
        self.weight = torch.randn((255, 4), generator=generator) * 0.25
        self.handler.init({"weight": self.weight})
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
        self.assertEqual(actual.shape, (3, 255))
        self.assertEqual(actual.dtype, torch.float32)
        # No candidate mask, bias, temperature or softmax belongs in the engine
        # output; those run exactly once, in the frontend.
        self.assertTrue((actual < 0).any())
        self.assertFalse(torch.allclose(actual.sum(-1), torch.ones(3)))
        Path(self.directory.name, "decision_config.json").write_text(
            json.dumps(decision_config(temperature=0.125, readout_dtype="float32"))
        )
        other = D3Handler(D3Config.load(self.directory.name))
        other.init({"weight": self.weight})
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

    def test_partition_preserves_every_readout(self):
        expected = self.forward()
        split = [
            self.forward(
                hidden_states=self.hidden[offset : offset + length],
                input_ids=self.ids[offset : offset + length],
                input_lengths=torch.tensor([length]),
            )
            for offset, length in zip((0, 4, 12), (4, 8, 3))
        ]
        # CPU GEMM picks its micro-kernel by row count (gemv for one row,
        # sgemm for several), and the two accumulate a four-element dot product
        # in a different order, so on some CPUs a partitioned run differs from
        # the whole-batch run by one ulp of float32 (about 1.2e-07 here). The
        # guarantee under test is that partitioning never changes a readout
        # materially — a wrong position, row or weight is an O(1) error — not
        # that the kernel is identical for every row count.
        torch.testing.assert_close(torch.cat(split), expected, rtol=1e-6, atol=1e-6)

    def test_autocast_does_not_round_fp32_readout_or_apply_final_norm_again(self):
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
        self.assertTrue(self.handler.prefill_only)
        self.assertFalse(self.handler.supports_prefix_cache)
        self.assertEqual(
            self.handler.extend_forward_args(),
            ["hidden_states", "input_ids", "input_lengths"],
        )
        # Readout uses expanded row lengths, including image features, rather
        # than selecting the last text position from a compressed prompt.
        hidden = torch.zeros((len(ids), 4))
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
            with (
                self.subTest(ids=ids),
                self.assertRaisesRegex(ValueError, "CPU token row"),
            ):
                self.handler.select_token_position(ids)
        for mask in (
            torch.tensor([1, 0, 0]),
            torch.ones(2),
            torch.ones((1, 3)),
            torch.empty(3, dtype=torch.int64, device="meta"),
        ):
            with (
                self.subTest(mask=mask),
                self.assertRaisesRegex(ValueError, "text_tokens_mask"),
            ):
                self.handler.select_token_position(tokens, mask)
        Path(self.directory.name, "decision_config.json").write_text(
            json.dumps(decision_config(max_length=8))
        )
        limited = D3Handler(D3Config.load(self.directory.name))
        limited.init({"weight": self.weight})
        with self.assertRaisesRegex(ValueError, "max_length"):
            limited.select_token_position(torch.arange(9))
        self.assertEqual(limited.select_token_position(torch.arange(8)), 7)

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
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.forward(**overrides)

    def test_readout_load_rejects_bad_shape_dtype_nonfinite_or_extra_weights(self):
        for tensors in (
            {},
            {"weight": self.weight, "bias": torch.zeros(255)},
            {"readout.weight": self.weight},
            {"weight": self.weight[:, :3]},
            {"weight": self.weight.half()},
            {"weight": torch.full_like(self.weight, float("nan"))},
        ):
            with self.subTest(keys=tensors.keys()), self.assertRaises(ValueError):
                D3Handler(self.config).init(tensors)
        self.handler = D3Handler(self.config)
        with self.assertRaisesRegex(RuntimeError, "initialized"):
            self.forward()

    def test_forward_never_materializes_token_ids_or_lengths_as_python_lists(self):
        with mock.patch.object(
            torch.Tensor, "tolist", side_effect=AssertionError("unexpected host copy")
        ):
            result = self.forward()
        self.assertEqual(result.shape, (3, 255))


class D3ModuleTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        Path(self.directory.name, "decision_config.json").write_text(
            json.dumps(decision_config())
        )
        Path(self.directory.name, "config.json").write_text(
            json.dumps(backbone_config())
        )
        self.tokenizer = FakeTokenizer(decision_config())
        self.weight = torch.arange(255 * 4, dtype=torch.float32).reshape(255, 4) / 100
        save_file(
            {"weight": self.weight},
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

    def fp8_block(self, **overrides):
        from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig

        fields = {
            "quantization": "FP8_PER_BLOCK",
            "quant_config": Fp8BlockWiseQuantConfig(),
            "quant_algo": SimpleNamespace(
                isQuant=lambda: True,
                isFp8=lambda: True,
                getWeightBits=lambda: 8,
                getGroupSize=lambda: 128,
            ),
        }
        fields.update(overrides)
        return self.config(**fields)

    def fp8_channel(self, **overrides):
        from rtp_llm.config.quant_config import Fp8PerChannelCompressedQuantConfig

        fields = {
            "quantization": "FP8_PER_CHANNEL_COMPRESSED",
            "quant_config": Fp8PerChannelCompressedQuantConfig(),
            "quant_algo": SimpleNamespace(
                isQuant=lambda: True,
                isFp8=lambda: False,
                isFp8PTPC=lambda: True,
                getWeightBits=lambda: 8,
                getGroupSize=lambda: 0,
            ),
        }
        fields.update(overrides)
        return self.config(**fields)

    def test_separate_readout_load_preserves_fp32_per_platform(self):
        for config, hip in (
            (self.config(), None),
            (self.fp8_block(quantization=None), None),
            (self.fp8_block(), None),
            (self.fp8_channel(quantization=None), "7.2.0"),
            (self.fp8_channel(), "7.2.0"),
        ):
            with (
                self.subTest(hip=hip),
                mock.patch.multiple(
                    torch.version, hip=hip, cuda=None if hip else "13.2"
                ),
            ):
                module = D3Module(config, self.tokenizer)
                self.assertEqual(module.get_custom_weight_info(), [])
                module.init(None)
                handler = module.get_handler()
                self.assertEqual(
                    handler.extend_forward_args(),
                    ["hidden_states", "input_ids", "input_lengths"],
                )
                self.assertEqual(handler._weight.dtype, torch.float32)
                torch.testing.assert_close(handler._weight, self.weight, rtol=0, atol=0)

    def test_module_rejects_platform_mismatched_quantization(self):
        with (
            mock.patch.object(torch.version, "hip", "7.2.0"),
            self.assertRaisesRegex(ValueError, "ROCm also supports"),
        ):
            D3Module(self.fp8_block(), self.tokenizer)
        for cuda_version in (None, "13.2"):
            with (
                self.subTest(cuda_version=cuda_version),
                mock.patch.multiple(torch.version, hip=None, cuda=cuda_version),
                self.assertRaisesRegex(ValueError, "ROCm also supports"),
            ):
                D3Module(self.fp8_channel(), self.tokenizer)

    def test_module_rejects_invalid_rocm_fp8_per_channel_config(self):
        from rtp_llm.config.quant_config import Fp8PerChannelCompressedQuantConfig

        valid_algo = vars(self.fp8_channel().quant_algo)
        invalid = [
            {"quant_config": None},
            {"quant_config": Fp8PerChannelCompressedQuantConfig(bits=4)},
            {
                "quant_config": SimpleNamespace(
                    get_method=lambda: "FP8_PER_CHANNEL_COMPRESSED",
                    bits=8,
                    group_size=lambda: 128,
                )
            },
            {"enable_nvfp4_dense_ffn": True},
            {"enable_w4a16_sm120_dense_ffn": True},
        ]
        for method in ("FP8_PER_CHANNEL_QUARK", "FP8_PER_TENSOR_COMPRESSED", "FP8"):
            invalid.append(
                {"quant_config": SimpleNamespace(get_method=lambda m=method: m)}
            )
        for name, value in (
            ("isQuant", False),
            ("isFp8PTPC", False),
            ("getWeightBits", 4),
            ("getGroupSize", 128),
        ):
            invalid.append(
                {
                    "quant_algo": SimpleNamespace(
                        **{**valid_algo, name: lambda v=value: v}
                    )
                }
            )
        for overrides in invalid:
            with (
                self.subTest(overrides=overrides),
                mock.patch.object(torch.version, "hip", "7.2.0"),
                self.assertRaisesRegex(ValueError, "only BF16 or FP8_PER_BLOCK"),
            ):
                D3Module(self.fp8_channel(**overrides), self.tokenizer)

    def test_module_rejects_unresolved_or_other_cuda_quantization(self):
        for config in (
            self.config(quantization="FP8_PER_BLOCK"),
            self.config(quant_algo=SimpleNamespace(isQuant=lambda: True)),
            self.config(quantization="fp8"),
            self.config(quant_config=object()),
            self.fp8_block(
                quant_algo=SimpleNamespace(
                    isQuant=lambda: True,
                    isFp8=lambda: True,
                    getWeightBits=lambda: 8,
                    getGroupSize=lambda: 64,
                )
            ),
            self.fp8_block(quant_config=SimpleNamespace(get_method=lambda: "FP8")),
            self.fp8_block(enable_nvfp4_dense_ffn=True),
            self.fp8_block(enable_w4a16_sm120_dense_ffn=True),
        ):
            with (
                self.subTest(config=config),
                mock.patch.multiple(torch.version, hip=None, cuda="13.2"),
                self.assertRaisesRegex(ValueError, "only BF16 or FP8_PER_BLOCK"),
            ):
                D3Module(config, self.tokenizer)

    def test_tokenizer_wrapper_and_backbone_contract(self):
        D3Module(
            self.config(), SimpleNamespace(get_real_tokenizer=lambda: self.tokenizer)
        )
        with self.assertRaisesRegex(ValueError, "hidden_size"):
            D3Module(self.config(hidden_size=8), self.tokenizer)
        self.tokenizer.mapping["A"] = 999
        with self.assertRaisesRegex(ValueError, "tokenizer"):
            D3Module(self.config(), self.tokenizer)
        for dtype in (torch.float16, torch.float32):
            with self.assertRaisesRegex(ValueError, "act_type bf16"):
                D3Module(self.config(compute_dtype=dtype), None)
        # The noncausal sidecar must meet a noncausal model config.
        with self.assertRaisesRegex(ValueError, "is_causal"):
            D3Module(self.config(attn_config=SimpleNamespace(is_causal=True)), None)
        Path(self.directory.name, "decision_config.json").write_text(
            json.dumps(decision_config(attention_mode="causal"))
        )
        D3Module(self.config(attn_config=SimpleNamespace(is_causal=True)), None)
        with self.assertRaisesRegex(ValueError, "keep attn_config.is_causal=True"):
            D3Module(self.config(), None)
        Path(self.directory.name, "decision_config.json").write_text(
            json.dumps(decision_config(readout_dtype="bfloat16"))
        )
        with self.assertRaisesRegex(ValueError, "float32"):
            D3Module(self.config(), None)


class D3FactoryTest(unittest.TestCase):
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
        spec = importlib.util.spec_from_file_location("d3_factory_under_test", path)
        self.factory = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, stubs):
            spec.loader.exec_module(self.factory)

    def test_d3_marker_precedes_internal_provider(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "decision_config.json").write_text(
                json.dumps(decision_config())
            )
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
                    "rtp_llm.models.downstream_modules.d3.D3Module"
                ) as module_type,
            ):
                tokenizer = FakeTokenizer(decision_config())
                module = self.factory.create_custom_module(config, tokenizer)
                self.assertIs(module, module_type.return_value)
                module_type.assert_called_once_with(config, tokenizer)
                provider.assert_not_called()

    def test_manifest_conflicts_are_rejected(self):
        for other in ("jev_config.json", "kev_config.json"):
            with tempfile.TemporaryDirectory() as directory:
                Path(directory, "decision_config.json").write_text(
                    json.dumps(decision_config())
                )
                Path(directory, other).write_text("{}")
                config = SimpleNamespace(
                    ckpt_path=directory, task_type=self.task_types.LANGUAGE_MODEL
                )
                with (
                    self.subTest(other=other),
                    self.assertRaisesRegex(ValueError, "cannot contain both D3"),
                ):
                    self.factory.create_custom_module(config, None)

    def test_marker_rejects_non_language_task(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "decision_config.json").write_text(
                json.dumps(decision_config())
            )
            config = SimpleNamespace(
                ckpt_path=directory, task_type=self.task_types.DENSE_EMBEDDING
            )
            with self.assertRaisesRegex(ValueError, "LANGUAGE_MODEL"):
                tokenizer = FakeTokenizer(decision_config())
                self.factory.create_custom_module(config, tokenizer)


class Qwen35DenseD3ConfigTest(unittest.TestCase):
    def test_d3_sidecar_switches_only_full_attention_layers_to_noncausal(self):
        with tempfile.TemporaryDirectory() as directory:
            config_json = backbone_config()
            # The sidecar binding reads the checkpoint's real vision token
            # ids (missing ids fail loudly, like the staged d3 checkpoint);
            # the valid d3 sidecar below must exercise the binding end to end.
            config_json["image_token_id"] = 248056
            config_json["video_token_id"] = 248057
            Path(directory, "config.json").write_text(json.dumps(config_json))
            Path(directory, "decision_config.json").write_text(
                json.dumps(decision_config())
            )
            d3 = Qwen35Dense.create_config(directory)
            Path(directory, "decision_config.json").write_text(
                json.dumps(decision_config(attention_mode="causal"))
            )
            causal_d3 = Qwen35Dense.create_config(directory)
            Path(directory, "decision_config.json").unlink()
            ordinary = Qwen35Dense.create_config(directory)
            Path(directory, "decision_config.json").write_text(
                json.dumps(decision_config(format_version=2))
            )
            with self.assertRaises(ValueError):
                Qwen35Dense.create_config(directory)

        self.assertFalse(d3.attn_config.is_causal)
        self.assertTrue(causal_d3.attn_config.is_causal)
        self.assertTrue(ordinary.attn_config.is_causal)
        # Every valid d3 sidecar (either attention mode) binds the
        # nested-aware leaf-span descriptors; an ordinary checkpoint keeps
        # the strict 2-separator form.
        expected_nested = [
            [
                config_json["vision_start_token_id"],
                config_json["vision_end_token_id"],
                248056,
            ],
            [
                config_json["vision_start_token_id"],
                config_json["vision_end_token_id"],
                248057,
            ],
        ]
        for model in (d3, causal_d3):
            self.assertEqual(
                [list(sep) for sep in model.mm_model_config.mm_sep_tokens],
                expected_nested,
            )
        self.assertEqual(
            [list(sep) for sep in ordinary.mm_model_config.mm_sep_tokens],
            [
                [
                    config_json["vision_start_token_id"],
                    config_json["vision_end_token_id"],
                ]
            ],
        )
        # Only attention causality changes: the layer layout stays identical,
        # so the Gated DeltaNet layers keep their causal scan.
        self.assertEqual(d3.num_layers, ordinary.num_layers)
        self.assertEqual(d3.hidden_size, ordinary.hidden_size)
        self.assertEqual(
            list(d3.hybrid_attention_config.hybrid_attention_types),
            list(ordinary.hybrid_attention_config.hybrid_attention_types),
        )

    def test_d3_sidecar_binds_nested_leaf_span_separators(self):
        # The d3 prompt nests every feature-bearing leaf pair (the image
        # placeholder's single pad; each 2-frame video group's pad run) inside
        # an outer template pair whose direct content is the group timestamps
        # plus the inner pairs. A d3 sidecar must opt into the nested-aware
        # 3-separator descriptors; ordinary Qwen3.5 keeps the strict
        # 2-separator form byte for byte.
        config_json = backbone_config()
        config_json["image_token_id"] = 248056
        config_json["video_token_id"] = 248057
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "config.json").write_text(json.dumps(config_json))
            Path(directory, "decision_config.json").write_text(
                json.dumps(decision_config())
            )
            nested = Qwen35Dense.create_config(directory)
            Path(directory, "decision_config.json").write_text(
                json.dumps(decision_config(attention_mode="causal"))
            )
            causal_d3 = Qwen35Dense.create_config(directory)
            Path(directory, "decision_config.json").unlink()
            ordinary = Qwen35Dense.create_config(directory)

        vision_start = config_json["vision_start_token_id"]
        vision_end = config_json["vision_end_token_id"]
        expected_nested = [
            [vision_start, vision_end, 248056],
            [vision_start, vision_end, 248057],
        ]
        for model, expected in (
            (nested, expected_nested),
            (causal_d3, expected_nested),
        ):
            with self.subTest(d3=model is nested):
                self.assertEqual(
                    [list(sep) for sep in model.mm_model_config.mm_sep_tokens], expected
                )
        self.assertEqual(
            [list(sep) for sep in ordinary.mm_model_config.mm_sep_tokens],
            [[vision_start, vision_end]],
        )
        # The leaf spans exclude the separators by default, so only the pad
        # runs are replaced with ViT features and every marker stays a token.
        self.assertFalse(ordinary.mm_model_config.include_sep_tokens)
        self.assertFalse(nested.mm_model_config.include_sep_tokens)

    def test_bare_backbone_architecture_routes_to_qwen35_dense(self):
        # Recreate the cold architecture map; the model module imported by this
        # test file must not hide a missing built-in lazy registration.
        with (
            mock.patch.dict(model_registry._hf_architecture_2_ft, {}, clear=True),
            mock.patch.object(model_registry, "_load_internal_legacy_models"),
        ):
            model_registry._register_builtin_lazy_models()
            for architecture in (
                "Qwen3_5Model",
                "Qwen3_5ForCausalLM",
                "Qwen3_5ForConditionalGeneration",
            ):
                with self.subTest(architecture=architecture):
                    self.assertEqual(
                        ModelDict.get_ft_model_type_by_config(
                            {"architectures": [architecture]}
                        ),
                        "qwen35_dense",
                    )
        self.assertIs(_model_factory["qwen35_dense"], Qwen35Dense)


if __name__ == "__main__":
    unittest.main()
