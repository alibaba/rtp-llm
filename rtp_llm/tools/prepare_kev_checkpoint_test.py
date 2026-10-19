"""Small synthetic tests; no model downloads or production checkpoint fixtures."""

import hashlib
import json
import os
import struct
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from rtp_llm.tools import prepare_kev_checkpoint as prepare

try:
    import torch
except ImportError:
    torch = None


try:
    import tokenizers
except ImportError:
    tokenizers = None


@unittest.skipIf(
    tokenizers is None, "tokenizers is supplied by the runtime environment"
)
class TokenizerTest(unittest.TestCase):
    EXTRA_TOKENS = (
        "<tool_response>",
        "</tool_response>",
        "<think>",
        "</think>",
        "<|audio_start|>",
        "<|audio_end|>",
        "<tts_pad>",
        "<tts_text_bos>",
        "<tts_text_eod>",
        "<tts_text_bos_single>",
        "<|audio_pad|>",
    )

    def setUp(self):
        self.tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE())
        self.tokenizer.normalizer = tokenizers.normalizers.NFC()
        source_regex = prepare.KEV_PRETOKENIZE_REGEX.replace(
            r"?\p{L}+", r"?[\p{L}\p{M}]+"
        ).replace(r"[^\s\p{L}\p{N}]", r"[^\s\p{L}\p{M}\p{N}]")
        self.tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Sequence(
            [
                tokenizers.pre_tokenizers.Split(
                    tokenizers.Regex(source_regex), behavior="isolated"
                ),
                tokenizers.pre_tokenizers.ByteLevel(
                    add_prefix_space=False, trim_offsets=False, use_regex=False
                ),
            ]
        )
        self.tokenizer.decoder = tokenizers.decoders.ByteLevel(
            add_prefix_space=False, trim_offsets=False, use_regex=False
        )
        self.tokenizer.post_processor = tokenizers.processors.ByteLevel(
            add_prefix_space=False, trim_offsets=False, use_regex=False
        )
        trainer = tokenizers.trainers.BpeTrainer(
            vocab_size=400,
            initial_alphabet=tokenizers.pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        )
        self.tokenizer.train_from_iterator(
            ["कि कि कि مَرْحَبًا Café naïve 中文 123 "] * 10, trainer
        )
        self.original = tokenizers.Tokenizer.from_str(self.tokenizer.to_str())
        start = self.tokenizer.get_vocab_size()
        # Small synthetic vocabulary; the actual checkpoint checks the pinned IDs.
        self.added = {
            str(start + i): dict(
                content=content,
                single_word=False,
                lstrip=False,
                rstrip=False,
                normalized=False,
                special=i >= 4,
            )
            for i, content in enumerate(self.EXTRA_TOKENS)
        }

    def test_combining_mark_segmentation_matches_reference_qwen2_behavior(self):
        self.assertEqual(
            len(self.original.encode("कि", add_special_tokens=False).ids), 1
        )
        prepare._normalize_tokenizer(self.tokenizer, self.added)
        self.assertEqual(
            self.tokenizer.encode("कि", add_special_tokens=False).ids,
            self.tokenizer.encode("क", add_special_tokens=False).ids
            + self.tokenizer.encode("ि", add_special_tokens=False).ids,
        )
        self.assertEqual(len(self.tokenizer.pre_tokenizer.pre_tokenize_str("कि")), 2)
        self.assertNotEqual(
            self.original.encode("مَرْحَبًا").ids, self.tokenizer.encode("مَرْحَبًا").ids
        )
        self.assertEqual(
            self.tokenizer.encode("Cafe\u0301").ids, self.tokenizer.encode("Café").ids
        )
        self.assertEqual(self.tokenizer.decode(self.tokenizer.encode("कि").ids), "कि")

    def test_all_eleven_missing_tokens_are_added_at_exact_ids(self):
        prepare._normalize_tokenizer(self.tokenizer, self.added)
        for token_id, attributes in self.added.items():
            with self.subTest(token=attributes["content"]):
                self.assertEqual(
                    self.tokenizer.encode(
                        attributes["content"], add_special_tokens=False
                    ).ids,
                    [int(token_id)],
                )
        encoded = json.loads(self.tokenizer.to_str())
        self.assertEqual(
            encoded["added_tokens"],
            [{"id": int(i), **a} for i, a in self.added.items()],
        )
        self.assertEqual(
            encoded["decoder"],
            dict(
                type="ByteLevel",
                add_prefix_space=True,
                trim_offsets=True,
                use_regex=True,
            ),
        )
        self.assertEqual(
            encoded["pre_tokenizer"]["pretokenizers"][1],
            dict(
                type="ByteLevel",
                add_prefix_space=False,
                trim_offsets=True,
                use_regex=False,
            ),
        )

    def test_normalization_is_idempotent_and_preserves_vocab_and_merges(self):
        before = json.loads(self.tokenizer.to_str())["model"]
        prepare._normalize_tokenizer(self.tokenizer, self.added)
        first = self.tokenizer.to_str()
        prepare._normalize_tokenizer(self.tokenizer, self.added)
        self.assertEqual(self.tokenizer.to_str(), first)
        self.assertEqual(json.loads(first)["model"], before)

    def test_wrong_added_token_id_is_rejected(self):
        token_id, attributes = next(iter(self.added.items()))
        with self.assertRaisesRegex(ValueError, "Added token ID"):
            prepare._normalize_tokenizer(
                self.tokenizer, {str(int(token_id) + 1): attributes}
            )

    def test_actual_auto_tokenizer_loads_prepared_large_vocabulary(self):
        try:
            from transformers import AutoTokenizer
        except ImportError:
            self.skipTest("transformers is supplied by the Bazel runtime")

        # Transformers 5.2 only exercises its Mistral compatibility path above
        # 100k tokens. A tiny vocabulary would miss the duplicate-keyword bug.
        encoded = json.loads(self.original.to_str())
        encoded["model"].update(continuing_subword_prefix="", end_of_word_suffix="")
        vocab = encoded["model"]["vocab"]
        vocab.update({f"<synthetic-vocab-{i}>": i for i in range(len(vocab), 248044)})
        backend = tokenizers.Tokenizer.from_str(json.dumps(encoded))
        contents = [f"<synthetic-added-{i}>" for i in range(33)]
        contents[0] = "<|endoftext|>"
        for token, token_id in zip(
            prepare.reference.SPECIAL_TOKENS, prepare.reference.SPECIAL_TOKEN_IDS
        ):
            contents[token_id - 248044] = token
        contents[22:] = self.EXTRA_TOKENS
        added = {
            str(248044 + i): dict(
                content=content,
                single_word=False,
                lstrip=False,
                rstrip=False,
                normalized=False,
                special=i < 14 or i >= 26,
            )
            for i, content in enumerate(contents)
        }
        backend.add_tokens(
            [tokenizers.AddedToken(**a) for a in list(added.values())[:22]]
        )
        with tempfile.TemporaryDirectory() as directory:
            base, output = Path(directory) / "base", Path(directory) / "output"
            base.mkdir()
            output.mkdir()
            backend.save(str(base / "tokenizer.json"))
            backend.model.save(str(base))
            prepare._write_json(
                base / "tokenizer_config.json",
                dict(
                    tokenizer_class="Qwen2Tokenizer",
                    added_tokens_decoder=added,
                    bos_token=None,
                    eos_token="<|endoftext|>",
                    pad_token="<|endoftext|>",
                    unk_token=None,
                    add_prefix_space=False,
                    # Regression: this must not reach AutoTokenizer on 5.2.
                    fix_mistral_regex=False,
                ),
            )
            prepare._prepare_tokenizer(
                base, output, SimpleNamespace(Tokenizer=tokenizers.Tokenizer)
            )
            self.assertNotIn(
                "fix_mistral_regex", prepare._json(output / "tokenizer_config.json")
            )
            canonical = tokenizers.Tokenizer.from_file(str(output / "tokenizer.json"))
            loaded = AutoTokenizer.from_pretrained(str(output), local_files_only=True)
            self.assertGreater(loaded.backend_tokenizer.get_vocab_size(), 100000)
            actual_json = json.loads(loaded.backend_tokenizer.to_str())
            expected_json = json.loads(canonical.to_str())
            self.assertEqual(set(actual_json), set(expected_json))
            for key in expected_json:
                self.assertTrue(
                    actual_json[key] == expected_json[key],
                    f"AutoTokenizer changed backend field: {key}",
                )
            for text in (
                "कि",
                "مَرْحَبًا",
                "Cafe\u0301 中文 123",
                " \\n\\t ",
                *self.EXTRA_TOKENS,
            ):
                with self.subTest(text=text):
                    self.assertEqual(
                        loaded.encode(text, add_special_tokens=False),
                        canonical.encode(text, add_special_tokens=False).ids,
                    )


class RecognitionTest(unittest.TestCase):
    def test_repository_latest_and_arbitrary_revisions_are_recognized(self):
        repo = prepare.reference.SOURCE_REPO
        for value, revision in (
            (repo, None),
            ("https://huggingface.co/" + repo, None),
            (repo + "@main", "main"),
            (repo + "@release-2027", "release-2027"),
            ("https://huggingface.co/" + repo + "/tree/custom-commit", "custom-commit"),
        ):
            with self.subTest(value=value):
                self.assertEqual(
                    prepare._recognize(value), ("remote", (repo, revision))
                )

    def test_ordinary_models_do_not_import_runtime_dependencies(self):
        with patch.object(prepare, "_dependencies", side_effect=AssertionError):
            for value in (
                "Qwen/Qwen3.5-4B",
                "jaredpalmer/kev-27b",
                "https://example.org/jaredpalmer/kev-4b",
            ):
                self.assertIsNone(prepare.prepare_kev_checkpoint(value))
            with tempfile.TemporaryDirectory() as directory:
                self.assertIsNone(prepare.prepare_kev_checkpoint(directory))

    def test_local_raw_requires_all_source_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "head.pt").write_bytes(b"synthetic")
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                prepare._recognize(directory)
            for name in prepare.reference.SOURCE_FILES:
                (path / name).touch()
            self.assertEqual(prepare._recognize(directory), ("raw", path.resolve()))

    def test_legacy_export_keeps_original_path_and_needs_no_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "kev_config.json").write_text("{}")
            original = directory + "/."
            deps = SimpleNamespace(snapshot_download=Mock())
            with patch.object(
                prepare, "_dependencies", return_value=deps
            ), patch.object(prepare, "_validate_export") as validate:
                self.assertEqual(prepare.prepare_kev_checkpoint(original), original)
                validate.assert_called_once_with(Path(directory).resolve(), deps)
                deps.snapshot_download.assert_not_called()


class MetadataTest(unittest.TestCase):
    def test_exact_shapes_adapter_count_and_base_dtypes(self):
        shapes, adapted = prepare._expected_shapes()
        self.assertEqual(len(shapes), 426)
        self.assertEqual(len(adapted), 248)
        self.assertEqual(shapes["layers.3.self_attn.q_proj.weight"], (8192, 2560))
        self.assertEqual(shapes["layers.0.linear_attn.conv1d.weight"], (8192, 1, 4))
        dtypes = prepare._base_dtypes(shapes)
        self.assertEqual(list(dtypes.values()).count("F32"), 48)
        self.assertEqual(dtypes["layers.0.linear_attn.A_log"], "F32")
        self.assertEqual(dtypes["layers.0.linear_attn.norm.weight"], "F32")
        self.assertEqual(dtypes["layers.0.linear_attn.dt_bias"], "BF16")
        header = {
            name: {"dtype": dtypes[name], "shape": list(shape)}
            for name, shape in shapes.items()
        }
        prepare._validate_tensor_set(header, shapes, dtypes)
        header["layers.0.linear_attn.dt_bias"]["dtype"] = "F32"
        with self.assertRaises(ValueError):
            prepare._validate_tensor_set(header, shapes, dtypes)

    def test_adapter_exact_key_shape_and_dtype_contract(self):
        shapes, adapted = prepare._expected_shapes()
        config = dict(
            base_model_name_or_path=prepare.reference.BASE_MODEL,
            peft_type="LORA",
            r=16,
            lora_alpha=32,
            bias="none",
            fan_in_fan_out=False,
            use_dora=False,
            use_rslora=False,
        )
        header = {}
        for target in adapted:
            out_size, in_size = shapes[target + ".weight"]
            header[prepare.ADAPTER_PREFIX + target + ".lora_A.weight"] = {
                "dtype": "F32",
                "shape": [16, in_size],
            }
            header[prepare.ADAPTER_PREFIX + target + ".lora_B.weight"] = {
                "dtype": "F32",
                "shape": [out_size, 16],
            }
        with tempfile.TemporaryDirectory() as directory, patch.object(
            prepare, "_tensor_header", return_value=header
        ):
            source = Path(directory)
            prepare._write_json(source / "adapter_config.json", config)
            self.assertEqual(
                len(prepare._validate_adapter(source, shapes, adapted)), 496
            )
            for field, value in (
                ("rank_pattern", {"x": 8}),
                ("lora_bias", True),
                ("trainable_token_indices", [5]),
            ):
                prepare._write_json(
                    source / "adapter_config.json", {**config, field: value}
                )
                with self.assertRaisesRegex(ValueError, field):
                    prepare._validate_adapter(source, shapes, adapted)
            prepare._write_json(source / "adapter_config.json", config)
            key = next(iter(header))
            header[key]["dtype"] = "BF16"
            with self.assertRaises(ValueError):
                prepare._validate_adapter(source, shapes, adapted)
            header[key]["dtype"] = "F32"
            del header[key]
            with self.assertRaises(ValueError):
                prepare._validate_adapter(source, shapes, adapted)

    def test_text_config_preserves_nested_and_root_rope(self):
        text = dict(
            model_type="qwen3_5_text",
            hidden_size=2560,
            intermediate_size=9216,
            num_hidden_layers=32,
            vocab_size=248320,
            num_attention_heads=16,
            num_key_value_heads=4,
            head_dim=256,
            tie_word_embeddings=True,
            rms_norm_eps=1e-6,
            layer_types=(["linear_attention"] * 3 + ["full_attention"]) * 8,
            linear_conv_kernel_dim=4,
            linear_key_head_dim=128,
            linear_num_key_heads=16,
            linear_num_value_heads=32,
            linear_value_head_dim=128,
            attn_output_gate=True,
            attention_bias=False,
            rope_parameters={
                "partial_rotary_factor": 0.25,
                "rope_theta": 10000000,
                "mrope_section": [11, 11, 10],
            },
        )
        with patch.object(
            prepare, "_json", return_value={"text_config": text, "vision_config": {}}
        ):
            result = prepare._text_config(Path("synthetic"))
        self.assertNotIn("vision_config", result)
        self.assertNotIn("text_config", result)
        self.assertEqual(result["partial_rotary_factor"], 0.25)
        self.assertEqual(result["rope_parameters"], text["rope_parameters"])
        self.assertEqual(result["architectures"], ["Qwen3_5ForCausalLM"])
        self.assertIsNone(result["bos_token_id"])
        self.assertIsNone(result["pad_token_id"])
        self.assertFalse(result["use_cache"])

    def test_export_manifest_accepts_dynamic_hashes_and_nullable_provenance(self):
        shapes, _ = prepare._expected_shapes()
        header = {
            prepare.TEXT_PREFIX + k: {"dtype": "BF16", "shape": list(v)}
            for k, v in shapes.items()
        }
        head_header = {
            k: {"dtype": "F32", "shape": list(v)}
            for k, v in prepare.reference.HEAD_SHAPES.items()
        }
        manifest = dict(
            format_version=1,
            head_dim=256,
            hidden_size=2560,
            special_token_ids=prepare.reference.SPECIAL_TOKEN_IDS,
            base_model=prepare.reference.BASE_MODEL,
            base_revision=None,
            source_revision=None,
            source_repo=None,
            source_sha256={
                name: hashlib.sha256(name.encode()).hexdigest()
                for name in prepare.reference.SOURCE_FILES
            },
            backbone_dtype="bfloat16",
            head_dtype="float32",
            add_special_tokens=False,
            temperature=1.0,
            max_options=32,
            max_state_tokens=65536,
            max_row_tokens=73728,
        )
        with tempfile.TemporaryDirectory() as directory, patch.object(
            prepare, "_text_config"
        ), patch.object(prepare, "_inventory", return_value=(header, {})), patch.object(
            prepare, "_tensor_header", return_value=head_header
        ), patch.object(
            prepare, "_tokenizer_ids"
        ):
            path = Path(directory)
            (path / "tokenizer_config.json").write_text("{}")
            for revision in (None, "any-compatible-release"):
                manifest["base_revision"] = revision
                manifest["source_revision"] = revision
                prepare._write_json(path / "kev_config.json", manifest)
                prepare._validate_export(path, SimpleNamespace())
            manifest["source_sha256"]["head.pt"] = "bad-digest"
            prepare._write_json(path / "kev_config.json", manifest)
            with self.assertRaisesRegex(ValueError, "fingerprints"):
                prepare._validate_export(path, SimpleNamespace())

    def test_tensor_header_validates_payload_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.safetensors"
            header = {"x": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}}
            encoded = json.dumps(header).encode()
            content = struct.pack("<Q", len(encoded)) + encoded + b"\0" * 4
            path.write_bytes(content)
            self.assertEqual(prepare._tensor_header(path), header)
            for invalid in (content[:-1], content + b"\0", content[:4]):
                path.write_bytes(invalid)
                with self.assertRaises(ValueError):
                    prepare._tensor_header(path)

    def test_tokenizer_delimiters_are_exact(self):
        tokenizer = Mock()
        tokenizer.token_to_id.side_effect = prepare.reference.SPECIAL_TOKEN_IDS
        deps = SimpleNamespace(Tokenizer=Mock())
        deps.Tokenizer.from_file.return_value = tokenizer
        self.assertEqual(
            prepare._tokenizer_ids(Path("synthetic"), deps),
            prepare.reference.SPECIAL_TOKEN_IDS,
        )
        tokenizer.token_to_id.side_effect = [
            *prepare.reference.SPECIAL_TOKEN_IDS[:4],
            0,
        ]
        with self.assertRaisesRegex(ValueError, "delimiter"):
            prepare._tokenizer_ids(Path("synthetic"), deps)


@unittest.skipIf(torch is None, "CPU torch is supplied by the remote test environment")
class MergeTest(unittest.TestCase):
    def test_fp32_merge_matches_reference_before_single_bf16_cast(self):
        generator = torch.Generator().manual_seed(81)
        weight = torch.randn(7, 5, generator=generator).bfloat16()
        a = torch.randn(16, 5, generator=generator)
        b = torch.randn(7, 16, generator=generator)
        result = prepare._merge_tensor(weight, a, b, torch)
        expected = (weight.float() + (b @ a) * 2).bfloat16()
        self.assertTrue(torch.equal(result, expected))
        self.assertEqual(result.dtype, torch.bfloat16)
        early_cast = (weight + (b.bfloat16() @ a.bfloat16()) * 2).bfloat16()
        self.assertFalse(torch.equal(result, early_cast))

    def test_invalid_operands_and_nonfinite_merge_are_rejected(self):
        weight, a, b = torch.ones(3, 2), torch.ones(16, 2), torch.ones(3, 16)
        for bad_a in (a.bfloat16(), a[:15], a * float("nan"), a * float("inf")):
            with self.subTest(shape=bad_a.shape, dtype=bad_a.dtype), self.assertRaises(
                ValueError
            ):
                prepare._merge_tensor(weight, bad_a, b, torch)


class CacheTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = (
            self.root / "models--jaredpalmer--kev-4b" / "snapshots" / ("a" * 40)
        )
        self.source.mkdir(parents=True)
        for name in prepare.reference.SOURCE_FILES:
            (self.source / name).write_bytes(("synthetic-" + name).encode())
        self.header = dict(
            base=prepare.reference.BASE_MODEL,
            base_revision="base-release",
            lora=16,
            head={},
        )
        prepare._write_json(self.source / "head.pt", self.header)
        self.base = (
            self.root / "models--Qwen--Qwen3.5-4B-Base" / "snapshots" / ("b" * 40)
        )
        self.base.mkdir(parents=True)
        (self.base / "config.json").write_text("{}")
        self.cache = self.root / "cache"
        self.download = Mock(
            side_effect=lambda repo, **kwargs: str(
                self.source if repo == prepare.reference.SOURCE_REPO else self.base
            )
        )
        self.mutex = threading.Lock()
        mutex = self.mutex

        class Lock:
            def __init__(self, path):
                self.path = path

            def __enter__(self):
                mutex.acquire()
                return self

            def __exit__(self, *args):
                mutex.release()

        self.deps = SimpleNamespace(
            FileLock=Lock,
            snapshot_download=self.download,
            torch=SimpleNamespace(
                load=lambda stream, **kwargs: json.loads(stream.read())
            ),
        )
        self.convert = Mock(side_effect=self.write_output)
        self.head_validator = prepare.reference._validate_head
        for patcher in (
            patch.object(prepare, "_dependencies", return_value=self.deps),
            patch.object(prepare, "_convert", self.convert),
            patch.object(prepare, "_validate_export", side_effect=self.validate_output),
            patch.object(prepare.reference, "_validate_head"),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    @staticmethod
    def write_output(source, base, output, deps, override, **kwargs):
        manifest = prepare.reference._source_manifest(
            source,
            kwargs["source_hashes"],
            kwargs["metadata"],
            resolved_base_revision=kwargs["resolved_base_revision"],
        )
        manifest["base_is_local_override"] = override
        manifest["temperature"] = kwargs["metadata"]["temperature"]
        prepare._write_json(output / "kev_config.json", manifest)
        (output / "model.safetensors").write_bytes(b"synthetic-output")

    @staticmethod
    def validate_output(path, deps):
        if (
            not (path / "kev_config.json").is_file()
            or not (path / "model.safetensors").is_file()
        ):
            raise ValueError("Incomplete synthetic export")

    def run_prepare(self, checkpoint=None, **kwargs):
        return prepare.prepare_kev_checkpoint(
            checkpoint or "https://huggingface.co/" + prepare.reference.SOURCE_REPO,
            cache_root=self.cache,
            **kwargs,
        )

    def assert_no_weight_download(self):
        self.assertFalse(
            any(
                "model*.safetensors" in call.kwargs["allow_patterns"]
                for call in self.download.call_args_list
            )
        )

    def test_latest_source_and_declared_base_ref_resolve_before_weight_download(self):
        result = Path(self.run_prepare())
        self.assertTrue((result / prepare.INTEGRITY_FILE).is_file())
        calls = self.download.call_args_list
        self.assertEqual(
            [call.args[0] for call in calls],
            [
                prepare.reference.SOURCE_REPO,
                prepare.reference.BASE_MODEL,
                prepare.reference.BASE_MODEL,
            ],
        )
        self.assertEqual(
            [call.kwargs["revision"] for call in calls],
            [None, "base-release", "b" * 40],
        )
        self.assertEqual(
            calls[0].kwargs["allow_patterns"], list(prepare.reference.SOURCE_FILES)
        )
        self.assertEqual(calls[1].kwargs["allow_patterns"], ["config.json"])
        manifest = prepare._json(result / "kev_config.json")
        self.assertEqual(manifest["base_revision"], "b" * 40)
        self.assertEqual(manifest["source_revision"], "a" * 40)
        self.assertEqual(
            manifest["source_sha256"], prepare.reference._read_source(self.source)[1]
        )
        self.download.reset_mock()
        self.convert.reset_mock()
        self.assertEqual(self.run_prepare(), str(result))
        self.assertEqual(self.download.call_count, 2)
        self.assert_no_weight_download()
        self.convert.assert_not_called()

    def test_arbitrary_source_ref_is_forwarded_and_actual_sha_is_recorded(self):
        result = Path(
            self.run_prepare(prepare.reference.SOURCE_REPO + "@release-canary")
        )
        self.assertEqual(
            self.download.call_args_list[0].kwargs["revision"], "release-canary"
        )
        self.assertEqual(
            prepare._json(result / "kev_config.json")["source_revision"], "a" * 40
        )

    def test_legacy_missing_revision_and_options_use_official_defaults(self):
        del self.header["base_revision"]
        prepare._write_json(self.source / "head.pt", self.header)
        result = Path(self.run_prepare())
        self.assertIsNone(self.download.call_args_list[1].kwargs["revision"])
        metadata = self.convert.call_args.kwargs["metadata"]
        self.assertEqual(metadata["temperature"], 1.0)
        self.assertEqual(metadata["weights_dtype"], "fp32")
        self.assertEqual(metadata["lora_placement"], "full")
        self.assertFalse(metadata["option_isolation"])
        self.assertEqual(
            prepare._json(result / "kev_config.json")["base_revision"], "b" * 40
        )

    def test_local_source_and_base_unknown_provenance_stays_null(self):
        local = self.root / "whale-model"
        local.mkdir()
        for name in prepare.reference.SOURCE_FILES:
            (local / name).write_bytes((self.source / name).read_bytes())
        header = dict(self.header)
        del header["base_revision"]
        prepare._write_json(local / "head.pt", header)
        result = Path(self.run_prepare(str(local), base=self.base))
        manifest = prepare._json(result / "kev_config.json")
        self.assertIsNone(manifest["source_repo"])
        self.assertIsNone(manifest["source_revision"])
        self.assertIsNone(manifest["base_revision"])
        self.assertTrue(manifest["base_is_local_override"])
        self.download.assert_not_called()

    def test_local_base_records_declared_revision_and_uses_content_identity(self):
        file = self.base / "weights.safetensors"
        file.write_bytes(b"first")
        result = Path(self.run_prepare(base=self.base))
        self.assertEqual(
            prepare._json(result / "kev_config.json")["base_revision"], "base-release"
        )
        stat = file.stat()
        file.write_bytes(b"other")
        os.utime(file, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        changed = Path(self.run_prepare(base=self.base))
        self.assertNotEqual(result, changed)
        self.assertEqual(self.convert.call_count, 2)

    def test_changed_source_bytes_at_same_path_create_new_cache(self):
        first = Path(self.run_prepare())
        file = self.source / "adapter_model.safetensors"
        stat = file.stat()
        data = file.read_bytes()
        file.write_bytes(bytes([data[0] ^ 1]) + data[1:])
        os.utime(file, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        second = Path(self.run_prepare())
        self.assertNotEqual(first, second)
        self.assertEqual(self.convert.call_count, 2)
        self.assertNotEqual(
            prepare._json(first / "kev_config.json")["source_sha256"],
            prepare._json(second / "kev_config.json")["source_sha256"],
        )

    def test_moving_base_ref_creates_a_distinct_cache(self):
        original = self.run_prepare()
        self.base = self.base.parent / ("c" * 40)
        self.base.mkdir()
        (self.base / "config.json").write_text("{}")
        self.assertNotEqual(self.run_prepare(), original)
        self.assertEqual(self.download.call_args.kwargs["revision"], "c" * 40)

    def test_base_snapshot_cannot_change_between_config_and_weights(self):
        other = self.base.parent / ("d" * 40)
        other.mkdir()
        (other / "config.json").write_text("{}")
        self.download.side_effect = lambda repo, **kwargs: str(
            self.source
            if repo == prepare.reference.SOURCE_REPO
            else self.base if kwargs["allow_patterns"] == ["config.json"] else other
        )
        with self.assertRaisesRegex(ValueError, "Base snapshot changed"):
            self.run_prepare()
        self.convert.assert_not_called()
        self.assertEqual(list(self.cache.iterdir()), [])

    def test_nonstandard_provider_fingerprints_complete_base_before_caching(self):
        self.base = self.root / "nonstandard-provider"
        self.base.mkdir()
        (self.base / "config.json").write_text("{}")
        data = [b"weights-a"]

        def download(repo, **kwargs):
            if repo == prepare.reference.SOURCE_REPO:
                return str(self.source)
            if "model*.safetensors" in kwargs["allow_patterns"]:
                (self.base / "model.safetensors").write_bytes(data[0])
            return str(self.base)

        self.download.side_effect = download
        first = self.run_prepare()
        self.assertEqual(self.run_prepare(), first)
        self.assertEqual(self.convert.call_count, 1)
        data[0] = b"weights-b"
        second = self.run_prepare()
        self.assertNotEqual(first, second)
        self.assertEqual(self.convert.call_count, 2)
        self.assertEqual(
            prepare._json(Path(second) / "kev_config.json")["base_revision"],
            "base-release",
        )

    def test_source_changed_during_conversion_is_not_published(self):
        def mutate(*args, **kwargs):
            self.write_output(*args, **kwargs)
            (self.source / "adapter_model.safetensors").write_bytes(b"new-weights")

        self.convert.side_effect = mutate
        with self.assertRaisesRegex(ValueError, "source changed"):
            self.run_prepare()
        self.assertEqual(list(self.cache.iterdir()), [])
        self.convert.side_effect = self.write_output
        self.assertTrue(Path(self.run_prepare()).is_dir())

    def test_base_changed_during_conversion_is_not_published(self):
        def mutate(*args, **kwargs):
            self.write_output(*args, **kwargs)
            (self.base / "config.json").write_text('{"changed":true}')

        self.convert.side_effect = mutate
        with self.assertRaisesRegex(ValueError, "Base content changed"):
            self.run_prepare(base=self.base)
        self.assertEqual(list(self.cache.iterdir()), [])

    def test_corrupt_head_is_rejected_before_resolving_base(self):
        (self.source / "head.pt").write_bytes(b"not-a-valid-head")
        with self.assertRaises(ValueError):
            self.run_prepare()
        self.assertEqual(self.download.call_count, 1)
        self.convert.assert_not_called()
        self.assertFalse(self.cache.exists())

    def test_unsupported_training_modes_are_rejected_before_resolving_base(self):
        for field, value in (("weights_dtype", "bf16"), ("lora_placement", "question")):
            with self.subTest(field=field), patch.object(
                prepare.reference, "_validate_head", self.head_validator
            ):
                prepare._write_json(
                    self.source / "head.pt", {**self.header, field: value}
                )
                self.download.reset_mock()
                with self.assertRaisesRegex(ValueError, field):
                    self.run_prepare()
                self.assertEqual(self.download.call_count, 1)
                self.convert.assert_not_called()

    def test_modified_payload_is_rejected_even_when_size_is_unchanged(self):
        result = Path(self.run_prepare())
        (result / "model.safetensors").write_bytes(b"Xynthetic-output")
        self.download.reset_mock()
        with self.assertRaisesRegex(ValueError, "Invalid Kev cache"):
            self.run_prepare()
        self.assert_no_weight_download()

    def test_cache_rejects_missing_manifest_file_and_symlink(self):
        result = Path(self.run_prepare())
        payload = result / "model.safetensors"
        data = payload.read_bytes()
        payload.unlink()
        self.assertFalse(prepare._valid_cache(result, result.name, self.deps))
        target = self.root / "payload"
        target.write_bytes(data)
        payload.symlink_to(target)
        self.assertFalse(prepare._valid_cache(result, result.name, self.deps))

    def test_failed_conversion_cleans_private_output_and_can_retry(self):
        def fail(source, base, output, deps, override, **kwargs):
            (output / "partial").write_bytes(b"partial")
            raise RuntimeError("synthetic conversion failure")

        self.convert.side_effect = fail
        with self.assertRaisesRegex(RuntimeError, "synthetic"):
            self.run_prepare()
        self.assertEqual(list(self.cache.iterdir()), [])
        self.convert.side_effect = self.write_output
        self.assertTrue(Path(self.run_prepare()).is_dir())

    def test_failed_publish_cleans_private_output_and_can_retry(self):
        with patch.object(
            prepare.os, "replace", side_effect=OSError("synthetic publish failure")
        ):
            with self.assertRaisesRegex(OSError, "publish"):
                self.run_prepare()
        self.assertEqual(list(self.cache.iterdir()), [])
        self.assertTrue(Path(self.run_prepare()).is_dir())

    def test_failed_lock_does_not_convert_or_download_weights_and_can_retry(self):
        lock = self.deps.FileLock
        self.deps.FileLock = Mock(side_effect=OSError("synthetic lock failure"))
        with self.assertRaisesRegex(OSError, "lock"):
            self.run_prepare()
        self.assert_no_weight_download()
        self.convert.assert_not_called()
        self.deps.FileLock = lock
        self.assertTrue(Path(self.run_prepare()).is_dir())

    def test_preparer_version_changes_cache_identity(self):
        hashes = prepare.reference._read_source(self.source)[1]
        original = prepare._cache_key(hashes, {})
        with patch.object(prepare, "PREPARER_VERSION", prepare.PREPARER_VERSION + 1):
            self.assertNotEqual(prepare._cache_key(hashes, {}), original)

    def test_two_workers_publish_only_one_conversion(self):
        entered, release = threading.Event(), threading.Event()
        results, errors = [], []

        def convert(*args, **kwargs):
            entered.set()
            if not release.wait(timeout=5):
                raise RuntimeError("test synchronization timed out")
            self.write_output(*args, **kwargs)

        def worker():
            try:
                results.append(self.run_prepare())
            except Exception as error:
                errors.append(error)

        self.convert.side_effect = convert
        first, second = threading.Thread(target=worker), threading.Thread(target=worker)
        first.start()
        try:
            self.assertTrue(entered.wait(timeout=5))
            second.start()
        finally:
            release.set()
            first.join(timeout=5)
            if second.ident is not None:
                second.join(timeout=5)
        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0], results[1])
        self.assertEqual(self.convert.call_count, 1)
        self.assertEqual(self.download.call_count, 5)

    def test_cache_root_environment_precedence(self):
        with patch.dict(prepare.os.environ, {"HF_HOME": "/synthetic/hf"}, clear=True):
            self.assertEqual(
                prepare._cache_root(None), Path("/synthetic/hf/hub/rtp-llm-kev")
            )
            with patch.dict(
                prepare.os.environ,
                {"HF_HUB_CACHE": "/synthetic/hub", "KEV_CACHE_DIR": "/synthetic/kev"},
            ):
                self.assertEqual(prepare._cache_root(None), Path("/synthetic/kev"))
                self.assertEqual(prepare._cache_root("/explicit"), Path("/explicit"))
            with patch.dict(prepare.os.environ, {"HF_HUB_CACHE": "/synthetic/hub"}):
                self.assertEqual(
                    prepare._cache_root(None), Path("/synthetic/hub/rtp-llm-kev")
                )


if __name__ == "__main__":
    unittest.main()
