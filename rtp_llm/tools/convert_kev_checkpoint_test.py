"""CPU-only metadata and mocked export tests; no model downloads or RTP ops."""

import contextlib
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from rtp_llm.tools import convert_kev_checkpoint as converter


class Tensor:
    def __init__(self, shape, dtype="fp32", finite=True):
        self.shape, self.dtype, self.finite = shape, dtype, finite

    def detach(self):
        return self

    def cpu(self):
        return self

    def float(self):
        self.dtype = "fp32"
        return self

    def contiguous(self):
        return self


def _meta():
    return {
        "base": converter.BASE_MODEL,
        "base_revision": "base-release",
        "lora": 16,
        "head_dim": 256,
        "option_isolation": False,
        "special_embeddings": False,
        "weights_dtype": "fp32",
        "lora_placement": "full",
        "weights": "lora",
        "temperature": 2.406050072164233,
        "head": {k: Tensor(v) for k, v in converter.HEAD_SHAPES.items()},
    }


def _torch(meta):
    return SimpleNamespace(
        Tensor=Tensor,
        float32="fp32",
        bfloat16="bf16",
        isfinite=lambda tensor: SimpleNamespace(
            all=lambda: SimpleNamespace(item=lambda: tensor.finite)
        ),
        load=mock.Mock(return_value=meta),
        no_grad=contextlib.nullcontext,
    )


class HeadValidationTest(unittest.TestCase):
    def test_supported_contract_accepts_arbitrary_base_revisions(self):
        converter._validate_head(_meta(), _torch(None))
        for revision in (None, "main", "older-compatible-release", "1" * 40):
            with self.subTest(revision=revision):
                meta = _meta()
                meta["base_revision"] = revision
                converter._validate_head(meta, _torch(None))

    def test_legacy_defaults_preserve_training_metadata_and_input(self):
        original = {
            "base": converter.BASE_MODEL,
            "lora": 16,
            "head": _meta()["head"],
            "future_training_details": {"experiment": "test"},
        }
        with self.assertLogs(converter.logger, level="INFO") as logs:
            meta = converter._normalize_head(original)
        converter._validate_head(meta, _torch(None))
        self.assertEqual(meta["future_training_details"], {"experiment": "test"})
        self.assertIs(meta["head"], original["head"])
        self.assertNotIn("weights_dtype", original)
        self.assertEqual(
            {name: meta[name] for name in converter.HEAD_DEFAULTS},
            converter.HEAD_DEFAULTS,
        )
        self.assertIn("weights_dtype", logs.output[0])
        self.assertIn("lora_placement", logs.output[0])

    def test_old_release_missing_only_dtype_and_placement_is_compatible(self):
        expected = _meta()
        old = dict(expected)
        del old["weights_dtype"]
        del old["lora_placement"]
        normalized = converter._normalize_head(old)
        converter._validate_head(normalized, _torch(None))
        self.assertEqual(normalized, expected)

    def test_default_normalization_requires_metadata_object(self):
        for value in (None, [], "metadata"):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "dictionary"
            ):
                converter._normalize_head(value)

    def test_unsupported_metadata_and_nonfinite_temperatures(self):
        invalid = {
            "base": ["Qwen/Qwen3.5-4B"],
            "base_revision": ["", " ", 17, True],
            "lora": [0, 8, True],
            "weights": ["full"],
            "weights_dtype": ["bf16"],
            "head_dim": [128],
            "option_isolation": [True, 0],
            "special_embeddings": [True],
            "lora_placement": ["dense", "question"],
            "temperature": [float("nan"), float("inf"), -1, 0, True, "2.4"],
        }
        for key, values in invalid.items():
            for value in values:
                with self.subTest(key=key, value=value):
                    meta = _meta()
                    meta[key] = value
                    with self.assertRaises(ValueError):
                        converter._validate_head(
                            converter._normalize_head(meta), _torch(None)
                        )
        meta = _meta()
        del meta["lora"]
        with self.assertRaisesRegex(ValueError, "lora"):
            converter._validate_head(converter._normalize_head(meta), _torch(None))

    def test_bad_shapes_dtypes_keys_and_nonfinite_head(self):
        for key, shape in converter.HEAD_SHAPES.items():
            for tensor in (
                Tensor((1,)),
                Tensor(shape, "bf16"),
                Tensor(shape, finite=False),
                None,
            ):
                with self.subTest(key=key, tensor=tensor):
                    meta = _meta()
                    meta["head"][key] = tensor
                    with self.assertRaises(ValueError):
                        converter._validate_head(meta, _torch(None))
        meta = _meta()
        del meta["head"]["q.bias"]
        with self.assertRaises(ValueError):
            converter._validate_head(meta, _torch(None))


class SourceIdentityTest(unittest.TestCase):
    def test_hub_sources_accept_unpinned_and_arbitrary_revisions(self):
        for source, revision in (
            (converter.SOURCE_REPO, None),
            (converter.SOURCE_REPO + "@main", "main"),
            (converter.SOURCE_REPO + "@older-release", "older-release"),
            (converter.SOURCE_REPO + "@refs/pr/3", "refs/pr/3"),
            ("https://huggingface.co/" + converter.SOURCE_REPO, None),
            (
                "https://huggingface.co/" + converter.SOURCE_REPO + "/tree/" + "c" * 40,
                "c" * 40,
            ),
            (
                "https://huggingface.co/"
                + converter.SOURCE_REPO
                + "/tree/refs%2Fpr%2F3",
                "refs/pr/3",
            ),
        ):
            with self.subTest(source=source):
                self.assertEqual(
                    converter._parse_source(source), (converter.SOURCE_REPO, revision)
                )

    def test_other_models_are_ignored_and_malformed_kev_sources_fail(self):
        for source in (
            "other/model",
            "https://huggingface.co/other/model",
            "nas://models/kev",
        ):
            self.assertIsNone(converter._parse_source(source))
        for source in (
            converter.SOURCE_REPO + "@",
            converter.SOURCE_REPO + "@main@other",
            converter.SOURCE_REPO + "@ ",
            "https://huggingface.co/" + converter.SOURCE_REPO + "?revision=main",
            "https://huggingface.co/" + converter.SOURCE_REPO + "/tree/main#other",
            "https://huggingface.co/" + converter.SOURCE_REPO + "/tree/",
            "https://huggingface.co/" + converter.SOURCE_REPO + "/resolve/main/head.pt",
        ):
            with self.subTest(source=source), self.assertRaises(ValueError):
                converter._parse_source(source)

    def test_only_hf_snapshot_paths_supply_source_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = root / "models--jaredpalmer--kev-4b" / "snapshots" / ("a" * 40)
            local = root / "snapshots" / ("b" * 40)
            snapshot.mkdir(parents=True)
            local.mkdir(parents=True)
            hashes = dict.fromkeys(converter.SOURCE_FILES, "c" * 64)
            hub = converter._source_manifest(snapshot, hashes, _meta(), "d" * 40)
            self.assertEqual(hub["source_revision"], "a" * 40)
            self.assertEqual(hub["source_repo"], converter.SOURCE_REPO)
            self.assertEqual(hub["base_revision"], "d" * 40)
            self.assertEqual(hub["source_sha256"], hashes)
            own = converter._source_manifest(local, hashes, _meta())
            self.assertIsNone(own["source_revision"])
            self.assertIsNone(own["source_repo"])
            self.assertEqual(own["base_revision"], "base-release")
            self.assertIsNone(
                converter._source_revision(
                    root / "models--owner--repo" / "snapshots" / "main"
                )
            )


class ExportTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.output = self.root / "output"
        self.hub_source = (
            self.root / "models--jaredpalmer--kev-4b" / "snapshots" / ("a" * 40)
        )
        self.hub_source.mkdir(parents=True)
        self.base = (
            self.root / "models--Qwen--Qwen3.5-4B-Base" / "snapshots" / ("b" * 40)
        )
        self.base.mkdir(parents=True)
        self.source_hashes = {}
        for name in converter.SOURCE_FILES:
            data = f"test fixture {name}".encode()
            if name == "adapter_config.json":
                data = json.dumps(
                    {
                        "base_model_name_or_path": converter.BASE_MODEL,
                        "peft_type": "LORA",
                        "r": 16,
                        "lora_alpha": 32,
                    }
                ).encode()
            (self.source / name).write_bytes(data)
            (self.hub_source / name).write_bytes(data)
            self.source_hashes[name] = hashlib.sha256(data).hexdigest()
        self.events = []
        self.meta = _meta()
        self.torch = _torch(self.meta)
        self.model = mock.Mock()
        self.model.config = SimpleNamespace(
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
        )
        self.model.to.side_effect = lambda dtype: self.events.append(("cast", dtype))
        self.model.save_pretrained.side_effect = self._save_model
        self.backbone = mock.Mock()
        self.model.model = self.backbone
        parameters = {}
        self.source_tensors = {}
        for layer in range(32):
            shapes = {
                "mlp.gate_proj": (9216, 2560),
                "mlp.up_proj": (9216, 2560),
                "mlp.down_proj": (2560, 9216),
            }
            if layer % 4 == 3:
                shapes.update(
                    {
                        "self_attn.q_proj": (8192, 2560),
                        "self_attn.k_proj": (1024, 2560),
                        "self_attn.v_proj": (1024, 2560),
                        "self_attn.o_proj": (2560, 4096),
                    }
                )
            else:
                shapes.update(
                    {
                        "linear_attn.in_proj_a": (32, 2560),
                        "linear_attn.in_proj_b": (32, 2560),
                        "linear_attn.in_proj_qkv": (8192, 2560),
                        "linear_attn.in_proj_z": (4096, 2560),
                        "linear_attn.out_proj": (2560, 4096),
                    }
                )
            for projection, shape in shapes.items():
                target = f"layers.{layer}.{projection}"
                parameters[target + ".weight"] = Tensor(shape)
                prefix = f"base_model.model.{target}"
                self.source_tensors[prefix + ".lora_A.weight"] = Tensor((16, shape[1]))
                self.source_tensors[prefix + ".lora_B.weight"] = Tensor((shape[0], 16))
        self.backbone.named_parameters.return_value = list(parameters.items())
        adapter_file = SimpleNamespace(
            keys=lambda: list(self.source_tensors),
            get_tensor=lambda name: self.source_tensors[name],
        )
        self.adapted = mock.Mock()
        self.adapted.named_parameters.return_value = [
            (
                name.replace(".lora_A.", ".lora_A.default.").replace(
                    ".lora_B.", ".lora_B.default."
                ),
                Tensor(tensor.shape),
            )
            for name, tensor in self.source_tensors.items()
        ]
        self.adapted.merge_and_unload.side_effect = self._merge
        self.tokenizer = mock.Mock()
        self.tokenizer.convert_tokens_to_ids.side_effect = dict(
            zip(converter.SPECIAL_TOKENS, converter.SPECIAL_TOKEN_IDS)
        ).get
        self.tokenizer.save_pretrained.side_effect = self._save_tokenizer
        self.deps = SimpleNamespace(
            torch=self.torch,
            snapshot_download=mock.Mock(
                side_effect=lambda repo, **kwargs: str(
                    {
                        converter.SOURCE_REPO: self.hub_source,
                        converter.BASE_MODEL: self.base,
                    }[repo]
                )
            ),
            PeftModel=SimpleNamespace(
                from_pretrained=mock.Mock(return_value=self.adapted)
            ),
            safe_open=mock.Mock(
                side_effect=lambda *args, **kwargs: contextlib.nullcontext(adapter_file)
            ),
            AutoModelForCausalLM=SimpleNamespace(
                from_pretrained=mock.Mock(return_value=self.model)
            ),
            AutoTokenizer=SimpleNamespace(
                from_pretrained=mock.Mock(return_value=self.tokenizer)
            ),
            save_file=mock.Mock(side_effect=self._save_head),
        )
        patch = mock.patch.object(converter, "_dependencies", return_value=self.deps)
        patch.start()
        self.addCleanup(patch.stop)

    def _merge(self, safe_merge):
        self.assertTrue(safe_merge)
        self.events.append(("merge", "fp32"))
        return self.backbone

    def _save_model(self, output, **kwargs):
        self.events.append(("save_model", kwargs))
        (output / "config.json").write_text("{}")
        (output / "model.safetensors").write_bytes(b"mock model")
        self.assertFalse((output / "kev_config.json").exists())

    def _save_tokenizer(self, output):
        (output / "tokenizer.json").write_text("{}")
        self.assertFalse((output / "kev_config.json").exists())

    def _save_head(self, head, filename, metadata):
        self.assertEqual({t.dtype for t in head.values()}, {"fp32"})
        self.assertEqual(set(head), set(converter.HEAD_SHAPES))
        Path(filename).write_bytes(b"mock head")
        self.assertFalse((Path(filename).parent / "kev_config.json").exists())

    def test_export_merges_before_cast_and_publishes_manifest_last(self):
        result = converter.convert(converter.SOURCE_REPO, self.output)
        self.assertEqual(
            self.deps.snapshot_download.call_args_list,
            [
                mock.call(
                    converter.SOURCE_REPO,
                    revision=None,
                    allow_patterns=list(converter.SOURCE_FILES),
                ),
                mock.call(converter.BASE_MODEL, revision="base-release"),
            ],
        )
        self.assertEqual(self.events[:2], [("merge", "fp32"), ("cast", "bf16")])
        self.deps.safe_open.assert_called_once_with(
            str(self.hub_source / "adapter_model.safetensors"),
            framework="pt",
            device="cpu",
        )
        load = self.deps.AutoModelForCausalLM.from_pretrained.call_args
        self.assertEqual(load.kwargs["dtype"], "fp32")
        self.assertTrue(load.kwargs["local_files_only"])
        self.assertNotIn("revision", load.kwargs)
        self.assertEqual(load.args, (str(self.base.resolve()),))
        self.deps.AutoTokenizer.from_pretrained.assert_called_once_with(
            str(self.base.resolve()), local_files_only=True
        )
        self.assertTrue(self.torch.load.call_args.kwargs["weights_only"])
        self.assertEqual(self.torch.load.call_args.kwargs["map_location"], "cpu")
        saved = json.loads((self.output / "kev_config.json").read_text())
        self.assertEqual(saved, result)
        self.assertEqual(saved["special_token_ids"], converter.SPECIAL_TOKEN_IDS)
        self.assertEqual(saved["source_revision"], "a" * 40)
        self.assertEqual(saved["source_repo"], converter.SOURCE_REPO)
        self.assertEqual(saved["source_sha256"], self.source_hashes)
        self.assertEqual(saved["base_revision"], "b" * 40)
        self.assertEqual(saved["temperature"], self.meta["temperature"])
        self.assertFalse(saved["add_special_tokens"])
        self.assertEqual(saved["max_options"], 32)
        self.assertEqual(saved["max_state_tokens"], 65536)
        self.assertEqual(saved["max_row_tokens"], 73728)
        self.assertFalse((self.output / "kev_config.json.partial").exists())

    def test_local_source_and_local_base_never_resolve_remote(self):
        base = self.root / "base"
        base.mkdir()
        result = converter.convert(str(self.source), self.output, str(base))
        self.deps.snapshot_download.assert_not_called()
        for loader in (self.deps.AutoModelForCausalLM, self.deps.AutoTokenizer):
            self.assertTrue(loader.from_pretrained.call_args.kwargs["local_files_only"])
            self.assertNotIn("revision", loader.from_pretrained.call_args.kwargs)
        self.assertEqual(result["base_source"], str(base.resolve()))
        self.assertTrue(result["base_is_local_override"])
        self.assertEqual(result["base_revision"], "base-release")
        self.assertIsNone(result["source_revision"])
        self.assertIsNone(result["source_repo"])

    def test_changed_source_files_are_accepted_and_actual_hashes_are_recorded(self):
        for name in converter.SOURCE_FILES:
            with self.subTest(name=name):
                path = self.source / name
                original = path.read_bytes()
                changed = original + b"\n"
                path.write_bytes(changed)
                result = converter.convert(str(self.source), self.root / name)
                self.assertEqual(
                    result["source_sha256"][name], hashlib.sha256(changed).hexdigest()
                )
                self.assertNotEqual(
                    result["source_sha256"][name], self.source_hashes[name]
                )
                path.write_bytes(original)

    def test_missing_empty_and_lfs_source_files_fail_before_pickle_or_model_load(self):
        for name in converter.SOURCE_FILES:
            for content in (None, b"", b"version https://git-lfs.github.com/spec/v1\n"):
                with self.subTest(name=name, content=content):
                    path = self.source / name
                    original = path.read_bytes()
                    if content is None:
                        path.unlink()
                    else:
                        path.write_bytes(content)
                    with self.assertRaisesRegex(ValueError, "Incomplete|LFS"):
                        converter.convert(str(self.source), self.output)
                    self.torch.load.assert_not_called()
                    self.deps.AutoModelForCausalLM.from_pretrained.assert_not_called()
                    self.assertFalse(self.output.exists())
                    path.write_bytes(original)

    def test_arbitrary_source_revision_and_legacy_base_default_export(self):
        for key in converter.HEAD_DEFAULTS:
            self.meta.pop(key, None)
        converter.convert(converter.SOURCE_REPO + "@older-release", self.output)
        self.assertEqual(
            self.deps.snapshot_download.call_args_list,
            [
                mock.call(
                    converter.SOURCE_REPO,
                    revision="older-release",
                    allow_patterns=list(converter.SOURCE_FILES),
                ),
                mock.call(converter.BASE_MODEL, revision=None),
            ],
        )
        saved = json.loads((self.output / "kev_config.json").read_text())
        self.assertEqual(saved["temperature"], 1.0)
        self.assertEqual(saved["base_revision"], "b" * 40)

    def test_unrelated_source_and_existing_output_are_rejected(self):
        with self.assertRaises(ValueError):
            converter.convert("other/model", self.output)
        self.output.mkdir()
        sentinel = self.output / "keep"
        sentinel.write_text("untouched")
        with self.assertRaises(FileExistsError):
            converter.convert(str(self.source), self.output)
        self.assertEqual(sentinel.read_text(), "untouched")

    def test_incompatible_adapter_semantics_are_rejected_before_model_load(self):
        path = self.source / "adapter_config.json"
        original = json.loads(path.read_text())
        for field, value in (
            ("lora_alpha", 64),
            ("r", 8),
            ("use_rslora", True),
            ("use_dora", True),
            ("rank_pattern", {"q_proj": 8}),
            ("modules_to_save", ["embed_tokens"]),
        ):
            with self.subTest(field=field):
                path.write_text(json.dumps({**original, field: value}))
                with self.assertRaisesRegex(ValueError, field):
                    converter.convert(str(self.source), self.output)
                self.deps.AutoModelForCausalLM.from_pretrained.assert_not_called()
                self.assertFalse(self.output.exists())

    def test_source_mutation_during_merge_does_not_publish_manifest(self):
        def mutate_source(safe_merge):
            merged = self._merge(safe_merge)
            (self.source / "adapter_model.safetensors").write_bytes(
                b"changed during merge"
            )
            return merged

        self.adapted.merge_and_unload.side_effect = mutate_source
        with self.assertRaisesRegex(ValueError, "changed during conversion"):
            converter.convert(str(self.source), self.output)
        self.assertFalse((self.output / "kev_config.json").exists())

    def test_missing_adapter_cannot_export_base_only(self):
        self.adapted.named_parameters.return_value = []
        with self.assertRaisesRegex(ValueError, "496"):
            converter.convert(str(self.source), self.output)
        self.model.save_pretrained.assert_not_called()
        self.assertTrue(self.output.is_dir())
        self.assertFalse((self.output / "kev_config.json").exists())

    def test_source_adapter_missing_or_substituted_key_is_rejected_before_peft(self):
        original = dict(self.source_tensors)
        key = "base_model.model.layers.3.self_attn.q_proj.lora_A.weight"
        for substitute in (False, True):
            with self.subTest(substitute=substitute):
                self.source_tensors = dict(original)
                tensor = self.source_tensors.pop(key)
                if substitute:
                    self.source_tensors[key.replace("layers.3.", "layers.32.")] = tensor
                    self.assertEqual(len(self.source_tensors), 496)
                with self.assertRaisesRegex(ValueError, "Source adapter tensor keys"):
                    converter.convert(
                        str(self.source), self.root / f"missing-{substitute}"
                    )
                self.deps.PeftModel.from_pretrained.assert_not_called()
                self.adapted.merge_and_unload.assert_not_called()

    def test_source_adapter_shape_and_dtype_are_checked_before_peft(self):
        for suffix, replacement in (
            ("lora_A.weight", Tensor((8, 2560))),
            ("lora_B.weight", Tensor((8192, 8))),
            ("lora_A.weight", Tensor((16, 2560), "bf16")),
        ):
            key = f"base_model.model.layers.3.self_attn.q_proj.{suffix}"
            with self.subTest(
                suffix=suffix, shape=replacement.shape, dtype=replacement.dtype
            ):
                original = self.source_tensors[key]
                self.source_tensors[key] = replacement
                self.assertEqual(len(self.source_tensors), 496)
                output = (
                    self.root
                    / f"bad-{suffix}-{replacement.dtype}-{replacement.shape[0]}"
                )
                with self.assertRaisesRegex(
                    ValueError, "Invalid source fp32 LoRA tensor"
                ):
                    converter.convert(str(self.source), output)
                self.deps.PeftModel.from_pretrained.assert_not_called()
                self.adapted.merge_and_unload.assert_not_called()
                self.assertFalse((output / "kev_config.json").exists())
                self.source_tensors[key] = original

    def test_loaded_adapter_targets_must_match_validated_source(self):
        loaded = self.adapted.named_parameters.return_value
        name, tensor = loaded[0]
        loaded[0] = (name.replace("mlp.gate_proj", "linear_attn.conv1d"), tensor)
        self.assertEqual(len(loaded), 496)
        with self.assertRaisesRegex(ValueError, "match the source targets"):
            converter.convert(str(self.source), self.output)
        self.deps.PeftModel.from_pretrained.assert_called_once()
        self.adapted.merge_and_unload.assert_not_called()
        self.assertFalse((self.output / "kev_config.json").exists())

    def test_wrong_base_and_rounded_adapter_are_rejected(self):
        self.model.config.hidden_size = 5120
        with self.assertRaisesRegex(ValueError, "hidden_size"):
            converter.convert(str(self.source), self.output)
        self.deps.PeftModel.from_pretrained.assert_not_called()
        self.model.save_pretrained.assert_not_called()
        self.assertFalse((self.output / "kev_config.json").exists())

        self.model.config.hidden_size = 2560
        self.adapted.named_parameters.return_value[0][1].dtype = "bf16"
        other_output = self.root / "other_output"
        with self.assertRaisesRegex(ValueError, "fp32 LoRA"):
            converter.convert(str(self.source), other_output)
        self.adapted.merge_and_unload.assert_not_called()
        self.assertFalse((other_output / "kev_config.json").exists())

    def test_failures_keep_partial_artifacts_without_enabling_manifest(self):
        self.tokenizer.convert_tokens_to_ids.side_effect = lambda token: -1
        with self.assertRaisesRegex(ValueError, "delimiter"):
            converter.convert(str(self.source), self.output)
        self.assertTrue((self.output / "model.safetensors").exists())
        self.assertFalse((self.output / "kev_config.json").exists())

    def test_manifest_rename_failure_keeps_disabled_partial(self):
        with mock.patch.object(converter.os, "replace", side_effect=OSError("disk")):
            with self.assertRaises(OSError):
                converter.convert(str(self.source), self.output)
        self.assertTrue((self.output / "kev_config.json.partial").exists())
        self.assertFalse((self.output / "kev_config.json").exists())


if __name__ == "__main__":
    unittest.main()
