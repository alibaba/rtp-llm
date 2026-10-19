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
        "base_revision": converter.BASE_REVISION,
        "lora": 16,
        "head_dim": 256,
        "option_isolation": False,
        "special_embeddings": False,
        "weights_dtype": "fp32",
        "lora_placement": "full",
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
    def test_pinned_contract(self):
        converter._validate_head(_meta(), _torch(None))

    def test_unsupported_metadata_and_nonfinite_temperatures(self):
        invalid = {
            "base": ["Qwen/Qwen3.5-4B"],
            "base_revision": ["main"],
            "lora": [0, 8, True],
            "weights": ["full"],
            "weights_dtype": ["bf16"],
            "head_dim": [128],
            "option_isolation": [True, 0],
            "special_embeddings": [True],
            "lora_placement": ["dense"],
            "temperature": [float("nan"), float("inf"), -1, 0, True, "2.4"],
            "unknown_head_option": [True],
        }
        for key, values in invalid.items():
            for value in values:
                with self.subTest(key=key, value=value):
                    meta = _meta()
                    meta[key] = value
                    with self.assertRaises(ValueError):
                        converter._validate_head(meta, _torch(None))

    def test_bad_shapes_dtypes_keys_and_nonfinite_head(self):
        for key, shape in converter.HEAD_SHAPES.items():
            for tensor in (
                Tensor((1,)),
                Tensor(shape, "bf16"),
                Tensor(shape, finite=False),
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


class ExportTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.output = self.root / "output"
        hashes = {}
        for name in converter.SOURCE_SHA256:
            data = f"test fixture {name}".encode()
            (self.source / name).write_bytes(data)
            hashes[name] = hashlib.sha256(data).hexdigest()
        self.hashes = mock.patch.object(converter, "SOURCE_SHA256", hashes)
        self.hashes.start()
        self.addCleanup(self.hashes.stop)
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
        self.adapted = mock.Mock()
        self.adapted.named_parameters.return_value = [
            (f"layers.{i}.lora_A.default.weight", Tensor((16, 2560)))
            for i in range(496)
        ]
        self.adapted.merge_and_unload.side_effect = self._merge
        self.tokenizer = mock.Mock()
        self.tokenizer.convert_tokens_to_ids.side_effect = dict(
            zip(converter.SPECIAL_TOKENS, converter.SPECIAL_TOKEN_IDS)
        ).get
        self.tokenizer.save_pretrained.side_effect = self._save_tokenizer
        self.deps = SimpleNamespace(
            torch=self.torch,
            snapshot_download=mock.Mock(return_value=str(self.source)),
            PeftModel=SimpleNamespace(
                from_pretrained=mock.Mock(return_value=self.adapted)
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
        return mock.Mock()

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
        self.assertFalse((self.output / "kev_config.json").exists())

    def test_export_merges_before_cast_and_publishes_manifest_last(self):
        result = converter.convert(converter.SOURCE_REPO, self.output)
        self.deps.snapshot_download.assert_called_once_with(
            converter.SOURCE_REPO,
            revision=converter.SOURCE_REVISION,
            allow_patterns=list(converter.SOURCE_SHA256),
        )
        self.assertEqual(self.events[:2], [("merge", "fp32"), ("cast", "bf16")])
        load = self.deps.AutoModelForCausalLM.from_pretrained.call_args
        self.assertEqual(load.kwargs["dtype"], "fp32")
        self.assertEqual(load.kwargs["revision"], converter.BASE_REVISION)
        self.assertTrue(self.torch.load.call_args.kwargs["weights_only"])
        self.assertEqual(self.torch.load.call_args.kwargs["map_location"], "cpu")
        saved = json.loads((self.output / "kev_config.json").read_text())
        self.assertEqual(saved, result)
        self.assertEqual(saved["special_token_ids"], converter.SPECIAL_TOKEN_IDS)
        self.assertEqual(saved["source_revision"], converter.SOURCE_REVISION)
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

    def test_bad_hash_is_rejected_before_pickle_or_model_load(self):
        for name in converter.SOURCE_SHA256:
            with self.subTest(name=name):
                path = self.source / name
                original = path.read_bytes()
                path.write_bytes(b"untrusted file")
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    converter.convert(str(self.source), self.output)
                self.torch.load.assert_not_called()
                self.deps.AutoModelForCausalLM.from_pretrained.assert_not_called()
                self.assertFalse(self.output.exists())
                path.write_bytes(original)

    def test_unpinned_source_and_existing_output_are_rejected(self):
        for source in ("other/model", converter.SOURCE_REPO + "@main"):
            with self.assertRaises(ValueError):
                converter.convert(source, self.output)
        self.output.mkdir()
        sentinel = self.output / "keep"
        sentinel.write_text("untouched")
        with self.assertRaises(FileExistsError):
            converter.convert(str(self.source), self.output)
        self.assertEqual(sentinel.read_text(), "untouched")

    def test_missing_adapter_cannot_export_base_only(self):
        self.adapted.named_parameters.return_value = []
        with self.assertRaisesRegex(ValueError, "496"):
            converter.convert(str(self.source), self.output)
        self.model.save_pretrained.assert_not_called()
        self.assertTrue(self.output.is_dir())
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
