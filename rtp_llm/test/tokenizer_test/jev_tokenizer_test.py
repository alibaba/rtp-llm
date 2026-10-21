import json
import tempfile
from pathlib import Path
from unittest import TestCase, main, mock

from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import AutoTokenizer

from rtp_llm.frontend.tokenizer_factory.tokenizer_factory import TokenizerFactory
from rtp_llm.frontend.tokenizer_factory.tokenizers.base_tokenizer import (
    BaseTokenizer,
)
from rtp_llm.frontend.tokenizer_factory.tokenizers.jev_tokenizer import JevTokenizer
from rtp_llm.frontend.tokenizer_factory.tokenizers.qwen_tokenizer import (
    QWenV2Tokenizer,
)

_PROBES = [
    "[kind]: choice\n[state]: image\n[options]:\nA) red\nB) blue\n[decision]:",
    "商品图片：女童白色休闲短裤 / Size 120cm，价格 ¥99.50",
    "Cafe\u0301 nai\u0308ve A\u030a; Café naïve Å",
    "汉字e\u0301和か\u3099、カ\u3099；ไทย ก้ हिंदी कि",
    "👩🏽‍💻 👍🏿 1️⃣ family 👨‍👩‍👧‍👦",
    "<|im_start|>user\n图片：<|im_end|>",
]
_RAW_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+"
    r"|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


class JevTokenizerTest(TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)
        backend = Tokenizer(models.BPE(unk_token="<unk>"))
        backend.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(Regex(_RAW_PATTERN), behavior="isolated"),
                pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
            ]
        )
        backend.decoder = decoders.ByteLevel()
        backend.train_from_iterator(
            _PROBES * 3,
            trainers.BpeTrainer(
                vocab_size=512,
                min_frequency=1,
                special_tokens=["<unk>", "<|im_start|>", "<|im_end|>"],
                initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                show_progress=False,
            ),
        )
        backend.save(str(self.path / "tokenizer.json"))
        backend.model.save(str(self.path))
        self._write(
            "tokenizer_config.json",
            {
                "tokenizer_class": "Qwen2Tokenizer",
                "unk_token": "<unk>",
                "eos_token": "<|im_end|>",
                "pad_token": "<|im_end|>",
                "additional_special_tokens": ["<|im_start|>", "<|im_end|>"],
                "model_max_length": 512,
            },
        )
        self.config = {"model_type": "qwen3_5"}
        self._write("config.json", self.config)
        self._write(
            "jev_config.json",
            {"schema_version": 1, "model_type": "jev", "mode": "system1"},
        )

    def _write(self, name, value):
        (self.path / name).write_text(json.dumps(value), encoding="utf-8")

    def test_matches_official_auto_tokenizer_including_combining_marks(self):
        official = AutoTokenizer.from_pretrained(
            str(self.path), local_files_only=True, trust_remote_code=False
        )
        tokenizer = TokenizerFactory.create(
            str(self.path), str(self.path), "qwen35_dense"
        )
        self.assertIsInstance(tokenizer, JevTokenizer)
        self.assertEqual(tokenizer.config_json["model_type"], "qwen3_5")
        for probe in _PROBES:
            with self.subTest(probe=probe):
                expected = official.encode(probe, add_special_tokens=False)
                self.assertEqual(
                    tokenizer.encode(probe, add_special_tokens=False), expected
                )
        self.assertEqual(
            tokenizer.get_real_tokenizer().backend_tokenizer.to_str(),
            official.backend_tokenizer.to_str(),
        )
        self.assertEqual(
            tokenizer.stop_words_id_list,
            [
                official.encode("<|im_end|>", add_special_tokens=False),
                official.encode("<|im_start|>", add_special_tokens=False),
            ],
        )

    def test_jev_uses_local_auto_loader_without_backend_repairs(self):
        load = AutoTokenizer.from_pretrained
        with mock.patch.object(
            AutoTokenizer, "from_pretrained", wraps=load
        ) as loader, mock.patch.object(
            BaseTokenizer,
            "_transformers_v5_kwargs",
            side_effect=AssertionError("JEV must not restore the raw backend"),
        ), mock.patch.object(
            BaseTokenizer,
            "_fix_post_processor",
            side_effect=AssertionError("JEV must preserve official postprocessing"),
        ):
            JevTokenizer(str(self.path), self.config)
        loader.assert_called_once_with(
            str(self.path), local_files_only=True, trust_remote_code=False
        )

    def test_normal_qwen_retains_existing_backend_path(self):
        (self.path / "jev_config.json").unlink()
        existing = QWenV2Tokenizer(str(self.path), self.config)
        selected = TokenizerFactory.create(
            str(self.path), str(self.path), "qwen35_dense"
        )
        self.assertIs(type(selected), QWenV2Tokenizer)
        for probe in _PROBES:
            self.assertEqual(
                selected.encode(probe, add_special_tokens=False),
                existing.encode(probe, add_special_tokens=False),
            )
        self.assertEqual(
            selected.get_real_tokenizer().backend_tokenizer.to_str(),
            existing.get_real_tokenizer().backend_tokenizer.to_str(),
        )

    def test_tokenizer_override_directory_does_not_change_task_detection(self):
        separate = self.path / "tokenizer"
        separate.mkdir()
        for filename in (
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
        ):
            (separate / filename).write_bytes((self.path / filename).read_bytes())
        selected = TokenizerFactory.create(
            str(self.path), str(separate), "qwen35_dense"
        )
        self.assertIsInstance(selected, JevTokenizer)
        self.assertEqual(selected.path, str(separate))

    def test_invalid_task_metadata_fails_without_falling_back(self):
        self._write(
            "jev_config.json",
            {"schema_version": 2, "model_type": "jev", "mode": "system1"},
        )
        with self.assertRaisesRegex(ValueError, "JEV tokenizer task metadata"):
            TokenizerFactory.create(str(self.path), str(self.path), "qwen35_dense")


if __name__ == "__main__":
    main()
