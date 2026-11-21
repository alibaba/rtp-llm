"""Real synthetic checkpoints for independent BERT config/input vocabularies."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from rtp_llm.models_py.bert_service_test import (
    make_checkpoint,
    packed_inputs,
    public_load,
)
from rtp_llm.models_py.model_loader import NewLoaderLoadMethod
from rtp_llm.models_py.new_models.bert.checkpoint_metadata import (
    checkpoint_input_vocab_size,
)
from rtp_llm.ops import TaskType


class BertInputVocabularyTest(unittest.TestCase):
    def check_loaded_outputs(self, owner, reference):
        """CPU target checks real input table values without native outputs."""
        inputs, _ = packed_inputs()
        inputs.input_ids[1] = 16
        weight = owner.embeddings.word_embeddings.weight
        actual = torch.nn.functional.embedding(inputs.input_ids, weight)
        expected = reference.embeddings.word_embeddings(inputs.input_ids)
        torch.testing.assert_close(actual, expected)
        for invalid in (-1, 17, 31):
            inputs.input_ids[1] = invalid
            with self.assertRaises(IndexError):
                torch.nn.functional.embedding(inputs.input_ids, weight)

    def test_rank_local_metadata_keeps_format_adapter_ownership(self):
        with tempfile.TemporaryDirectory() as path:
            for rank in range(2):
                torch.save(
                    {"embeddings.word_embeddings.weight": torch.ones(17, 4)},
                    Path(path) / f"consolidated.{rank:02d}.pth",
                )
            config = SimpleNamespace(
                ckpt_path=path, vocab_size=32, input_vocab_size=0, hidden_size=8
            )
            for rank in range(2):
                self.assertEqual(
                    checkpoint_input_vocab_size(
                        config, "bert", tp_rank=rank, tp_size=2
                    ),
                    32,
                )

    @torch.inference_mode()
    def test_stale_config_preserves_real_rows_and_valid_last_token(self):
        for method, extension in (
            ("scratch", "bin"),
            ("scratch", "safetensors"),
            ("fastsafetensors", "safetensors"),
        ):
            with self.subTest(
                method=method, extension=extension
            ), tempfile.TemporaryDirectory() as path:
                root = Path(path)
                reference = make_checkpoint(root)
                reference.resize_token_embeddings(17, mean_resizing=False)
                state = {
                    key: value.contiguous()
                    for key, value in reference.state_dict().items()
                }
                # Config and tokenizer still advertise 32 IDs. We preserve the
                # 17 pretrained input rows, without changing output head meaning.
                if extension == "bin":
                    (root / "model.safetensors").unlink()
                    torch.save({"state_dict": state}, root / "pytorch_model.bin")
                else:
                    save_file(state, str(root / "model.safetensors"))
                tokenizer_config = json.loads(
                    (root / "tokenizer_config.json").read_text()
                )
                tokenizer_config["additional_special_tokens"] = ["word26"]  # ID 31
                (root / "tokenizer_config.json").write_text(
                    json.dumps(tokenizer_config)
                )
                with self.assertLogs(level="WARNING") as logs:
                    model = public_load(root, method, TaskType.DENSE_EMBEDDING)
                self.assertTrue(
                    any("checkpoint input rows=17" in line for line in logs.output)
                )
                self.assertTrue(
                    any("tokenizer reachable IDs" in line for line in logs.output)
                )
                self.assertEqual(model.model_config.vocab_size, 32)
                self.assertEqual(model.model_config.input_vocab_size, 17)
                self.assertIn(31, model.tokenizer.get_real_tokenizer().all_special_ids)
                self.assertEqual(
                    model.model_weights_loader.loaded_method,
                    NewLoaderLoadMethod(method),
                )
                owner = model.py_model
                self.assertEqual(
                    owner.embeddings.word_embeddings.weight.shape, (17, 16)
                )
                torch.testing.assert_close(
                    owner.embeddings.word_embeddings.weight,
                    state["embeddings.word_embeddings.weight"],
                )
                self.check_loaded_outputs(owner, reference)
                with self.assertRaises(ValueError):
                    owner.embeddings.word_embeddings.checkpoint_read_spec(
                        "weight", (32, 16)
                    )

    def test_metadata_is_exact_and_explicit_input_contract_is_not_overridden(self):
        with tempfile.TemporaryDirectory() as path:
            root = Path(path)
            config = SimpleNamespace(
                ckpt_path=path, vocab_size=30522, input_vocab_size=0, hidden_size=8
            )
            shape = (21128, 8)
            save_file(
                {"bert.embeddings.word_embeddings.weight": torch.randn(shape)},
                str(root / "model.safetensors"),
            )
            self.assertEqual(checkpoint_input_vocab_size(config, "bert"), 21128)
            config.input_vocab_size = 30522
            with self.assertRaisesRegex(ValueError, "Explicit BERT input_vocab_size"):
                checkpoint_input_vocab_size(config, "bert")
            config.input_vocab_size = 0
            config.hidden_size = 16
            with self.assertRaisesRegex(ValueError, "hidden size"):
                checkpoint_input_vocab_size(config, "bert")
            config.hidden_size = 8
            save_file(
                {"embeddings.word_embeddings.weight": torch.randn(shape)},
                str(root / "duplicate.safetensors"),
            )
            with self.assertRaisesRegex(ValueError, "duplicate"):
                checkpoint_input_vocab_size(config, "bert")


if __name__ == "__main__":
    unittest.main()
