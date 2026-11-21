"""Full target->draft handoff and storage/negative tests, not a serving smoke."""

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file
from torch import nn

from rtp_llm.models_py.borrowed_parameters import BorrowedParameter
from rtp_llm.models_py.model_loader import (
    NewLoaderConfig,
    NewLoaderLoadMethod,
    NewModelLoader,
)
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.new_models.qwen3_moe_eagle3.body import Eagle3DraftBody
from rtp_llm.models_py.new_models.qwen3_moe_eagle3.draft_body_test import (
    _CPUAttention,
    _oracle,
)
from rtp_llm.models_py.new_models.qwen3_moe_eagle3.proposal import (
    Eagle3FullVocabularyDraft,
    target_vocabulary_loans,
)
from rtp_llm.models_py.registry import register_model


@register_model("_eagle3_loaded_target_vocab_test")
class _TargetVocabulary(RtpModule):
    def __init__(self, config, load_config):
        super().__init__()
        self.embed_tokens = nn.Embedding(23, 8)
        self.lm_head = nn.Linear(8, 23, bias=False)
        self.requires_grad_(False)


@register_model("_eagle3_full_vocab_test")
class _Draft(Eagle3FullVocabularyDraft):
    fastsafetensors_load_isolated = True

    def __init__(self, config, load_config):
        super().__init__(config, torch.float32)


class Eagle3ProposalTest(unittest.TestCase):
    def test_target_vocabulary_loan_checks_row_identity_not_padded_width(self):
        self.assertEqual(target_vocabulary_loans(SimpleNamespace(output_vocab_ids=[])), ("embedding", "lm_head"))
        for config in (
            SimpleNamespace(enable_output_vocab_pruning=True),
            SimpleNamespace(vocab_size=8, output_vocab_padded_size=8, output_vocab_ids=[0, 3, 7]),
        ):
            with self.assertRaisesRegex(ValueError, "full target LM head"):
                target_vocabulary_loans(config)

    g = (8, 12, 4, 2, 4, 1e-5, torch.float32)

    def setUp(self):
        torch.manual_seed(123)
        self.cfg = dict(
            model_type="_eagle3_full_vocab_test",
            architectures=["LlamaForCausalLMEagle3"],
            vocab_size=23,
            hidden_size=8,
            intermediate_size=12,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            rms_norm_eps=1e-5,
            num_hidden_layers=1,
            hidden_act="silu",
            eagle_config=dict(
                eagle_aux_hidden_state_layer_ids=[2, 0, 1],
                use_aux_hidden_state=True,
                use_input_layernorm_in_first_layer=True,
                use_last_layernorm=True,
            ),
        )
        self.target_state = {
            "embed_tokens.weight": torch.randn(23, 8),
            "lm_head.weight": torch.randn(23, 8),
        }
        self.body_state = {
            n: torch.randn_like(p) / p.shape[-1] ** 0.5
            for n, p in Eagle3DraftBody(*self.g).named_parameters()
        }
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        (self.base / "target").mkdir()
        (self.base / "draft").mkdir()
        save_file(self.target_state, str(self.base / "target/model.safetensors"))
        self.target = NewModelLoader(
            {"model_type": "_eagle3_loaded_target_vocab_test"},
            NewLoaderConfig(device="cpu", load_method="scratch"),
            model_path=str(self.base / "target"),
        ).load()
        self.aliases = {n: BorrowedParameter(self.target, n) for n in self.target_state}

    def load_draft(self, method="scratch", **kw):
        loader = NewModelLoader(
            self.cfg,
            NewLoaderConfig(device="cpu", load_method=method),
            model_path=str(self.base / "draft"),
            parameter_aliases=kw.get("aliases", self.aliases),
        )
        draft = loader.load()
        self.assertEqual(loader._selected_load_method, NewLoaderLoadMethod(method))
        return draft

    def exercise(self, method):
        # Poisoned copies in composite artifacts must not mutate the target.
        t = dict(self.body_state)
        t.update(
            {
                "model.embed_tokens.weight": torch.full((23, 8), 1234.0),
                "lm_head.weight": torch.full((23, 8), -1234.0),
            }
        )
        save_file(t, str(self.base / "draft/model.safetensors"))
        model = self.load_draft(method)
        captures = {i: torch.randn(3, 8) for i in range(3)}
        first = model.merge_target_captures(captures)
        self.assertTrue(
            torch.equal(first, torch.cat([captures[2], captures[0], captures[1]], -1))
        )
        with self.assertRaisesRegex(ValueError, "configured"):
            model.merge_target_captures({1: captures[1]})
        fmha = _CPUAttention(4, 2, 4, object())
        concept = first
        for ids in (torch.tensor([0, 1, 2]), torch.tensor([5, 7, 3])):
            ref, h = _oracle(
                self.body_state,
                self.target_state["embed_tokens.weight"][ids],
                concept,
                self.g,
            )
            out = model.propose(ids, concept, fmha, fmha.cache)
            torch.testing.assert_close(
                out.logits, ref @ self.target_state["lm_head.weight"].T
            )
            torch.testing.assert_close(out.recurrent_hidden, h)
            concept = out.recurrent_hidden
        for n, v in self.target_state.items():
            self.assertIs(self.target.get_parameter(n), model.get_parameter(n))
            torch.testing.assert_close(self.target.get_parameter(n), v, rtol=0, atol=0)

    def test_scratch(self):
        self.exercise("scratch")

    @unittest.skipUnless(
        os.getenv("NEWLOADER_TEST_NATIVE_FST") == "1", "native FST required"
    )
    def test_native(self):
        self.exercise("fastsafetensors")

    def test_no_partial_loan_or_corrupt_publication(self):
        t = dict(self.body_state)
        del t["fc.weight"]
        save_file(t, str(self.base / "draft/model.safetensors"))
        with self.assertRaisesRegex(ValueError, "both target"):
            self.load_draft(aliases={})
        with self.assertRaisesRegex(ValueError, "both target"):
            self.load_draft(aliases={"lm_head.weight": self.aliases["lm_head.weight"]})
        with self.assertRaisesRegex(RuntimeError, "missing"):
            self.load_draft()
        for n, v in self.target_state.items():
            torch.testing.assert_close(self.target.get_parameter(n), v, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
