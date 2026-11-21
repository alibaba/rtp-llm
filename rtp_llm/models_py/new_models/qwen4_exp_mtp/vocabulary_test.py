"""Real safetensors -> loaded target -> draft vocabulary ownership (CPU)."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from rtp_llm.config.qwen4_mtp_vocab import MTPVocabPolicy, mtp_vocab_policy
from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.new_models.qwen4_exp.hyper_connection import (
    Qwen4ProjectionWeight,
)
from rtp_llm.models_py.new_models.qwen4_exp_mtp.input_owner import Qwen4MTPInputOwner
from rtp_llm.models_py.new_models.qwen4_exp_mtp.vocabulary import (
    Qwen4MTPVocabLoan,
    VocabLeaf,
    mtp_checkpoint_filter,
)
from rtp_llm.models_py.registry import register_model
from rtp_llm.models_py.weight_mapper import WeightsMapper


@register_model("_q4_mtp_vocab_target")
class _Target(RtpModule):
    def __init__(self, config, load):
        super().__init__()
        for name in ("embed", "lm_head"):
            setattr(self, name, Qwen4ProjectionWeight((16, 16), fp8_storage=config.fp8))


@register_model("_q4_mtp_vocab_draft")
class _Draft(RtpModule):
    WEIGHTS_MAPPER = WeightsMapper(prefix_mapping={"model.language_model.": ""})
    fastsafetensors_load_isolated = True

    def __init__(self, config, load):
        super().__init__()
        self.policy = config.policy
        self.loan = config.loan
        self.mtp = Qwen4MTPInputOwner(16, 3, norm_eps=1e-6)
        if self.loan is None:
            self.embed_tokens = Qwen4ProjectionWeight((16, 16))
            self.lm_head = Qwen4ProjectionWeight((16, 16))

    def checkpoint_weight_name_filter(self):
        return mtp_checkpoint_filter(self.policy)

    def load_weights(self, weights):
        return super().load_weights(self.WEIGHTS_MAPPER.apply(weights))

    def process_weights_after_loading(self):
        if self.loan is not None:
            self.loan.validate_binding()


def _config(**kw):
    base = dict(
        vocab_size=16,
        hidden_size=16,
        data_type="bf16",
        enable_fp32_lm_head=False,
        _qwen4_full_fp8_nvfp4=True,
        _qwen4_mtp_use_dedicated_embeddings=False,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _capture(target, target_config, draft_config, *, scaled, shape=(16, 16)):
    return Qwen4MTPVocabLoan.capture(
        target,
        target_config,
        draft_config,
        embedding=VocabLeaf("embed.weight", "embed.storage_scale" if scaled else None),
        head=VocabLeaf("lm_head.weight", "lm_head.storage_scale" if scaled else None),
        head_shape=shape,
        embedding_shape=shape,
        device="cpu",
    )


class VocabTest(unittest.TestCase):
    def target(self, fp8):
        values = {
            n: torch.arange(256).view(16, 16).bfloat16() + i
            for i, n in enumerate(("embed.weight", "lm_head.weight"))
        }
        with tempfile.TemporaryDirectory() as d:
            save_file(values, str(Path(d, "model.safetensors")))
            conf = _config(model_type="_q4_mtp_vocab_target", fp8=fp8)
            return NewModelLoader(
                conf,
                NewLoaderConfig(
                    device="cpu", load_method="scratch", compute_dtype=torch.bfloat16
                ),
                model_path=d,
            ).load()

    def test_source_bf16_draft_reads_original_checkpoint_not_fp8(self):
        target = self.target(True)
        draft = _config(
            _qwen4_dense_storage="source_bf16",
            _qwen4_mtp_dense_storage_mode="source_bf16",
        )
        policy = mtp_vocab_policy(_config(), draft)
        self.assertIs(policy, MTPVocabPolicy.LOAD_SOURCE)
        self.assertIsNone(_capture(target, _config(), draft, scaled=True))
        owns = mtp_checkpoint_filter(policy)
        self.assertTrue(owns("model.language_model.embed_tokens.weight"))
        self.assertTrue(owns("lm_head.weight"))
        self.assertTrue(owns("mtp.layers.0.self_attn.indexer.q_layernorm.weight"))
        self.assertFalse(owns("model.language_model.layers.0.mlp.gate.weight"))

    def test_shared_checkpoint_loading_omits_or_reloads_vocab_by_storage_recipe(self):
        target = self.target(True)
        values = {
            "mtp.fc_embedding.weight": torch.randn(16, 16).bfloat16(),
            "mtp.fc_hidden.weight": torch.randn(16, 16).bfloat16(),
            "mtp.pre_fc_norm_embedding.weight": torch.zeros(16).bfloat16(),
            "mtp.pre_fc_norm_hidden.weight": torch.zeros(48).bfloat16(),
            "lm_head.weight": torch.randn(16, 16).bfloat16(),
            "model.language_model.embed_tokens.weight": torch.randn(16, 16).bfloat16(),
            "model.language_model.layers.0.unrelated": torch.ones(1),
        }
        with tempfile.TemporaryDirectory() as folder:
            name = str(Path(folder, "model.safetensors"))
            save_file(values, name)
            for draft in (
                _config(),
                _config(
                    _qwen4_dense_storage="source_bf16",
                    _qwen4_mtp_dense_storage_mode="source_bf16",
                ),
            ):
                policy = mtp_vocab_policy(_config(), draft)
                loan = _capture(target, _config(), draft, scaled=True)
                conf = SimpleNamespace(
                    model_type="_q4_mtp_vocab_draft", policy=policy, loan=loan
                )
                model = NewModelLoader(
                    conf,
                    NewLoaderConfig(
                        device="cpu",
                        load_method="scratch",
                        compute_dtype=torch.bfloat16,
                    ),
                    model_path=folder,
                ).load()
                if loan is None:
                    self.assertEqual(model.embed_tokens.weight.dtype, torch.bfloat16)
                    torch.testing.assert_close(
                        model.embed_tokens.weight,
                        values["model.language_model.embed_tokens.weight"],
                    )
                    torch.testing.assert_close(
                        model.lm_head.weight, values["lm_head.weight"]
                    )
                else:
                    self.assertIs(model.loan.head.weight, target.lm_head.weight)
                    self.assertNotIn(
                        "embed_tokens.weight", dict(model.named_parameters())
                    )
                    self.assertNotIn("lm_head.weight", dict(model.named_parameters()))
            values.pop("mtp.fc_hidden.weight")
            save_file(values, name)
            loan = _capture(target, _config(), _config(), scaled=True)
            conf = SimpleNamespace(
                model_type="_q4_mtp_vocab_draft",
                policy=MTPVocabPolicy.BORROW_SCALED_FP8,
                loan=loan,
            )
            with self.assertRaises((RuntimeError, ValueError)):
                NewModelLoader(
                    conf,
                    NewLoaderConfig(device="cpu", load_method="scratch"),
                    model_path=folder,
                ).load()
            loan.validate_binding()

    def test_dense_loan_and_borrowed_checkpoint_filter(self):
        target = self.target(False)
        config = _config(_qwen4_dense_storage="source_bf16")
        loan = _capture(target, config, config, scaled=False)
        loan.validate_binding()
        self.assertIs(loan.embedding.weight, target.embed.weight)
        self.assertIsNone(loan.embedding.scale)
        owns = mtp_checkpoint_filter(mtp_vocab_policy(config, config))
        self.assertTrue(owns("mtp.fc_embedding.weight"))
        self.assertFalse(owns("lm_head.weight"))
        self.assertFalse(owns("model.language_model.embed_tokens.weight"))

    def test_source_bf16_shared_vocab_keeps_both_quantized_scales(self):
        target = self.target(True)
        draft = _config(
            _qwen4_dense_storage="source_bf16",
            _qwen4_vocab_storage="fp8",
            _qwen4_mtp_dense_storage_mode="source_bf16_shared_vocab",
        )
        loan = _capture(target, _config(), draft, scaled=True)
        loan.validate_binding()
        self.assertIs(loan.embedding.scale, target.embed.storage_scale)
        self.assertIs(loan.head.scale, target.lm_head.storage_scale)
        self.assertIs(loan.embedding.weight, target.embed.weight)
        self.assertIs(loan.head.weight, target.lm_head.weight)
        with torch.inference_mode():
            target.embed.storage_scale = target.embed.storage_scale.clone()
        with self.assertRaisesRegex(RuntimeError, "scale changed"):
            loan.validate_binding()

    def test_invalid_aliases_do_not_write_target(self):
        target = self.target(True)
        orig_weight = target.lm_head.weight
        orig_bytes = orig_weight.view(torch.uint8).clone()
        for config in (
            _config(_qwen4_mtp_use_dedicated_embeddings=True),
            _config(enable_output_vocab_pruning=True),
            _config(hidden_size=8),
        ):
            with self.subTest(config=config), self.assertRaises(ValueError):
                _capture(target, _config(), config, scaled=True)
        with self.assertRaisesRegex(ValueError, "shape"):
            _capture(target, _config(), _config(), scaled=True, shape=(15, 16))
        with self.assertRaisesRegex(ValueError, "paired"):
            _capture(target, _config(), _config(), scaled=False)
        self.assertIs(target.lm_head.weight, orig_weight)
        self.assertTrue(
            torch.equal(target.lm_head.weight.view(torch.uint8), orig_bytes)
        )


if __name__ == "__main__":
    unittest.main()
