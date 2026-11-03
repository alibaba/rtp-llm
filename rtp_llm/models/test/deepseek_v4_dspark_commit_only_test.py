from types import SimpleNamespace
from unittest import TestCase, mock

import torch
import torch.nn as nn

from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig
from rtp_llm.model_loader.model_weight_info import ModelWeightInfo
from rtp_llm.model_loader.per_block_fp8_quant_weight import V4PerBlockFp8Weight
from rtp_llm.model_loader.weight_module import AtomicWeight
from rtp_llm.models.deepseek_v4 import DeepSeekV4DSparkWeight, DeepSeekV4Weight
from rtp_llm.models.deepseek_v41 import DeepSeekV41DSparkWeight
from rtp_llm.models_py.model_desc.deepseek_v41_base_dspark_model import (
    DeepSeekV4DSparkModel as DeepSeekV41DSparkModel,
)
from rtp_llm.models_py.modules.dsv4 import block as block_module
from rtp_llm.models_py.modules.dsv4 import transformer as transformer_module
from rtp_llm.models_py.modules.dsv4.transformer import V4Args, V4Transformer
from rtp_llm.utils.model_weight import CkptWeightInfo, W, identity


def _fp8_descriptor(name: str, checkpoint_name: str) -> V4PerBlockFp8Weight:
    source = AtomicWeight(
        name,
        [CkptWeightInfo(checkpoint_name, identity)],
        identity,
    )
    return V4PerBlockFp8Weight(
        source,
        Fp8BlockWiseQuantConfig(is_quanted=True),
        name=source.name,
    )


class DeepSeekV4DSparkCommitOnlyWeightTest(TestCase):
    def _full_info(self) -> ModelWeightInfo:
        wkv = _fp8_descriptor(W.v4_attn_wkv_w, "mtp.{i}.attn.wkv.weight")
        main_proj = _fp8_descriptor(W.v4_dspark_main_proj_w, "mtp.0.main_proj.weight")
        return ModelWeightInfo(
            layer_weights=[
                [
                    wkv,
                    AtomicWeight(W.v4_attn_kv_norm, [], identity),
                    AtomicWeight(W.v4_attn_wq_a_w, [], identity),
                ]
                for _ in range(3)
            ],
            weights=[
                AtomicWeight(W.embedding, [], identity),
                AtomicWeight(W.lm_head, [], identity),
                AtomicWeight(W.v4_dspark_main_norm, [], identity),
                main_proj,
                AtomicWeight(W.dspark_markov_w1, [], identity),
                AtomicWeight(W.dspark_markov_w2, [], identity),
            ],
        )

    def test_prefill_keeps_exact_commit_keys_and_fp8_scales(self) -> None:
        descriptor = DeepSeekV4DSparkWeight.__new__(DeepSeekV4DSparkWeight)
        descriptor.role_type = "PREFILL"
        full = self._full_info()

        with mock.patch.object(DeepSeekV4Weight, "get_weight_info", return_value=full):
            filtered = descriptor.get_weight_info()

        expected_layer_names = {W.v4_attn_wkv_w, W.v4_attn_kv_norm}
        for layer in filtered.layer_weights:
            self.assertEqual({weight.name for weight in layer}, expected_layer_names)
            wkv = next(weight for weight in layer if weight.name == W.v4_attn_wkv_w)
            self.assertEqual(
                set(wkv.sub_weights),
                {W.v4_attn_wkv_w, W.v4_attn_wkv_s},
            )

        self.assertEqual(
            {weight.name for weight in filtered.weights},
            {
                W.embedding,
                W.lm_head,
                W.v4_dspark_main_norm,
                W.v4_dspark_main_proj_w,
            },
        )
        main_proj = next(
            weight
            for weight in filtered.weights
            if weight.name == W.v4_dspark_main_proj_w
        )
        self.assertEqual(
            set(main_proj.sub_weights),
            {W.v4_dspark_main_proj_w, W.v4_dspark_main_proj_s},
        )

    def test_non_prefill_roles_keep_full_descriptor(self) -> None:
        for role in ("DECODE", "PDFUSION"):
            with self.subTest(role=role):
                descriptor = DeepSeekV4DSparkWeight.__new__(DeepSeekV4DSparkWeight)
                descriptor.role_type = role
                full = self._full_info()
                with mock.patch.object(
                    DeepSeekV4Weight, "get_weight_info", return_value=full
                ):
                    self.assertIs(descriptor.get_weight_info(), full)


class DeepSeekV4DSparkCommitOnlyConstructionTest(TestCase):
    def test_transformer_omits_non_commit_modules(self) -> None:
        args = V4Args(n_layers=3, n_mtp_layers=0, commit_only=True)
        weights = SimpleNamespace(global_weights={}, weights=[{}, {}, {}])
        layers = [nn.Identity(), nn.Identity(), nn.Identity()]

        with mock.patch.object(
            transformer_module, "_build_block", side_effect=layers
        ) as build:
            model = V4Transformer(args, weights)

        self.assertEqual(build.call_count, 3)
        self.assertTrue(
            all(call.kwargs["commit_only"] for call in build.call_args_list)
        )
        self.assertIsNone(model.embed)
        self.assertIsNone(model.norm)
        self.assertIsNone(model.head_weight)
        self.assertIsNone(model.head_hc)

    def test_commit_block_constructs_attention_only(self) -> None:
        args = V4Args(n_layers=1, n_mtp_layers=0, commit_only=True)
        with mock.patch.object(
            block_module,
            "CommitOnlyAttentionFP8",
            return_value=nn.Identity(),
        ) as attention, mock.patch.object(block_module, "ChunkedFp8Fp4MoeLayer") as moe:
            block = transformer_module._build_block(
                0, args, layer_weights={}, commit_only=True
            )

        attention.assert_called_once()
        moe.assert_not_called()
        self.assertIsNone(block.ffn)
        self.assertIsNone(block.attn_norm)
        self.assertIsNone(block.ffn_norm)
        self.assertIsNone(block.attn_hc)
        self.assertIsNone(block.ffn_hc)


class DeepSeekV41ConfidenceTest(TestCase):
    def test_constructor_keeps_commit_only_confidence_disabled(self):
        def base_init(model, *_args, **_kwargs):
            model._gen_num_per_cycle = 2
            model._v4_args = SimpleNamespace(
                n_layers=3,
                compress_ratios=(0, 0, 0),
                window_size=128,
                dim=3,
            )

        for enabled in (False, True):
            for role in ("PREFILL", "DECODE", "PDFUSION"):
                with self.subTest(enabled=enabled, role=role), mock.patch(
                    "rtp_llm.models_py.model_desc.deepseek_v41_base_dspark_model."
                    "DeepSeekV4Model.__init__",
                    base_init,
                ):
                    config = SimpleNamespace(
                        dspark_confidence_enabled=enabled,
                        dspark_noise_token_id=1,
                        dspark_target_layer_ids=[1],
                        dspark_markov_rank=2,
                    )
                    model = DeepSeekV41DSparkModel(
                        config,
                        SimpleNamespace(role_type=role, tp_size=1, tp_rank=0),
                        weights=None,
                        moe_config=None,
                        max_generate_batch_size=2,
                    )
                    outputs = model.dspark_empty_outputs(2, torch.device("cpu"))
                    self.assertEqual(
                        outputs.dspark_confidence_hidden_logits is not None,
                        enabled and role != "PREFILL",
                    )

    def test_descriptor_is_v41_only_and_excludes_disabled_and_commit_roles(self):
        for enabled in (False, True):
            for role in ("PREFILL", "DECODE", "PDFUSION"):
                with self.subTest(enabled=enabled, role=role):
                    descriptor = DeepSeekV41DSparkWeight.__new__(
                        DeepSeekV41DSparkWeight
                    )
                    descriptor._num_layers = 3
                    descriptor.role_type = role
                    descriptor.model_config = SimpleNamespace(
                        dspark_confidence_enabled=enabled
                    )
                    info = ModelWeightInfo(layer_weights=[[], [], []], weights=[])
                    with mock.patch.object(
                        DeepSeekV4DSparkWeight, "_get_weight_info", return_value=info
                    ):
                        result = descriptor._get_weight_info()
                    heads = [
                        w for w in result.weights if w.name == W.dspark_confidence_head
                    ]
                    self.assertEqual(len(heads), int(enabled and role != "PREFILL"))
                    if heads:
                        self.assertEqual(heads[0].data_type, torch.float32)
                        self.assertEqual(
                            heads[0].weights[0].name,
                            "mtp.2.confidence_head.proj.weight",
                        )

        descriptor = DeepSeekV4DSparkWeight.__new__(DeepSeekV4DSparkWeight)
        descriptor._num_layers = 3
        descriptor._hidden_size = 8
        with mock.patch.object(
            DeepSeekV4DSparkWeight, "_get_hf_layer_weight_info", return_value=[]
        ):
            info = descriptor._get_weight_info()
        self.assertNotIn(W.dspark_confidence_head, {w.name for w in info.weights})

    def _model(self, enabled):
        model = DeepSeekV41DSparkModel.__new__(DeepSeekV41DSparkModel)
        model._dspark_confidence_enabled = enabled
        model._dspark_confidence_hidden_weight = torch.tensor([[1.0, -2.0, 3.0]])
        model._v4_args = SimpleNamespace(dim=3)
        model.init_dspark_proposer(
            width=2, noise_token_id=1, aux_feature_dim=9, hidden_dim=3
        )
        model.v4 = SimpleNamespace(
            _hc_head_reduce=mock.Mock(side_effect=lambda h: h.mean(dim=-2)),
            norm=lambda h: h + 7,
        )
        return model

    def test_scalar_output_uses_pre_norm_hidden_and_one_head_reduction(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                model = self._model(enabled)
                hidden = torch.arange(24, dtype=torch.bfloat16).reshape(2, 2, 2, 3)
                expected_hidden = hidden.mean(dim=-2).reshape(4, 3)
                outputs = model.compute_draft_outputs(hidden)
                model.v4._hc_head_reduce.assert_called_once()
                torch.testing.assert_close(outputs.hidden_states, expected_hidden + 7)
                if enabled:
                    partial = outputs.dspark_confidence_hidden_logits
                    self.assertEqual(partial.dtype, torch.float32)
                    self.assertEqual(tuple(partial.shape), (4, 1))
                    torch.testing.assert_close(
                        partial,
                        expected_hidden.float()
                        @ model._dspark_confidence_hidden_weight.t(),
                    )
                else:
                    self.assertIsNone(outputs.dspark_confidence_hidden_logits)

    def test_warmup_and_empty_rank_preserve_enabled_scalar_contract(self):
        for enabled in (False, True):
            model = self._model(enabled)
            for batch_size in (0, 3):
                with self.subTest(enabled=enabled, batch_size=batch_size):
                    outputs = model.dspark_empty_outputs(
                        batch_size, torch.device("cpu")
                    )
                    self.assertEqual(
                        tuple(outputs.hidden_states.shape), (batch_size * 2, 3)
                    )
                    partial = outputs.dspark_confidence_hidden_logits
                    if enabled:
                        self.assertEqual(tuple(partial.shape), (batch_size * 2, 1))
                        self.assertEqual(partial.dtype, torch.float32)
                    else:
                        self.assertIsNone(partial)

    def test_enabled_model_rejects_missing_or_malformed_weight(self):
        model = self._model(True)
        model._dspark_markov_rank = 2
        with self.assertRaises(KeyError):
            model._load_extra_weights(SimpleNamespace(global_weights={}))
        for weight in (torch.zeros(5), torch.zeros(1, 5, dtype=torch.bfloat16)):
            with self.subTest(shape=tuple(weight.shape), dtype=weight.dtype):
                with self.assertRaisesRegex(ValueError, "FP32 weight"):
                    model._load_extra_weights(
                        SimpleNamespace(
                            global_weights={W.dspark_confidence_head: weight}
                        )
                    )


if __name__ == "__main__":
    import unittest

    unittest.main()
