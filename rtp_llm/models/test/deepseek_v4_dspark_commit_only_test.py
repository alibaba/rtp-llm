from types import SimpleNamespace
from unittest import TestCase, mock

import torch.nn as nn

from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig
from rtp_llm.model_loader.model_weight_info import ModelWeightInfo
from rtp_llm.model_loader.per_block_fp8_quant_weight import V4PerBlockFp8Weight
from rtp_llm.model_loader.weight_module import AtomicWeight
from rtp_llm.models.deepseek_v4 import DeepSeekV4DSparkWeight, DeepSeekV4Weight
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
        main_proj = _fp8_descriptor(
            W.v4_dspark_main_proj_w, "mtp.0.main_proj.weight"
        )
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
                AtomicWeight(W.v4_dspark_markov_w1, [], identity),
                AtomicWeight(W.v4_dspark_markov_w2, [], identity),
            ],
        )

    def test_prefill_keeps_exact_commit_keys_and_fp8_scales(self) -> None:
        descriptor = DeepSeekV4DSparkWeight.__new__(DeepSeekV4DSparkWeight)
        descriptor.role_type = "PREFILL"
        full = self._full_info()

        with mock.patch.object(
            DeepSeekV4Weight, "get_weight_info", return_value=full
        ):
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
        self.assertTrue(all(call.kwargs["commit_only"] for call in build.call_args_list))
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
        ) as attention, mock.patch.object(block_module, "MoE") as moe:
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


if __name__ == "__main__":
    import unittest

    unittest.main()
