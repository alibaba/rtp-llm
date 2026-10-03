"""M3.1-only raw norm loading, with legacy gamma compatibility."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models.minimax_m3 import MiniMaxM3Weight, add_unit_offset
from rtp_llm.models.minimax_m31 import M31_RAW_ATTENTION_NORMS, MiniMaxM31Weight
from rtp_llm.utils.model_weight import W, identity, sp_id


class RawNormTest(unittest.TestCase):
    def test_all_sparse_norm_names_keep_raw_values(self):
        weight = object.__new__(MiniMaxM31Weight)
        weight.prefix = "language_model."
        weight._use_qk_norm = True
        with patch.object(
            MiniMaxM3Weight, "_get_hf_layer_weight_info", return_value=[]
        ), patch.object(weight, "_should_load_msa_index", return_value=True):
            modules = weight._get_hf_layer_weight_info(59)
        self.assertEqual({m.name for m in modules}, set(M31_RAW_ATTENTION_NORMS))
        for module in modules:
            self.assertIs(module.process_fun, identity)
            self.assertIs(module._get_split_func(), sp_id)
            self.assertTrue(module.disable_quantization)
            self.assertEqual(
                module.weights[0].name,
                "language_model.model.layers.{i}.self_attn."
                + M31_RAW_ATTENTION_NORMS[module.name]
                + ".weight",
            )
            raw = torch.arange(128, dtype=torch.bfloat16)
            for tp, ep, dp in ((1, 4, 1), (1, 4, 4), (4, 4, 1), (1, 8, 8)):
                config = SimpleNamespace(
                    tp_size=tp,
                    tp_rank=0,
                    ep_size=ep,
                    ep_rank=0,
                    dp_size=dp,
                    dp_rank=0,
                    ffn_tp_size=1,
                    ffn_tp_rank=0,
                    hidden_size=6144,
                    head_num=64,
                    head_num_kv=4,
                    size_per_head=128,
                    moe_pure_tp_mode=False,
                    bit=16,
                )
                split = module._split({module.name: raw}, config)
                self.assertTrue(torch.equal(split[module.name], raw))

    def test_m31_keeps_raw_checkpoint_norms_without_changing_m3(self):
        for cls in (MiniMaxM3Weight, MiniMaxM31Weight):
            weight = object.__new__(cls)
            weight.prefix = "language_model."
            weight._hidden_size = 6144
            weight._size_per_head = 128
            weight._head_num = 64
            weight._head_num_kv = 4
            weight._use_qk_norm = True
            weight._sparse_layer_set = set()
            with patch.object(
                weight, "_get_hf_ffn_layer_weight_info", return_value=[]
            ), patch.object(weight, "_should_load_msa_index", return_value=False):
                modules = weight._get_hf_layer_weight_info(0)
            components = {c.name: c for m in modules for c in m.get_components()}
            self.assertIs(components[W.q_ln_gamma].process_fun, add_unit_offset)
            self.assertIs(components[W.k_ln_gamma].process_fun, add_unit_offset)
            raw_keys = set(components) & set(M31_RAW_ATTENTION_NORMS)
            if cls is MiniMaxM3Weight:
                self.assertFalse(raw_keys)
            else:
                self.assertEqual(
                    raw_keys, {"minimax_m31.raw_q_norm", "minimax_m31.raw_k_norm"}
                )
                for key in raw_keys:
                    component = components[key]
                    self.assertIs(component.process_fun, identity)
                    self.assertEqual(component.data_type, torch.bfloat16)
                    raw = torch.tensor([0.001953125, -0.99609375], dtype=torch.bfloat16)
                    self.assertTrue(torch.equal(component.process_fun([raw]), raw))
                    self.assertEqual(
                        component.weights[0].name,
                        "language_model.model.layers.{i}.self_attn."
                        + M31_RAW_ATTENTION_NORMS[key]
                        + ".weight",
                    )


if __name__ == "__main__":
    unittest.main()
