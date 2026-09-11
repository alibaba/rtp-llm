"""CPU wiring tests; not a substitute for distributed MegaMoE numerics."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.model_desc import generic_moe
from rtp_llm.models_py.modules.glm5_mega_moe import mega_moe_se_wrapper


class CompleteMoE(torch.nn.Module):
    expert_num = 256
    topk_ids_dtype = torch.int32

    def forward(self, hidden_states, **kwargs):
        return hidden_states + 7

    def forward_prepacked(self, hidden_states):
        return hidden_states + 7


class Tp8SharedContractTest(unittest.TestCase):
    def test_tp8_ep8_initialization_forward_and_clone_keep_complete_output(self):
        parallel = SimpleNamespace(
            tp_size=8,
            ep_size=8,
            dp_size=1,
            get_ffn_tp_size=lambda: 8,
        )
        config = SimpleNamespace(
            hidden_size=8,
            inter_size=4,
            expert_num=256,
            moe_k=8,
            eplb_config=SimpleNamespace(phy_exp_num=lambda n: n),
            quant_config=None,
            moe_style=2,
        )
        moe_config = SimpleNamespace(
            moe_strategy="mega_moe_se", fake_balance_expert=False
        )

        def select(logits, ids, weights):
            ids.zero_()
            weights.fill_(1 / 8)

        with patch.object(
            generic_moe.LinearFactory,
            "create_linear_from_weights",
            return_value=torch.nn.Identity(),
        ), patch.object(generic_moe, "SelectTopk", return_value=select), patch.object(
            mega_moe_se_wrapper, "MegaMoeSEWrapper", return_value=CompleteMoE()
        ), patch.object(
            generic_moe,
            "all_reduce",
            side_effect=AssertionError("complete MoE must not be reduced"),
        ):
            layer = generic_moe.GenericMoeLayer(config, parallel, {}, moe_config)
            self.assertEqual(layer.ffn_tp_size, 1)
            self.assertEqual(parallel.get_ffn_tp_size(), 8)
            self.assertIsInstance(
                layer.shared_expert, generic_moe._FusedSharedExpertSentinel
            )
            x = torch.ones((4, 8))
            for candidate in (layer, layer.clone_for_cuda_graph()):
                torch.testing.assert_close(candidate(x), x + 7)
                torch.testing.assert_close(
                    candidate.forward_prepacked(
                        x, torch.zeros((4, 8), dtype=torch.int32), torch.ones((4, 8))
                    ),
                    x + 7,
                )


if __name__ == "__main__":
    unittest.main()
