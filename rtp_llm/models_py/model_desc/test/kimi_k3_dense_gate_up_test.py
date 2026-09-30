import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from rtp_llm.models_py.model_desc.kimi_k3 import KimiK3DenseMLP
from rtp_llm.models_py.modules.kimi_k3.moe import situ
from rtp_llm.utils.model_weight import W


class KimiK3DenseGateUpTest(unittest.TestCase):
    def test_merged_gate_up_preserves_output_and_weight_ownership(self):
        torch.manual_seed(304)
        hidden_size, intermediate = 8, 16
        weights = {
            W.ffn_w1: torch.randn(hidden_size, intermediate, dtype=torch.bfloat16),
            W.ffn_w3: torch.randn(hidden_size, intermediate, dtype=torch.bfloat16),
            W.ffn_w2: torch.randn(intermediate, hidden_size, dtype=torch.bfloat16),
        }
        hidden = torch.randn(4, hidden_size, dtype=torch.bfloat16)
        beta, linear_beta = 2.0, 2.0
        gate = F.linear(hidden, weights[W.ffn_w1].T)
        up = F.linear(hidden, weights[W.ffn_w3].T)
        expected = F.linear(situ(gate, up, beta, linear_beta), weights[W.ffn_w2].T)

        config = SimpleNamespace(
            k3_runtime_config=SimpleNamespace(
                activation_situ_beta=beta, activation_situ_linear_beta=linear_beta
            )
        )
        model = KimiK3DenseMLP(config, None, weights, None)
        torch.testing.assert_close(model(hidden), expected, rtol=0.02, atol=0.02)

        # ModelWeights retains this dictionary. Its two logical weight keys must
        # reference the merged GEMM storage, not duplicate the dense parameters.
        merged_storage = model.gate_up.weight.untyped_storage().data_ptr()
        self.assertEqual(weights[W.ffn_w1].untyped_storage().data_ptr(), merged_storage)
        self.assertEqual(weights[W.ffn_w3].untyped_storage().data_ptr(), merged_storage)
        self.assertEqual(tuple(weights[W.ffn_w1].shape), (hidden_size, intermediate))
        self.assertEqual(tuple(weights[W.ffn_w3].shape), (hidden_size, intermediate))


if __name__ == "__main__":
    unittest.main()
