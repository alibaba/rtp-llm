import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from rtp_llm.models_py.model_desc.kimi_k3 import KimiK3DenseMLP
from rtp_llm.models_py.modules.factory.linear.impl.cuda.f16_linear import CudaF16Linear
from rtp_llm.models_py.modules.kimi_k3.linear import bf16_linear
from rtp_llm.models_py.modules.kimi_k3.moe import situ
from rtp_llm.ops import RoleType
from rtp_llm.utils.model_weight import W


class KimiK3DenseGateUpTest(unittest.TestCase):
    def test_decode_split_gate_up_matches_dense_reference(self):
        torch.manual_seed(306)
        hidden_size, intermediate = 8, 16
        weights = {
            W.ffn_w1: torch.randn(hidden_size, intermediate, dtype=torch.bfloat16),
            W.ffn_w3: torch.randn(hidden_size, intermediate, dtype=torch.bfloat16),
            W.ffn_w2: torch.randn(intermediate, hidden_size, dtype=torch.bfloat16),
        }
        hidden = torch.randn(4, hidden_size, dtype=torch.bfloat16)
        beta, linear_beta = 2.0, 2.0
        config = SimpleNamespace(k3_runtime_config=SimpleNamespace(
            activation_situ_beta=beta, activation_situ_linear_beta=linear_beta,
        ))
        gate = F.linear(hidden, weights[W.ffn_w1].T)
        up = F.linear(hidden, weights[W.ffn_w3].T)
        expected = F.linear(situ(gate, up, beta, linear_beta), weights[W.ffn_w2].T)
        model = KimiK3DenseMLP(
            config, SimpleNamespace(role_type=RoleType.DECODE, tp_size=1, tp_rank=0), weights, None
        )
        self.assertIsNone(model.gate_up)
        self.assertIsInstance(model.gate, CudaF16Linear)
        self.assertIsInstance(model.up, CudaF16Linear)
        self.assertIsInstance(model.down, CudaF16Linear)
        torch.testing.assert_close(model(hidden), expected, rtol=0.02, atol=0.02)

    def test_decode_tp_dense_weights_shard_on_matching_axes(self):
        torch.manual_seed(307)
        weights = {
            W.ffn_w1: torch.randn(8, 16, dtype=torch.bfloat16),
            W.ffn_w3: torch.randn(8, 16, dtype=torch.bfloat16),
            W.ffn_w2: torch.randn(16, 8, dtype=torch.bfloat16),
        }
        originals = {name: value.clone() for name, value in weights.items()}
        config = SimpleNamespace(k3_runtime_config=SimpleNamespace(
            activation_situ_beta=2.0, activation_situ_linear_beta=2.0,
        ))
        model = KimiK3DenseMLP(
            config, SimpleNamespace(role_type=RoleType.DECODE, tp_size=2, tp_rank=1),
            weights, None,
        )
        self.assertIsNone(model.gate_up)
        torch.testing.assert_close(weights[W.ffn_w1], originals[W.ffn_w1][:, 8:])
        torch.testing.assert_close(weights[W.ffn_w3], originals[W.ffn_w3][:, 8:])
        torch.testing.assert_close(weights[W.ffn_w2], originals[W.ffn_w2][8:, :])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_bf16_linear_two_dimensional_result_owns_storage_for_situ(self):
        hidden = torch.randn(4, 8, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(16, 8, device="cuda", dtype=torch.bfloat16)
        output = bf16_linear(hidden, weight)
        self.assertTrue(output.is_contiguous())
        self.assertIsNone(output._base)

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

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_merged_gate_up_runs_cuda_situ_with_split_views(self):
        torch.manual_seed(305)
        weights = {
            W.ffn_w1: torch.randn(8, 16, device="cuda", dtype=torch.bfloat16),
            W.ffn_w3: torch.randn(8, 16, device="cuda", dtype=torch.bfloat16),
            W.ffn_w2: torch.randn(16, 8, device="cuda", dtype=torch.bfloat16),
        }
        hidden = torch.randn(4, 8, device="cuda", dtype=torch.bfloat16)
        gate = F.linear(hidden, weights[W.ffn_w1].T)
        up = F.linear(hidden, weights[W.ffn_w3].T)
        expected = F.linear(situ(gate, up, 2.0, 2.0), weights[W.ffn_w2].T)
        config = SimpleNamespace(
            k3_runtime_config=SimpleNamespace(
                activation_situ_beta=2.0, activation_situ_linear_beta=2.0
            )
        )
        model = KimiK3DenseMLP(config, None, weights, None)
        actual = model(hidden)
        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)


if __name__ == "__main__":
    unittest.main()
