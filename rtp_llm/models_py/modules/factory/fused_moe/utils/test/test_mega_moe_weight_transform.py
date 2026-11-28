import types
import unittest
from unittest import mock

import torch

from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors import mega_moe
from rtp_llm.utils.model_weight import W


class MegaMoeWeightTransformTest(unittest.TestCase):
    def _transform(self, intermediate):
        executor = mega_moe.MegaMoeExecutor.__new__(mega_moe.MegaMoeExecutor)
        torch.nn.Module.__init__(executor)
        executor.cfg = types.SimpleNamespace(
            n_local_experts=2, dim=64, moe_inter_dim=32, moe_w1_layout="gate_up"
        )
        down = torch.full((2, 64, 16), 7, dtype=torch.int8)

        def scale(shape):
            return torch.full(shape, 127, dtype=torch.uint8).view(torch.float8_e8m0fnu)

        weights = {
            W.moe_w1: torch.full((2, 64, 32), 3, dtype=torch.int8),
            W.moe_s1: scale((2, 64, 2)),
            W.moe_w2: down,
            W.moe_s2: scale((2, 64, 1)),
        }
        # The CUDA transform retains the L2 input storage. Keep that contract
        # while testing the executor's real tensor allocations on the CPU.
        deep_gemm = types.SimpleNamespace(
            transform_weights_for_mega_moe=lambda l1, l2: (l1, l2)
        )
        with mock.patch.object(
            mega_moe,
            "prepare_fp4_weight_scale_for_deepgemm",
            side_effect=lambda value, *args: value.view(torch.uint8).to(torch.int32),
        ), mock.patch("torch.cuda.empty_cache"):
            executor._transform_routed_weights(weights, deep_gemm, intermediate)
        return down, executor._mega_l2_w

    def test_resident_down_weights_do_not_pin_loader_storage(self):
        source, resident = self._transform(32)

        self.assertNotEqual(
            source.untyped_storage().data_ptr(),
            resident.untyped_storage().data_ptr(),
        )
        source.zero_()
        torch.testing.assert_close(
            resident, torch.full((2, 64, 16), 7, dtype=torch.int8)
        )

    def test_padded_down_weights_remain_independent_and_zero_filled(self):
        source, resident = self._transform(64)

        source.zero_()
        expected = torch.zeros((2, 64, 32), dtype=torch.int8)
        expected[:, :, :16] = 7
        torch.testing.assert_close(resident, expected)


if __name__ == "__main__":
    unittest.main()
