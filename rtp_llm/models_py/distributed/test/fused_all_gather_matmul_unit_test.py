import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import rtp_llm.models_py.distributed.symm_mem as fused_ag_gemm


class FusedAllGatherMatmulUnitTest(unittest.TestCase):
    def test_workspace_reservation_forwards_group_and_size(self) -> None:
        group = SimpleNamespace(group_name="tp-test")
        with patch.object(
            fused_ag_gemm, "torch_symm_mem_available", True
        ), patch.object(
            fused_ag_gemm.torch_symm_mem,
            "get_symm_mem_workspace",
        ) as reserve:
            fused_ag_gemm.reserve_fused_all_gather_matmul_workspace(group, 123456)

        reserve.assert_called_once_with("tp-test", min_size=123456)

    def test_fused_call_is_a_thin_operator_wrapper(self) -> None:
        group = SimpleNamespace(group_name="tp-test")
        local_a = torch.empty((2, 3))
        weight = torch.empty((3, 4))
        output = torch.empty((2, 4))
        with patch.object(
            torch.ops.symm_mem,
            "fused_all_gather_matmul",
            return_value=(None, [output]),
        ) as fused_op:
            gathered, outputs = fused_ag_gemm.fused_all_gather_matmul(
                local_a,
                [weight],
                group,
                return_gathered=False,
            )

        fused_op.assert_called_once_with(
            local_a,
            [weight],
            0,
            "tp-test",
            return_A=False,
        )
        self.assertIsNone(gathered)
        self.assertEqual(len(outputs), 1)
        self.assertIs(outputs[0], output)


if __name__ == "__main__":
    unittest.main()
