import unittest

import torch

from rtp_llm.models_py.modules.kimi_k3.moe import _requires_nccl_ep
from rtp_llm.models_py.triton_kernels.common.activation import (
    situ_and_mul,
    situ_mul_fp8_quant_packed_masked,
)


class KimiK3NcclEpTopologyTest(unittest.TestCase):
    def test_single_host_keeps_mega_moe(self) -> None:
        self.assertFalse(_requires_nccl_ep(8, 8))
        self.assertFalse(_requires_nccl_ep(16, 16))

    def test_multi_host_uses_nccl_ep(self) -> None:
        self.assertTrue(_requires_nccl_ep(16, 8))
        self.assertTrue(_requires_nccl_ep(32, 8))

    def test_invalid_topology_is_rejected(self) -> None:
        for world_size, local_world_size in ((0, 8), (8, 0), (10, 8)):
            with self.subTest(
                world_size=world_size,
                local_world_size=local_world_size,
            ):
                with self.assertRaises(ValueError):
                    _requires_nccl_ep(world_size, local_world_size)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_masked_situ_packed_quant_matches_reference(self) -> None:
        torch.manual_seed(7)
        experts, capacity, hidden = 2, 8, 128
        gate_up = torch.randn(
            experts,
            capacity,
            2 * hidden,
            dtype=torch.bfloat16,
            device="cuda",
        )
        counts = torch.tensor([3, 5], dtype=torch.int32, device="cuda")
        quantized, packed_scale = situ_mul_fp8_quant_packed_masked(
            gate_up,
            counts,
            beta=1.7,
            linear_beta=2.3,
        )
        self.assertEqual(tuple(quantized.shape), (experts, capacity, hidden))
        self.assertEqual(tuple(packed_scale.shape), (experts, capacity, 1))
        for expert, count in enumerate((3, 5)):
            reference = situ_and_mul(
                gate_up[expert, :count, :hidden].contiguous(),
                gate_up[expert, :count, hidden:].contiguous(),
                beta=1.7,
                linear_beta=2.3,
            ).float()
            exponent = (
                packed_scale[expert, :count, 0].bitwise_and(0xFF).float()
                - 127.0
            )
            dequantized = quantized[expert, :count].float() * torch.exp2(
                exponent
            ).unsqueeze(-1)
            torch.testing.assert_close(
                dequantized,
                reference,
                rtol=0.08,
                atol=0.08,
            )


if __name__ == "__main__":
    unittest.main()
