import os
import unittest

import torch
from torch import nn
from transformers.models.t5.modeling_t5 import T5LayerNorm

from rtp_llm.ops.compute_ops import rtp_llm_ops


class TestT5LayerNorm(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

    def setUp(self):
        torch.manual_seed(734876213)

    def test_logn_attention(self):
        torch.classes.load_library(
            os.environ["TEST_SRCDIR"] + "/rtp_llm/tests/libtest_ops.so"
        )

        eps = 1e-6
        for batch_size in range(32, 4096, 512):
            for hidden_units in range(1024, 4096, 1024):

                self.gamm = torch.rand(hidden_units).cuda()
                self.T5LayerNormOp = torch.classes.unittest.T5LayerNormOp(eps)
                self.T5LayerNorm = T5LayerNorm(hidden_units, eps)
                self.T5LayerNorm.weight = torch.nn.Parameter(self.gamm)

                hidden_states = torch.rand(batch_size, hidden_units).cuda()
                result = self.T5LayerNorm(hidden_states)
                test = self.T5LayerNormOp.forward(hidden_states, self.gamm)
                torch.testing.assert_close(test, result)

    @staticmethod
    def get_scale(weight: torch.Tensor):
        max_abs_value = weight.to(dtype=torch.float32).abs().max()
        max_fp8 = 448.0
        scaling_factor = max_abs_value / max_fp8
        min_scaling_factor = 1.0 / (max_fp8 * 512.0)
        scaling_factor = max(min_scaling_factor, scaling_factor)
        return scaling_factor

    def test_logn_attention_fp8(self):
        torch.classes.load_library(
            os.environ["TEST_SRCDIR"] + "/rtp_llm/tests/libtest_ops.so"
        )

        eps = 1e-6
        for batch_size in range(32, 4096, 512):
            for hidden_units in range(1024, 4096, 1024):

                self.gamm = torch.rand(hidden_units).cuda()
                self.T5LayerNormOp = torch.classes.unittest.T5LayerNormOp(eps)
                self.T5LayerNorm = T5LayerNorm(hidden_units, eps)
                self.T5LayerNorm.weight = torch.nn.Parameter(self.gamm)

                hidden_states = torch.rand(batch_size, hidden_units).cuda()
                expect_fp16 = self.T5LayerNorm(hidden_states)

                scale = self.get_scale(expect_fp16).to(dtype=torch.float32).cuda()
                scale_inv = (1.0 / scale).to(dtype=torch.float32).cuda()

                expect_fp16 = self.T5LayerNorm(hidden_states)
                fp8_res = self.T5LayerNormOp.forward_fp8(
                    hidden_states, scale_inv, self.gamm
                )
                actual = (
                    fp8_res.view(torch.float8_e4m3fn)
                    .view(-1)[: batch_size * hidden_units]
                    .view(batch_size, hidden_units)
                    .to(torch.float32)
                    * scale
                )
                fp16 = self.T5LayerNormOp.forward(hidden_states, self.gamm)
                criterion = nn.MSELoss()

                loss = criterion(expect_fp16, actual).cpu().item()

                self.assertTrue(
                    loss <= 0.00026,
                    f"scale {scale}, loss is too big, actual:{actual}, expect:{expect_fp16}, loss:{loss}",
                )

                fp8_res = self.T5LayerNormOp.forward_fp8(
                    hidden_states, torch.tensor([1.0]).cuda(), self.gamm
                )
                actual = (
                    fp8_res.view(torch.float8_e4m3fn)
                    .view(-1)[: batch_size * hidden_units]
                    .view(batch_size, hidden_units)
                    .to(torch.float32)
                )
                loss = criterion(expect_fp16, actual).cpu().item()
                self.assertTrue(
                    loss <= 0.00026,
                    f"scale 1.0, loss is too big, actual:{actual}, expect:{expect_fp16}, loss:{loss}",
                )

    def test_stride_norm(self):
        torch.classes.load_library(
            os.environ["TEST_SRCDIR"] + "/rtp_llm/tests/libtest_ops.so"
        )

        eps = 1e-6
        for batch_size in [1, 2, 12, 192]:
            for hidden_units_1, hidden_units_2 in [
                (128, 128),
                (1024, 256),
                (4096, 512),
                (1024, 128),
            ]:
                size_per_head = 128

                self.T5LayerNormOp = torch.classes.unittest.T5LayerNormOp(eps)
                self.gamm1 = torch.rand(size_per_head).cuda()
                self.T5LayerNorm1 = T5LayerNorm(size_per_head, eps)
                self.T5LayerNorm1.weight = torch.nn.Parameter(self.gamm1)

                self.gamm2 = torch.rand(size_per_head).cuda()
                self.T5LayerNorm2 = T5LayerNorm(size_per_head, eps)
                self.T5LayerNorm2.weight = torch.nn.Parameter(self.gamm2)

                hidden_states_1 = torch.rand(batch_size, hidden_units_1).cuda()
                hidden_states_2 = torch.rand(batch_size, hidden_units_2).cuda()
                hidden_states_3 = torch.rand(batch_size, hidden_units_2).cuda()

                hidden_states = torch.cat(
                    [hidden_states_1, hidden_states_2, hidden_states_3], dim=1
                )

                expect_result_1 = (
                    self.T5LayerNorm1(
                        hidden_states_1.reshape([batch_size, -1, size_per_head])
                    )
                    .reshape([batch_size, -1])
                    .cpu()
                )
                expect_result_2 = (
                    self.T5LayerNorm2(
                        hidden_states_2.reshape([batch_size, -1, size_per_head])
                    )
                    .reshape([batch_size, -1])
                    .cpu()
                )

                stride = hidden_units_1 + hidden_units_2 * 2
                torch.set_printoptions(threshold=float("inf"))

                actual_1 = (
                    self.T5LayerNormOp.stride_forward(
                        hidden_states, self.gamm1, hidden_units_1, 0, stride
                    )
                    .reshape([batch_size, -1])
                    .cpu()
                )
                actual_2 = (
                    self.T5LayerNormOp.stride_forward(
                        hidden_states,
                        self.gamm2,
                        hidden_units_2,
                        hidden_units_1,
                        stride,
                    )
                    .reshape([batch_size, -1])
                    .cpu()
                )

                torch.testing.assert_close(
                    actual_1,
                    expect_result_1,
                    rtol=0.001,
                    atol=0.001,
                    msg=f"[{batch_size}, {hidden_units_1}, {hidden_units_2}], input_1: {hidden_states_1}, actual_1: {actual_1}, expect_1:{expect_result_1}",
                )
                torch.testing.assert_close(
                    actual_2,
                    expect_result_2,
                    rtol=0.001,
                    atol=0.001,
                    msg=f"[{batch_size}, {hidden_units_1}, {hidden_units_2}], input_2: {hidden_states_2}, actual_2: {actual_2}, expect_2:{expect_result_2}",
                )


if __name__ == "__main__":
    unittest.main()
