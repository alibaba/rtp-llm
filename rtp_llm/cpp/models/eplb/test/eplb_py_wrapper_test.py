import os
import random
import sys
import unittest
from dataclasses import dataclass
import pytest

import torch

EplbPyWrapperOP = pytest.importorskip("rtp_llm.cpp.models.eplb.test.libth_eplb_py_wrapper_test").EplbPyWrapperOP


@dataclass
class TestMoeConfig:
    layer_num: int = 61
    log_exp_num: int = 256
    phy_exp_num: int = 288
    ep_rank: int = 0
    ep_size: int = 144
    moe_inter_size: int = 2048
    hidden_size: int = 7168
    use_fp8: bool = False
    quant_group_size: int = 64


class FakeResult:
    def __init__(self, config: TestMoeConfig):
        self.layer_id: int = random.randint(0, config.layer_num - 1)
        self.layer_id_buf = torch.tensor([self.layer_id], dtype=torch.int32)
        self.logic_expert_cnt = torch.randint(
            0, config.layer_num, (config.log_exp_num,), dtype=torch.int32
        )
        max_exp_num = config.phy_exp_num - config.log_exp_num + 1
        exp_per_ep = config.phy_exp_num // config.ep_size
        self.log2phy = torch.randint(
            0, config.phy_exp_num, (config.log_exp_num, max_exp_num), dtype=torch.int32
        )
        self.phy2log = torch.randint(
            0, config.log_exp_num, (config.phy_exp_num,), dtype=torch.int32
        )

        if config.use_fp8:
            self.moe_w1 = torch.randn(
                (exp_per_ep, config.moe_inter_size * 2, config.hidden_size),
            ).to(dtype=torch.float8_e4m3fn)
            self.moe_w2 = torch.randn(
                (exp_per_ep, config.hidden_size, config.moe_inter_size),
            ).to(dtype=torch.float8_e4m3fn)
            self.moe_w1_s = torch.randn(
                (
                    exp_per_ep,
                    config.moe_inter_size * 2 // config.quant_group_size,
                    config.hidden_size // config.quant_group_size,
                ),
                dtype=torch.float32,
            )
            self.moe_w2_s = torch.randn(
                (
                    exp_per_ep,
                    config.hidden_size // config.quant_group_size,
                    config.moe_inter_size // config.quant_group_size,
                ),
                dtype=torch.float32,
            )
        else:
            self.moe_w1 = torch.randn(
                (exp_per_ep, config.moe_inter_size * 2, config.hidden_size),
                dtype=torch.bfloat16,
            )
            self.moe_w2 = torch.randn(
                (exp_per_ep, config.hidden_size, config.moe_inter_size),
                dtype=torch.bfloat16,
            )
            self.moe_w1_s = torch.empty([0])
            self.moe_w2_s = torch.empty([0])


class FakeExpertBalancer:
    def __init__(self, res: FakeResult):
        self.res = res

    def create_balance_plan(self, log_stats: torch.Tensor, gpu_loads: torch.Tensor):
        return (
            self.res.layer_id_buf,
            self.res.logic_expert_cnt,
            self.res.log2phy,
            self.res.phy2log,
        )

    def load_moe_weight(
        self,
        layer_id_tensor: torch.Tensor,
        ep_rank: int,
        ep_size: int,
        phy2log: torch.Tensor,
    ):
        layer_id = layer_id_tensor.item()
        return (
            layer_id,
            self.res.moe_w1,
            self.res.moe_w2,
            self.res.moe_w1_s,
            self.res.moe_w2_s,
        )


class TestEplbPyWrapper(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        sys.path.append(os.environ["TEST_SRCDIR"] + "/rtp_llm/rtp_llm/cpp/eplb/test")

    def _single_test(self, config: TestMoeConfig):
        eplb_op = EplbPyWrapperOP()
        res = FakeResult(config)
        expert_balancer = FakeExpertBalancer(res)
        eplb_op.init(expert_balancer)
        logic_expert_counts = torch.tensor([[1, 2, 3, 4]])
        physic_expert_counts = torch.tensor([[1, 0, -3, 4]])
        eplb_op.create_balance_plan(logic_expert_counts, physic_expert_counts)
        eplb_op.load_moe_weight(config.ep_rank, config.ep_size)

        (
            layer_id,
            layer_id_buf,
            logic_expert_cnt,
            log2phy,
            phy2log,
            moe_weight_1,
            moe_weight_2,
            moe_w1_s,
            moe_w2_s,
        ) = eplb_op.get_result()

        assert layer_id == res.layer_id
        torch.testing.assert_close(layer_id_buf, res.layer_id_buf)
        torch.testing.assert_close(logic_expert_cnt, res.logic_expert_cnt)
        torch.testing.assert_close(log2phy, res.log2phy)
        torch.testing.assert_close(phy2log, res.phy2log)
        torch.testing.assert_close(moe_weight_1, res.moe_w1)
        torch.testing.assert_close(moe_weight_2, res.moe_w2)

        if config.use_fp8:
            torch.testing.assert_close(moe_w1_s, res.moe_w1_s)
            torch.testing.assert_close(moe_w2_s, res.moe_w2_s)

    def test_create_balance_plan_bf16(self):
        config = TestMoeConfig()
        self._single_test(config)

    def test_create_balance_plan_fp8(self):
        config = TestMoeConfig()
        config.use_fp8 = True
        self._single_test(config)


if __name__ == "__main__":
    unittest.main()
