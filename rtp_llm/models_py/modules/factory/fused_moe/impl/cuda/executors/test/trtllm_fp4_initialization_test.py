import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.trtllm_fp4_executor import (
    TrtllmFp4Executor,
)
from rtp_llm.utils.model_weight import W


class TrtllmFp4InitializationTest(unittest.TestCase):
    def test_cold_build_synchronizes_only_stage_peers(self):
        from rtp_llm.models_py.distributed import collective_torch

        weights = {
            key: torch.ones(1)
            for key in (
                W.moe_w1,
                W.moe_w2,
                W.moe_s1,
                W.moe_s2,
                W.moe_w1_i_s,
                W.moe_w1_s2,
                W.moe_w2_i_s,
                W.moe_w2_s2,
            )
        }
        module = TrtllmFp4Executor.__module__
        for ep_size in (1, 2):
            with self.subTest(ep_size=ep_size), patch(
                module + ".device_support_pdl", return_value=False
            ), patch(
                module + ".get_trtllm_moe_sm100_module"
            ) as compile_module, patch.object(
                collective_torch, "barrier"
            ) as stage_barrier, patch.object(
                torch.distributed, "barrier"
            ) as world_barrier:
                TrtllmFp4Executor(
                    SimpleNamespace(ep_size=ep_size, expert_num=2), None, weights
                )
                compile_module.assert_called_once()
                if ep_size > 1:
                    stage_barrier.assert_called_once_with(collective_torch.Group.STAGE)
                else:
                    stage_barrier.assert_not_called()
                world_barrier.assert_not_called()


if __name__ == "__main__":
    unittest.main()
