import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.config.maxpander_config import MaxPanderConfig
from rtp_llm.models_py.model_desc.deepseek_v4_model import DeepSeekV4Model
from rtp_llm.models_py.model_desc.deepseek_v41_base_model import (
    DeepSeekV4Model as DeepSeekV41Model,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.jit_warmup import (
    make_moe_warmup_expert_ids,
)


class DeepSeekMaxPanderWarmupTest(unittest.TestCase):
    def test_descriptor_warmup_uses_logical_ids_only_when_enabled(self):
        for model_type, experts in (
            (DeepSeekV4Model, 256),
            (DeepSeekV41Model, 384),
        ):
            for enabled in (False, True):
                for physical in (experts, experts * 5 // 4, experts * 2):
                    for ep_size in (2, 4):
                        for rank in range(ep_size):
                            for strategy in ("mega_moe", "mega_moe_se"):
                                with self.subTest(
                                    model=model_type,
                                    enabled=enabled,
                                    physical=physical,
                                    ep_size=ep_size,
                                    rank=rank,
                                    strategy=strategy,
                                ):
                                    self.check_descriptor(
                                        model_type,
                                        experts,
                                        physical,
                                        ep_size,
                                        rank,
                                        strategy,
                                        enabled,
                                    )

    def check_descriptor(
        self, model_type, experts, physical, ep_size, rank, strategy, enabled
    ):
        config = SimpleNamespace(
            dim=16,
            n_routed_experts=experts,
            n_activated_experts=6,
            ep_size=ep_size,
            ep_rank=rank,
            local_expert_start=rank * (physical // ep_size),
            local_expert_end=(rank + 1) * (physical // ep_size),
            maxpander_config=MaxPanderConfig(enable=enabled).resolve("PREFILL"),
        )
        executor = Mock()
        executor.cfg = config
        executor.name = "mega" if strategy == "mega_moe" else "mega_se"
        owner = SimpleNamespace(
            v4=SimpleNamespace(
                layers=[
                    SimpleNamespace(
                        ffn=SimpleNamespace(
                            strategy_name=strategy,
                            fused_moe=SimpleNamespace(fused_experts=executor),
                            _strategy=executor,
                        )
                    )
                ]
            )
        )
        with patch("torch.distributed.is_available", return_value=True), patch(
            "torch.distributed.is_initialized", return_value=True
        ), patch("torch.distributed.barrier") as barrier, patch(
            "torch.cuda.synchronize"
        ) as synchronize:
            model_type._prewarm_mega_moe(owner, "cpu")
        executor.assert_called_once()
        self.assertEqual(barrier.call_count, 2)
        synchronize.assert_called_once()
        inputs, weights, indices = executor.call_args.args
        self.assertEqual(inputs.shape, (1, 16))
        self.assertEqual(inputs.dtype, torch.bfloat16)
        torch.testing.assert_close(weights, torch.tensor([[1, 0, 0, 0, 0, 0.0]]))
        self.assertEqual(indices.dtype, torch.int64)
        expected_start = rank * ((experts if enabled else physical) // ep_size)
        expected = torch.arange(6).unsqueeze(0) + expected_start
        torch.testing.assert_close(indices, expected)
        torch.testing.assert_close(
            indices[0], make_moe_warmup_expert_ids(config, "cpu")
        )
        if enabled:
            self.assertTrue(torch.all((indices >= 0) & (indices < experts)))

    def test_non_mega_descriptor_does_not_warm_up(self):
        owner = SimpleNamespace(
            v4=SimpleNamespace(
                layers=[
                    SimpleNamespace(
                        ffn=SimpleNamespace(
                            strategy_name="other",
                            _strategy=SimpleNamespace(name="other"),
                        )
                    )
                ]
            )
        )
        with patch("torch.distributed.barrier") as barrier, patch(
            "torch.cuda.synchronize"
        ) as synchronize:
            DeepSeekV4Model._prewarm_mega_moe(owner, "cpu")
            DeepSeekV41Model._prewarm_mega_moe(owner, "cpu")
        barrier.assert_not_called()
        synchronize.assert_not_called()


if __name__ == "__main__":
    unittest.main()
