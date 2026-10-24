import unittest
from types import SimpleNamespace
from unittest import mock

import torch
import torch.distributed as dist

from rtp_llm.models_py.model_desc.deepseek_v4_model import DeepSeekV4Model
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe import (
    MegaMoeExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_se import (
    MegaMoeSEExecutor,
)


class MoeWarmupTest(unittest.TestCase):
    def test_direct_executor_contract_and_collective_order(self):
        for strategy, executor_type in (
            ("mega_moe", MegaMoeExecutor),
            ("mega_moe_se", MegaMoeSEExecutor),
        ):
            for distributed in (False, True):
                with self.subTest(strategy=strategy, distributed=distributed):
                    executor = executor_type.__new__(executor_type)
                    torch.nn.Module.__init__(executor)
                    executor.cfg = SimpleNamespace(
                        dim=8,
                        n_activated_experts=4,
                        n_routed_experts=4,
                        physical_expert_num=6,
                        local_expert_start=3,
                        local_expert_end=6,
                    )
                    model = DeepSeekV4Model.__new__(DeepSeekV4Model)
                    torch.nn.Module.__init__(model)
                    model.v4 = SimpleNamespace(
                        layers=[
                            SimpleNamespace(
                                ffn=SimpleNamespace(
                                    strategy_name=strategy,
                                    fused_moe=SimpleNamespace(fused_experts=executor),
                                )
                            )
                        ]
                    )
                    events = []
                    with (
                        mock.patch.object(dist, "is_available", return_value=True),
                        mock.patch.object(
                            dist, "is_initialized", return_value=distributed
                        ),
                        mock.patch.object(
                            dist,
                            "barrier",
                            side_effect=lambda: events.append("barrier"),
                        ),
                        mock.patch.object(
                            torch.cuda,
                            "synchronize",
                            side_effect=lambda: events.append("synchronize"),
                        ),
                        mock.patch.object(
                            executor, "forward", autospec=True
                        ) as forward,
                    ):
                        forward.side_effect = lambda *args, **kwargs: events.append(
                            "launch"
                        )
                        model._warmup_mega_moe("cpu")

                    forward.assert_called_once()
                    x, weights, indices = forward.call_args.args
                    self.assertEqual(forward.call_args.kwargs, {"expert_load": None})
                    self.assertEqual(x.shape, (1, 8))
                    self.assertEqual(x.dtype, torch.bfloat16)
                    self.assertEqual(x.count_nonzero().item(), 0)
                    self.assertEqual(weights.dtype, torch.float32)
                    self.assertEqual(weights.tolist(), [[1.0, 0.0, 0.0, 0.0]])
                    self.assertEqual(indices.dtype, torch.int64)
                    self.assertTrue(indices.is_contiguous())
                    self.assertEqual(indices.tolist(), [[3, 4, 5, 3]])
                    self.assertEqual(
                        events,
                        (
                            ["barrier", "launch", "synchronize", "barrier"]
                            if distributed
                            else ["launch", "synchronize"]
                        ),
                    )

    def test_other_strategies_skip_executor_and_synchronization(self):
        model = DeepSeekV4Model.__new__(DeepSeekV4Model)
        torch.nn.Module.__init__(model)
        model.v4 = SimpleNamespace(
            layers=[SimpleNamespace(ffn=SimpleNamespace(strategy_name="local_loop"))]
        )
        with (
            mock.patch.object(dist, "barrier") as barrier,
            mock.patch.object(torch.cuda, "synchronize") as synchronize,
        ):
            model._warmup_mega_moe("cpu")
        barrier.assert_not_called()
        synchronize.assert_not_called()


if __name__ == "__main__":
    unittest.main()
