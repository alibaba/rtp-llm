"""CPU phase-gate/collective-contract tests with the real model and wrapper."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.model_desc import generic_moe
from rtp_llm.models_py.model_desc.minimax_m31 import MiniMaxM31Model
from rtp_llm.models_py.modules.glm5_mega_moe.mega_moe_nvfp4_wrapper import (
    MegaMoeNvfp4Wrapper,
)
from rtp_llm.models_py.modules.glm5_mega_moe.prefill_chunk_plan import PrefillChunkPlan


class PrefillPlanIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.wrapper = object.__new__(MegaMoeNvfp4Wrapper)
        torch.nn.Module.__init__(self.wrapper)
        self.group = object()
        self.wrapper.mega_moe = SimpleNamespace(
            _mega_group=self.group,
            _mega_buf=SimpleNamespace(num_max_tokens_per_rank=8),
        )
        self.model = SimpleNamespace(layer_num=2)
        self.layers = [
            SimpleNamespace(mlp=SimpleNamespace()),
            SimpleNamespace(mlp=SimpleNamespace(fused_moe=self.wrapper)),
        ]

    def plan(self, *, prefill=True, verify=False, warmup=False, capture=False, rows=1):
        inputs = SimpleNamespace(
            attention_inputs=SimpleNamespace(
                is_prefill=prefill, is_target_verify=verify
            )
        )
        with patch.object(
            generic_moe, "cuda_graph_capture_forward_enabled", return_value=capture
        ), patch.object(
            generic_moe, "cuda_graph_warmup_forward_enabled", return_value=warmup
        ):
            return MiniMaxM31Model._prepare_prefill_moe_chunk_plan(
                self.model, inputs, torch.ones(rows, 2), self.layers
            )

    def test_decode_verify_and_graph_never_collect(self):
        with patch("torch.distributed.all_reduce") as reduce:
            for kw in [
                dict(prefill=False),
                dict(verify=True),
                dict(warmup=True),
                dict(capture=True),
            ]:
                self.assertIsNone(self.plan(**kw))
            reduce.assert_not_called()

    def test_fake_rank_joins_global_max_and_plan_is_forward_local(self):
        calls = []

        def reduce(tensor, op, group):
            self.assertIs(group, self.group)
            self.assertIs(op, torch.distributed.ReduceOp.MAX)
            calls.append(tensor.tolist())
            if len(calls) == 1:
                tensor[0] = 3

        with patch("torch.distributed.get_world_size", return_value=8), patch(
            "torch.distributed.all_reduce", side_effect=reduce
        ):
            self.assertEqual(self.plan(rows=1), PrefillChunkPlan(8, 3))
            self.assertEqual(self.plan(rows=1), PrefillChunkPlan(8, 1))
        self.assertEqual(calls, [[1, 8, -8], [1, 8, -8]])

    def test_capacity_mismatch_across_ranks_fails(self):
        def reduce(tensor, **kwargs):
            tensor[1] = 16

        with patch("torch.distributed.get_world_size", return_value=8), patch(
            "torch.distributed.all_reduce", side_effect=reduce
        ):
            with self.assertRaisesRegex(ValueError, "differs across EP ranks"):
                self.plan()

    def test_single_rank_never_collects(self):
        with patch("torch.distributed.get_world_size", return_value=1), patch(
            "torch.distributed.all_reduce"
        ) as reduce:
            self.assertIsNone(self.plan())
            reduce.assert_not_called()

    def test_non_nvfp4_never_collects(self):
        self.layers = [SimpleNamespace(mlp=SimpleNamespace())]
        with patch("torch.distributed.all_reduce") as reduce:
            self.assertIsNone(self.plan())
            reduce.assert_not_called()

    def test_real_wrapper_copies_shared_output_before_dummy(self):
        class ReusingMoE(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self._mega_buf = SimpleNamespace(num_max_tokens_per_rank=8)
                self.scratch = torch.empty(8, 2)
                self.calls = []

            def forward(self, h, w, ids, **kwargs):
                self.calls.append((len(h), w.clone(), kwargs))
                self.scratch.fill_(-777)
                self.scratch[: len(h)].copy_(h * 3)
                return self.scratch[: len(h)]

        self.wrapper.mega_moe = ReusingMoE()
        self.wrapper.expert_num = 4
        h, w, ids = torch.ones(1, 2), torch.ones(1, 4), torch.arange(4).unsqueeze(0)
        extra = {"prefill_chunk_plan": PrefillChunkPlan(8, 3), "swiglu_alpha": 1.7}
        output = self.wrapper(
            h, w, ids, activation="swiglu_oai", extra_expert_args=extra
        )
        torch.testing.assert_close(output, h * 3, rtol=0, atol=0)
        self.assertEqual(len(self.wrapper.mega_moe.calls), 3)
        self.assertIn("prefill_chunk_plan", extra)  # caller-owned dict is untouched
        for _, _, kw in self.wrapper.mega_moe.calls:
            self.assertEqual(kw["extra_expert_args"], {"swiglu_alpha": 1.7})
        # A subsequent legacy call has no retained plan and accepts positional API.
        self.wrapper(h, w, ids, False, "swiglu_oai")
        self.assertEqual(len(self.wrapper.mega_moe.calls), 4)


if __name__ == "__main__":
    unittest.main()
