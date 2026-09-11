"""Exercise MoE routing order and pending shared-output ownership."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.platforms.ppu.models.dsv4.ppu_ep_moe import PpuEPMoE as MoE
from rtp_llm.platforms.ppu.models.dsv4.ppu_decode_provider import PpuDecodeProvider
from rtp_llm.platforms.ppu.models.dsv4.manifest import DECODE_EXECUTION_OPTIONS


def _combine(routed, shared, dtype, *, out):
    out.copy_((routed + shared).to(dtype))
    return out


class SharedScheduleTest(unittest.TestCase):
    def model(self, early, *, route_error=False, fused=False):
        model = MoE.__new__(MoE)
        torch.nn.Module.__init__(model)
        model.dim = 8
        model.layer_id = 5
        model.max_tokens_per_rank = 8
        model._is_decode_role = False
        model.chunking_enabled = True
        model._observer_factory = None
        model._record_function_scope = nullcontext
        model.shared_experts = object()
        calls, pending = [], []

        def gate(x, ids):
            calls.append("route")
            if route_error:
                raise ValueError("routing failed")
            self.assertEqual(x.shape[0], ids.numel())
            return torch.ones_like(ids), ids

        def start(shared, x):
            self.assertIs(shared, model.shared_experts)
            self.assertFalse(pending)
            calls.append("start")
            pending.append(x.float() * 2)

        def finish():
            self.assertEqual(len(pending), 1)
            calls.append("finish")
            return pending.pop()

        def routed(x, weights, indices):
            calls.append("routed")
            return x.float() * 0.5

        model.gate = gate
        model._strategy = routed
        model._shared_executor = (
            None
            if fused
            else SimpleNamespace(start_before_routing=early, start=start, finish=finish)
        )
        return model, calls, pending

    def test_full_forward_and_chunk_use_selected_order(self):
        with patch(
            "rtp_llm.models_py.modules.dsv4._record_tensor.should_record_layer",
            return_value=False,
        ), patch(
            "rtp_llm.platforms.ppu.models.dsv4.ppu_ep_moe.combine_routed_and_shared",
            side_effect=_combine,
        ):
            for early in (False, True):
                for chunk in (False, True):
                    model, calls, pending = self.model(early)
                    for batch in (1, 3, 8):
                        x = torch.arange(batch * 8, dtype=torch.bfloat16).reshape(
                            batch, 8
                        )
                        ids = torch.arange(batch)
                        if chunk:
                            actual = torch.full_like(x, float("nan"))
                            model._run_chunk(x, ids, actual)
                        else:
                            actual = model(x, ids)
                        torch.testing.assert_close(
                            actual, (x.float() * 2.5).to(x.dtype)
                        )
                        prefix = ["start", "route"] if early else ["route", "start"]
                        self.assertEqual(calls, prefix + ["routed", "finish"])
                        self.assertFalse(pending)
                        calls.clear()

    def test_route_failure_drains_only_started_work(self):
        for early in (False, True):
            model, calls, pending = self.model(early, route_error=True)
            with self.assertRaisesRegex(ValueError, "routing failed"):
                model._route_and_start_shared(torch.ones(1, 8), torch.zeros(1))
            self.assertEqual(
                calls, ["start", "route", "finish"] if early else ["route"]
            )
            self.assertFalse(pending)

    def test_routed_failure_drains_started_shared_work(self):
        model, calls, pending = self.model(True)

        def fail(*args):
            raise ValueError("routed failed")

        model._strategy = fail
        with self.assertRaisesRegex(ValueError, "routed failed"):
            model(torch.ones(1, 8), torch.zeros(1))
        self.assertEqual(calls, ["start", "route", "finish"])
        self.assertFalse(pending)

    def test_provider_owns_early_shared_execution(self):
        provider = PpuDecodeProvider(DECODE_EXECUTION_OPTIONS)
        executor = provider.build_shared_expert_executor()
        self.assertTrue(executor.start_before_routing)
        self.assertIsNot(
            provider.stream_pool,
            PpuDecodeProvider(DECODE_EXECUTION_OPTIONS).stream_pool,
        )


if __name__ == "__main__":
    unittest.main()
