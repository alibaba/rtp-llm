import unittest
from typing import Optional
from unittest import mock

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models_py.distributed.collective_torch import Group
from rtp_llm.models_py.modules.factory.fused_moe.defs import config_adapter
from rtp_llm.models_py.modules.factory.fused_moe.defs import fused_moe as moe_defs
from rtp_llm.models_py.modules.factory.fused_moe.defs import quant_config
from rtp_llm.models_py.modules.factory.fused_moe.impl.common.router import (
    batched_data_router,
)
from rtp_llm.ops import MoeConfig, ParallelismConfig

EXPERT_NUM = 8
TP_SIZE = 2
EP_RANK = 1
LOCAL_EXPERTS = EXPERT_NUM // TP_SIZE
LOCAL_LO = LOCAL_EXPERTS * EP_RANK
TOP_K = 2
HIDDEN = 4
NUM_TOKENS = 6


class BatchedDataRouterEpTest(unittest.TestCase):
    """Covers the non-local-expert branch that tp_size == ep_size == 1 hides:
    with ep_rank > 0 most top-k slots fall outside this rank, so ``routed`` is a
    real mask over the scratch column and over finalize's zeroing."""

    @staticmethod
    def _make_config(max_tokens: int) -> config_adapter.MoEConfigAdapter:
        model_config = ModelConfig()
        model_config.hidden_size = HIDDEN
        model_config.expert_num = EXPERT_NUM
        model_config.moe_k = TOP_K
        parallelism = ParallelismConfig()
        parallelism.tp_size = TP_SIZE
        parallelism.ep_size = TP_SIZE
        parallelism.ep_rank = EP_RANK
        moe_config = MoeConfig()
        moe_config.ll_num_max_token = max_tokens
        return config_adapter.MoEConfigAdapter(
            model_config=model_config,
            parallelism_config=parallelism,
            moe_config=moe_config,
        )

    @classmethod
    def _make_router(cls, max_tokens: int) -> batched_data_router.BatchedDataRouter:
        return batched_data_router.BatchedDataRouter(
            config=cls._make_config(max_tokens),
            quant_config=quant_config.FusedMoEQuantConfig(quant_dtype=None),
        )

    def setUp(self) -> None:
        self.router = self._make_router(NUM_TOKENS)
        torch.manual_seed(0)
        self.a1 = torch.arange(NUM_TOKENS * HIDDEN, dtype=torch.float32).view(
            NUM_TOKENS, HIDDEN
        )
        # Boundary ids: LOCAL_LO-1 is the last non-local id, LOCAL_LO the first
        # local one. Local expert 0 stays empty so its placeholder row is poison.
        self.topk_ids = torch.tensor(
            [(LOCAL_LO - 1, LOCAL_LO + 1), (LOCAL_LO + 1, EXPERT_NUM - 1)]
            + [(0, LOCAL_LO + 2), (0, LOCAL_LO - 1)]
            + [(EXPERT_NUM - 1, LOCAL_LO + 1), (LOCAL_LO + 2, 1)],
            dtype=torch.int32,
        )
        self.topk_weights = torch.rand(NUM_TOKENS, TOP_K) + 0.5
        self.payload = self._prepare(self.a1, self.topk_weights, self.topk_ids)
        meta = self.payload.expert_tokens_meta
        assert meta is not None and meta.expert_num_tokens is not None
        self.counts = meta.expert_num_tokens

    def _prepare(
        self, a1: torch.Tensor, weights: torch.Tensor, ids: torch.Tensor
    ) -> moe_defs.ExpertForwardPayload:
        return self.router.prepare(a1, None, None, weights, ids)

    def _finalize(
        self,
        expert_output: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        ids: Optional[torch.Tensor] = None,
        prepared: Optional[moe_defs.ExpertForwardPayload] = None,
    ) -> torch.Tensor:
        prepared = self.payload if prepared is None else prepared
        reducer = mock.MagicMock(side_effect=lambda t, _g: t)
        with mock.patch.object(batched_data_router, "all_reduce", reducer):
            result = self.router.finalize(
                moe_defs.CombineForwardPayload(
                    fused_expert_output=expert_output,
                    router_context=prepared.router_context,
                ),
                self.topk_weights if weights is None else weights,
                self.topk_ids if ids is None else ids,
                False,
                None,
            )
        reducer.assert_called_once()
        self.assertIs(reducer.call_args.args[1], Group.TP)
        return result

    def _assert_round_trip(
        self,
        payload: moe_defs.ExpertForwardPayload,
        a1: torch.Tensor,
        ids: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Pack ``expert_output`` from the plan, poison every padding tail with a
        non-finite value, finalize, and compare against the reference weighted
        sum. The poison proves rows past each expert's count are never gathered,
        and a fully non-local token must come out exactly zero -- masking has to
        precede the weight multiply, else NaN * 0 == NaN."""
        meta = payload.expert_tokens_meta
        assert meta is not None and meta.expert_num_tokens is not None
        expert_output = torch.zeros(LOCAL_EXPERTS, a1.size(0), HIDDEN)
        for e in range(LOCAL_EXPERTS):
            tokens = (ids == e + LOCAL_LO).any(dim=1).nonzero().flatten()
            self.assertEqual(
                int(meta.expert_num_tokens[e]), tokens.numel(), f"expert {e} count"
            )
            live = tokens.numel()
            expert_output[e, :live] = payload.expert_x[e, :live]
            expert_output[e, live:] = float("inf") if e % 2 else float("nan")

        out = self._finalize(expert_output, weights, ids, payload)
        self.assertTrue(torch.isfinite(out).all(), "padding leaked into the output")
        local = (ids >= LOCAL_LO) & (ids < EXPERT_NUM)
        torch.testing.assert_close(out, a1 * (local * weights).sum(1, keepdim=True))
        return out

    def test_plan_packs_every_local_slot_exactly_once(self) -> None:
        for e in range(LOCAL_EXPERTS):
            tokens = (self.topk_ids == e + LOCAL_LO).any(dim=1).nonzero().flatten()
            self.assertEqual(int(self.counts[e]), tokens.numel(), f"expert {e} count")
            # Exact, not a multiset compare: packed row ranks fix token order
            # inside each expert block and finalize relies on that order.
            packed = self.payload.expert_x[e, : tokens.numel()]
            self.assertTrue(torch.equal(packed, self.a1[tokens]), f"expert {e} rows")

    def test_finalize_ignores_poisoned_padding(self) -> None:
        self._assert_round_trip(self.payload, self.a1, self.topk_ids, self.topk_weights)

    def test_prepared_payloads_are_independent(self) -> None:
        a1 = torch.arange(3 * HIDDEN, dtype=torch.float32).view(3, HIDDEN)
        ids = torch.tensor(
            [(LOCAL_LO, 0), (LOCAL_LO + 1, LOCAL_LO + 2), (1, EXPERT_NUM - 1)],
            dtype=torch.int32,
        )
        weights = torch.rand(3, TOP_K) + 0.5
        self._assert_round_trip(self._prepare(a1, weights, ids), a1, ids, weights)
        # The setUp fixture must survive an unrelated prepare/finalize: routing
        # state travels on the payload, not on the router instance.
        self._assert_round_trip(self.payload, self.a1, self.topk_ids, self.topk_weights)

    def test_finalize_rejects_missing_or_foreign_context(self) -> None:
        expert_output = self.payload.expert_x.clone()
        for router_context in (None, object()):
            with self.subTest(router_context=router_context), self.assertRaisesRegex(
                TypeError, "prepared routing context"
            ):
                self.router.finalize(
                    moe_defs.CombineForwardPayload(
                        fused_expert_output=expert_output,
                        router_context=router_context,
                    ),
                    self.topk_weights,
                    self.topk_ids,
                    False,
                    None,
                )

    def test_round_trip_matches_reference_across_token_counts(self) -> None:
        """Cover singleton input and prefill beyond the configured decode capacity."""
        for n in (1, 33):
            with self.subTest(num_tokens=n):
                router = self._make_router(1)
                a1 = torch.randn(n, HIDDEN)
                ids = torch.randint(0, EXPERT_NUM, (n, TOP_K), dtype=torch.int32)
                weights = torch.rand(n, TOP_K) + 0.5
                self._assert_round_trip(
                    router.prepare(a1, None, None, weights, ids), a1, ids, weights
                )

    def test_zero_tokens_round_trips(self) -> None:
        """Empty batch is reachable on a DP rank; deriving the counts from the
        last cumsum row used to raise IndexError."""
        empty = torch.zeros((0, TOP_K))
        payload = self._prepare(torch.zeros((0, HIDDEN)), empty, empty.to(torch.int32))
        meta = payload.expert_tokens_meta
        assert meta is not None and meta.expert_num_tokens is not None

        self.assertEqual(payload.expert_x.shape, (LOCAL_EXPERTS, 0, HIDDEN))
        self.assertTrue(
            torch.equal(
                meta.expert_num_tokens, torch.zeros(LOCAL_EXPERTS, dtype=torch.int32)
            )
        )
        out = self._finalize(
            torch.zeros((LOCAL_EXPERTS, 0, HIDDEN)),
            empty,
            empty.to(torch.int32),
            payload,
        )
        self.assertEqual(out.shape, (0, HIDDEN))

    def test_zero_tokens_round_trip_through_fused_moe(self) -> None:
        experts = mock.Mock()
        fused_moe = moe_defs.FusedMoe(self.router, experts, EXPERT_NUM)
        empty = torch.zeros((0, TOP_K))
        with mock.patch.object(batched_data_router, "all_reduce", lambda t, _: t):
            out = fused_moe(torch.zeros((0, HIDDEN)), empty, empty.to(torch.int32))
        experts.execute.assert_not_called()
        self.assertEqual(out.shape, (0, HIDDEN))


if __name__ == "__main__":
    unittest.main()
