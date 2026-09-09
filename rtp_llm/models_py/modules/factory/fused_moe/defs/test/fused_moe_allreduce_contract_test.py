"""Platform-neutral FusedMoe TP all-reduce contract tests."""

from unittest import TestCase, main
from unittest.mock import Mock

import torch

from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    SKIP_TP_ALLREDUCE_ARG,
    CombineForwardPayload,
    ExpertForwardPayload,
    FusedMoe,
    FusedMoeDataRouter,
    FusedMoeExpertExecutor,
)


def _extract_extra_finalize_args(router_finalize_mock):
    args = router_finalize_mock.call_args.args
    if len(args) != 5:
        raise AssertionError(
            "router.finalize must receive the five positional finalize arguments"
        )
    return args[4]


class FusedMoeSkipAllreduceTest(TestCase):
    def _make_fused_moe(self, supports_skip):
        hidden_states = torch.randn(4, 8)
        topk_ids = torch.zeros(4, 2, dtype=torch.int32)
        topk_weights = torch.ones(4, 2, dtype=torch.float32)

        router = Mock(spec=FusedMoeDataRouter)
        router.supports_skip_tp_allreduce = supports_skip
        router.prepare.return_value = ExpertForwardPayload(
            expert_x=hidden_states,
            expert_topk_ids=topk_ids,
            expert_topk_weights=topk_weights,
        )
        experts = Mock(spec=FusedMoeExpertExecutor)
        experts.execute.return_value = CombineForwardPayload(
            fused_expert_output=hidden_states.clone()
        )
        router.finalize.return_value = hidden_states.clone()
        return (
            FusedMoe(router, experts, expert_num=8),
            router,
            experts,
            hidden_states,
            topk_weights,
            topk_ids,
        )

    def test_forward_passes_skip_tp_allreduce_to_supported_router(self):
        fused_moe, router, _, hidden_states, topk_weights, topk_ids = (
            self._make_fused_moe(True)
        )
        fused_moe(
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            skip_tp_allreduce=True,
        )

        extra_finalize_args = _extract_extra_finalize_args(router.finalize)
        self.assertTrue(extra_finalize_args[SKIP_TP_ALLREDUCE_ARG])

    def test_forward_defaults_skip_tp_allreduce_false_for_all_routers(self):
        for supports_skip in (True, False):
            with self.subTest(supports_skip=supports_skip):
                fused_moe, router, _, hidden_states, topk_weights, topk_ids = (
                    self._make_fused_moe(supports_skip)
                )
                fused_moe(
                    hidden_states=hidden_states,
                    topk_weights=topk_weights,
                    topk_ids=topk_ids,
                )

                extra_finalize_args = _extract_extra_finalize_args(router.finalize)
                self.assertFalse(extra_finalize_args[SKIP_TP_ALLREDUCE_ARG])

    def test_forward_overrides_conflicting_finalize_skip_key(self):
        fused_moe, router, _, hidden_states, topk_weights, topk_ids = (
            self._make_fused_moe(True)
        )
        extra_finalize_args = {SKIP_TP_ALLREDUCE_ARG: True}
        fused_moe(
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            extra_finalize_args=extra_finalize_args,
            skip_tp_allreduce=False,
        )

        self.assertTrue(extra_finalize_args[SKIP_TP_ALLREDUCE_ARG])
        self.assertFalse(
            _extract_extra_finalize_args(router.finalize)[SKIP_TP_ALLREDUCE_ARG]
        )

    def test_unsupported_router_cannot_be_bypassed_by_finalize_key(self):
        fused_moe, router, experts, hidden_states, topk_weights, topk_ids = (
            self._make_fused_moe(False)
        )
        extra_finalize_args = {SKIP_TP_ALLREDUCE_ARG: True}
        fused_moe(
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            extra_finalize_args=extra_finalize_args,
            skip_tp_allreduce=False,
        )

        self.assertTrue(extra_finalize_args[SKIP_TP_ALLREDUCE_ARG])
        router.prepare.assert_called_once()
        experts.execute.assert_called_once()
        self.assertFalse(
            _extract_extra_finalize_args(router.finalize)[SKIP_TP_ALLREDUCE_ARG]
        )

    def test_forward_passes_router_context_to_finalize(self):
        fused_moe, router, _, hidden_states, topk_weights, topk_ids = (
            self._make_fused_moe(False)
        )
        router_context = object()
        router.prepare.return_value.router_context = router_context

        fused_moe(
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
        )

        self.assertIs(router.finalize.call_args.args[0].router_context, router_context)

    def test_forward_rejects_skip_tp_allreduce_for_unsupported_router(self):
        fused_moe, router, experts, hidden_states, topk_weights, topk_ids = (
            self._make_fused_moe(False)
        )
        with self.assertRaisesRegex(ValueError, "supports_skip_tp_allreduce"):
            fused_moe(
                hidden_states=hidden_states,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                skip_tp_allreduce=True,
            )

        router.prepare.assert_not_called()
        experts.execute.assert_not_called()


if __name__ == "__main__":
    main()
