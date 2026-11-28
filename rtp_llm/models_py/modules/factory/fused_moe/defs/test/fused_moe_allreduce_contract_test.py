"""Platform-neutral FusedMoe TP all-reduce contract tests."""

from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import Mock

import torch

from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    SKIP_TP_ALLREDUCE_ARG,
    CombineForwardPayload,
    ExpertForwardPayload,
    ExpertTokensMetadata,
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

    def test_monitor_context_reaches_executor_without_router_double_record(self):
        fused_moe, router, experts, hidden_states, weights, ids = self._make_fused_moe(
            False
        )
        context = SimpleNamespace(expert_token_counts=torch.zeros(4, dtype=torch.int32))
        fused_moe(hidden_states, weights, ids, expert_load=context)
        self.assertIs(experts.execute.call_args.args[0].expert_load, context)
        fused_moe(hidden_states, weights, ids, expert_load=None)
        self.assertIsNone(experts.execute.call_args.args[0].expert_load)

    def test_executor_without_monitor_argument_remains_compatible(self):
        fused_moe, router, experts, hidden_states, weights, ids = self._make_fused_moe(
            False
        )

        def execute(
            payload,
            activation,
            expert_map,
            a2_scale,
            apply_router_weight_on_input,
            extra_expert_args,
        ):
            return CombineForwardPayload(fused_expert_output=payload.expert_x)

        experts.execute.side_effect = execute
        context = SimpleNamespace(expert_token_counts=torch.zeros(4, dtype=torch.int32))
        for expert_load in (None, context):
            with self.subTest(monitor_enabled=expert_load is not None):
                result = fused_moe(hidden_states, weights, ids, expert_load=expert_load)
                torch.testing.assert_close(result, hidden_states)
        self.assertEqual(context.expert_token_counts.tolist(), [0] * 4)

    def test_backend_without_recording_keeps_zero_counts(self):
        from rtp_llm.models_py.utils.expert_load_monitor import ExpertLoadContext

        fused_moe, router, experts, hidden_states, weights, ids = self._make_fused_moe(
            False
        )
        context = ExpertLoadContext(torch.zeros(4, dtype=torch.int32))
        result = fused_moe(hidden_states, weights, ids, expert_load=context)
        torch.testing.assert_close(result, hidden_states)
        self.assertEqual(context.expert_token_counts.tolist(), [0] * 4)
        router.prepare.assert_called_once()
        experts.execute.assert_called_once()

    def test_dispatch_counts_are_not_recorded_again_by_executor(self):
        from rtp_llm.models_py.utils.expert_load_monitor import ExpertLoadContext

        fused_moe, router, experts, hidden_states, weights, ids = self._make_fused_moe(
            False
        )
        router.prepare.return_value.expert_tokens_meta = ExpertTokensMetadata(
            expert_num_tokens=torch.tensor([2, 0, 3, 1], dtype=torch.int32),
        )
        context = ExpertLoadContext(torch.zeros(4, dtype=torch.int32))
        fused_moe(hidden_states, weights, ids, expert_load=context)
        self.assertEqual(context.expert_token_counts.tolist(), [2, 0, 3, 1])
        self.assertIsNone(experts.execute.call_args.args[0].expert_load)

    def test_padded_dispatch_counts_are_recorded_by_executor(self):
        fused_moe, router, experts, hidden_states, weights, ids = self._make_fused_moe(
            False
        )
        router.prepare.return_value.expert_tokens_meta = ExpertTokensMetadata(
            expert_num_tokens=torch.full((4,), 128, dtype=torch.int32),
            counts_are_padded=True,
        )
        context = SimpleNamespace(expert_token_counts=torch.zeros(4, dtype=torch.int32))
        fused_moe(hidden_states, weights, ids, expert_load=context)
        self.assertIs(experts.execute.call_args.args[0].expert_load, context)

    def test_empty_executor_input_preserves_prior_counts(self):
        from rtp_llm.models_py.utils.expert_load_monitor import ExpertLoadContext

        fused_moe, router, experts, hidden_states, weights, ids = self._make_fused_moe(
            False
        )
        experts.execute_empty_inputs = False
        router.prepare.return_value.expert_x = hidden_states[:0]
        router.finalize.return_value = hidden_states[:0]
        context = ExpertLoadContext(torch.full((4,), 9, dtype=torch.int32))
        fused_moe(hidden_states[:0], weights[:0], ids[:0], expert_load=context)
        self.assertEqual(context.expert_token_counts.tolist(), [9] * 4)
        experts.execute.assert_not_called()

    def test_forward_passes_skip_tp_allreduce_to_supported_router(self):
        fused_moe, router, _, hidden_states, topk_weights, topk_ids = (
            self._make_fused_moe(True)
        )
        fused_moe(
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            skip_tp_allreduce=True,
            expert_load=None,
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
                    expert_load=None,
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
            expert_load=None,
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
            expert_load=None,
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
            expert_load=None,
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
                expert_load=None,
            )

        router.prepare.assert_not_called()
        experts.execute.assert_not_called()
        router.finalize.assert_not_called()

    def test_gate_pack_rejects_skip_before_dispatch(self):
        fused_moe, router, experts, hidden_states, _, _ = self._make_fused_moe(False)
        router.supports_gate_pack = True
        experts.supports_gate_pack = True
        with self.assertRaisesRegex(ValueError, "supports_skip_tp_allreduce"):
            fused_moe.forward_gate_pack(
                hidden_states, Mock(), skip_tp_allreduce=True, expert_load=None
            )

        router.prepare_gate_pack.assert_not_called()
        experts.execute.assert_not_called()
        router.finalize.assert_not_called()

    def test_gate_pack_passes_skip_to_supported_router(self):
        fused_moe, router, experts, hidden_states, _, _ = self._make_fused_moe(True)
        router.supports_gate_pack = True
        experts.supports_gate_pack = True
        router.prepare_gate_pack.return_value = router.prepare.return_value
        fused_moe.forward_gate_pack(
            hidden_states, Mock(), skip_tp_allreduce=True, expert_load=None
        )

        router.prepare_gate_pack.assert_called_once()
        experts.execute.assert_called_once()
        self.assertTrue(
            _extract_extra_finalize_args(router.finalize)[SKIP_TP_ALLREDUCE_ARG]
        )


if __name__ == "__main__":
    main()
