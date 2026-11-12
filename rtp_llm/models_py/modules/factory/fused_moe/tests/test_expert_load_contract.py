from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import Mock

import torch

from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    CombineForwardPayload,
    ExpertForwardPayload,
    ExpertTokensMetadata,
    FusedMoe,
    FusedMoeDataRouter,
    FusedMoeExpertExecutor,
)


class FusedMoeExpertLoadTest(TestCase):
    def _make_fused_moe(self):
        hidden_states = torch.randn(4, 8)
        topk_ids = torch.zeros(4, 2, dtype=torch.int32)
        topk_weights = torch.ones(4, 2, dtype=torch.float32)

        router = Mock(spec=FusedMoeDataRouter)
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
        fused_moe, router, experts, hidden_states, weights, ids = self._make_fused_moe()
        context = SimpleNamespace(expert_token_counts=torch.zeros(4, dtype=torch.int32))
        fused_moe(hidden_states, weights, ids, expert_load=context)
        self.assertIs(experts.execute.call_args.args[0].expert_load, context)
        fused_moe(hidden_states, weights, ids, expert_load=None)
        self.assertIsNone(experts.execute.call_args.args[0].expert_load)

    def test_native_gate_physical_ids_skip_mapping(self):
        fused_moe, router, experts, hidden_states, weights, ids = self._make_fused_moe()
        mapping = Mock()
        physical_ids = ids + 4
        mapping.map_ids.return_value = physical_ids
        fused_moe.eplb_layer = mapping
        fused_moe(hidden_states, weights, ids)
        mapping.map_ids.assert_called_once_with(ids)
        self.assertIs(router.prepare.call_args.args[-1], physical_ids)
        mapping.reset_mock()
        fused_moe(hidden_states, weights, physical_ids, ids_are_physical=True)
        mapping.map_ids.assert_not_called()
        self.assertIs(router.prepare.call_args.args[-1], physical_ids)

    def test_executor_without_monitor_argument_remains_compatible(self):
        fused_moe, router, experts, hidden_states, weights, ids = self._make_fused_moe()

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

        fused_moe, router, experts, hidden_states, weights, ids = self._make_fused_moe()
        context = ExpertLoadContext(torch.zeros(4, dtype=torch.int32))
        result = fused_moe(hidden_states, weights, ids, expert_load=context)
        torch.testing.assert_close(result, hidden_states)
        self.assertEqual(context.expert_token_counts.tolist(), [0] * 4)
        router.prepare.assert_called_once()
        experts.execute.assert_called_once()

    def test_dispatch_counts_are_not_recorded_again_by_executor(self):
        from rtp_llm.models_py.utils.expert_load_monitor import ExpertLoadContext

        fused_moe, router, experts, hidden_states, weights, ids = self._make_fused_moe()
        router.prepare.return_value.expert_tokens_meta = ExpertTokensMetadata(
            expert_num_tokens=torch.tensor([2, 0, 3, 1], dtype=torch.int32),
        )
        context = ExpertLoadContext(torch.zeros(4, dtype=torch.int32))
        fused_moe(hidden_states, weights, ids, expert_load=context)
        self.assertEqual(context.expert_token_counts.tolist(), [2, 0, 3, 1])
        self.assertIsNone(experts.execute.call_args.args[0].expert_load)

    def test_padded_dispatch_counts_are_recorded_by_executor(self):
        fused_moe, router, experts, hidden_states, weights, ids = self._make_fused_moe()
        router.prepare.return_value.expert_tokens_meta = ExpertTokensMetadata(
            expert_num_tokens=torch.full((4,), 128, dtype=torch.int32),
            counts_are_padded=True,
        )
        context = SimpleNamespace(expert_token_counts=torch.zeros(4, dtype=torch.int32))
        fused_moe(hidden_states, weights, ids, expert_load=context)
        self.assertIs(experts.execute.call_args.args[0].expert_load, context)

    def test_empty_executor_input_preserves_prior_counts(self):
        from rtp_llm.models_py.utils.expert_load_monitor import ExpertLoadContext

        fused_moe, router, experts, hidden_states, weights, ids = self._make_fused_moe()
        experts.execute_empty_inputs = False
        router.prepare.return_value.expert_x = hidden_states[:0]
        router.finalize.return_value = hidden_states[:0]
        context = ExpertLoadContext(torch.full((4,), 9, dtype=torch.int32))
        fused_moe(hidden_states[:0], weights[:0], ids[:0], expert_load=context)
        self.assertEqual(context.expert_token_counts.tolist(), [9] * 4)
        experts.execute.assert_not_called()


if __name__ == "__main__":
    main()
