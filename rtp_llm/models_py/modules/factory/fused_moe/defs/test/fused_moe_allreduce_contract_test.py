"""Platform-neutral FusedMoe TP all-reduce contract tests."""

from unittest import TestCase, main
from unittest.mock import Mock, patch

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
        router.max_inp_tokens = None
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
        router.finalize.assert_not_called()

    def test_chunked_forward_validates_skip_before_dispatch(self):
        fused_moe, router, experts, hidden_states, topk_weights, topk_ids = (
            self._make_fused_moe(False)
        )
        router.max_inp_tokens = 2

        with self.assertRaisesRegex(ValueError, "supports_skip_tp_allreduce"):
            fused_moe(
                hidden_states=hidden_states,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                skip_tp_allreduce=True,
            )

        router.prepare.assert_not_called()
        experts.execute.assert_not_called()
        router.finalize.assert_not_called()

    def test_chunked_forward_passes_skip_to_every_chunk(self):
        fused_moe, router, experts, hidden_states, topk_weights, topk_ids = (
            self._make_fused_moe(True)
        )
        router.max_inp_tokens = 2
        router.prepare.side_effect = lambda a1, _, __, weights, ids: (
            ExpertForwardPayload(
                expert_x=a1,
                expert_topk_ids=ids,
                expert_topk_weights=weights,
            )
        )
        experts.execute.side_effect = lambda payload, **_: CombineForwardPayload(
            fused_expert_output=payload.expert_x.clone()
        )
        router.finalize.side_effect = lambda payload, *_: payload.fused_expert_output

        with patch.object(
            torch.cuda, "is_current_stream_capturing", return_value=False
        ):
            output = fused_moe(
                hidden_states=hidden_states,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                skip_tp_allreduce=True,
            )

        torch.testing.assert_close(output, hidden_states)
        self.assertEqual(router.finalize.call_count, 2)
        for finalize_call in router.finalize.call_args_list:
            self.assertTrue(finalize_call.args[4][SKIP_TP_ALLREDUCE_ARG])

    def test_chunked_forward_slices_token_aligned_expert_args(self):
        fused_moe, router, experts, hidden_states, topk_weights, topk_ids = (
            self._make_fused_moe(False)
        )
        router.max_inp_tokens = 2
        router.prepare.side_effect = lambda a1, _, __, weights, ids: (
            ExpertForwardPayload(
                expert_x=a1,
                expert_topk_ids=ids,
                expert_topk_weights=weights,
            )
        )
        experts.execute.side_effect = lambda payload, **_: CombineForwardPayload(
            fused_expert_output=payload.expert_x.clone()
        )
        router.finalize.side_effect = lambda payload, *_: payload.fused_expert_output

        token_values = torch.arange(8).reshape(4, 2)
        scalar_tensor = torch.tensor(3)
        global_tensor = torch.arange(3)
        metadata = object()
        extra_expert_args = {
            "router_logits": token_values,
            "layer_idx": 7,
            "scalar_tensor": scalar_tensor,
            "global_tensor": global_tensor,
            "metadata": metadata,
        }

        with patch.object(
            torch.cuda, "is_current_stream_capturing", return_value=False
        ):
            output = fused_moe(
                hidden_states=hidden_states,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                extra_expert_args=extra_expert_args,
            )

        torch.testing.assert_close(output, hidden_states)
        self.assertEqual(experts.execute.call_count, 2)
        chunk_args = [
            call.kwargs["extra_expert_args"] for call in experts.execute.call_args_list
        ]
        torch.testing.assert_close(chunk_args[0]["router_logits"], token_values[:2])
        torch.testing.assert_close(chunk_args[1]["router_logits"], token_values[2:])
        self.assertIsNot(chunk_args[0], extra_expert_args)
        self.assertIsNot(chunk_args[1], extra_expert_args)
        self.assertIsNot(chunk_args[0], chunk_args[1])
        for args in chunk_args:
            self.assertEqual(args["layer_idx"], 7)
            self.assertIs(args["scalar_tensor"], scalar_tensor)
            self.assertIs(args["global_tensor"], global_tensor)
            self.assertIs(args["metadata"], metadata)
        self.assertIs(extra_expert_args["router_logits"], token_values)
        self.assertEqual(extra_expert_args["router_logits"].shape, (4, 2))

    def test_gate_pack_rejects_skip_before_dispatch(self):
        fused_moe, router, experts, hidden_states, _, _ = self._make_fused_moe(False)
        router.supports_gate_pack = True
        experts.supports_gate_pack = True
        with self.assertRaisesRegex(ValueError, "supports_skip_tp_allreduce"):
            fused_moe.forward_gate_pack(hidden_states, Mock(), skip_tp_allreduce=True)

        router.prepare_gate_pack.assert_not_called()
        experts.execute.assert_not_called()
        router.finalize.assert_not_called()

    def test_gate_pack_passes_skip_to_supported_router(self):
        fused_moe, router, experts, hidden_states, _, _ = self._make_fused_moe(True)
        router.supports_gate_pack = True
        experts.supports_gate_pack = True
        router.prepare_gate_pack.return_value = router.prepare.return_value
        fused_moe.forward_gate_pack(hidden_states, Mock(), skip_tp_allreduce=True)

        router.prepare_gate_pack.assert_called_once()
        experts.execute.assert_called_once()
        self.assertTrue(
            _extract_extra_finalize_args(router.finalize)[SKIP_TP_ALLREDUCE_ARG]
        )


if __name__ == "__main__":
    main()
