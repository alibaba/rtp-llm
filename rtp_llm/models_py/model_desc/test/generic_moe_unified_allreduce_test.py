"""CPU regression tests for routed/shared unified TP all-reduce."""

import os
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch

import torch
from torch import nn

from rtp_llm.models_py.distributed.collective_torch import Group
from rtp_llm.models_py.model_desc.generic_moe import (
    GenericMoeLayer,
    _moe_unified_allreduce_enabled,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    CombineForwardPayload,
    DeferredOutputReduction,
    ExpertForwardPayload,
    FusedMoe,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.common.router.batched_data_router import (
    BatchedDataRouter,
)


class _Gate(nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.zeros(hidden_states.size(0), 2, dtype=hidden_states.dtype)


class _SelectTopK(nn.Module):
    def forward(self, scores, topk_ids, topk_weights):
        topk_ids.zero_()
        topk_weights.fill_(1.0)


class _FusedMoe(nn.Module):
    topk_ids_dtype = torch.int64

    def __init__(
        self,
        output: torch.Tensor,
        reduction: DeferredOutputReduction | None,
    ) -> None:
        super().__init__()
        self.output = output
        self.deferred_output_reduction = reduction
        self.defer_calls: list[bool] = []

    def forward(self, **kwargs) -> torch.Tensor:
        self.defer_calls.append(kwargs["defer_output_reduction"])
        return self.output.clone()


class _SharedExpert(nn.Module):
    def __init__(self, output: torch.Tensor) -> None:
        super().__init__()
        self.output = output
        self.skip_calls: list[bool] = []

    def forward(self, hidden_states, skip_allreduce=False):
        self.skip_calls.append(skip_allreduce)
        return self.output.clone()


class _SharedGate(nn.Module):
    def __init__(self, output: torch.Tensor) -> None:
        super().__init__()
        self.output = output

    def forward(self, hidden_states):
        return self.output.clone()


class _SigmoidGateScaleAdd(nn.Module):
    def forward(self, gate, shared, routed):
        routed.add_(torch.sigmoid(gate) * shared)
        return routed


class _Router:
    def __init__(self, reduction):
        self.deferred_output_reduction = reduction
        self.defer_calls = []
        self.config = SimpleNamespace(
            enable_cuda_graph=False,
            parallelism_config=SimpleNamespace(world_rank=0),
            ep_size=1,
            moe_k=1,
        )

    def prepare(self, a1, a1_scale, a2_scale, topk_weights, topk_ids):
        return ExpertForwardPayload(
            expert_x=a1,
            expert_topk_ids=topk_ids,
            expert_topk_weights=topk_weights,
        )

    def finalize(
        self,
        payload,
        topk_weights,
        topk_ids,
        apply_router_weight_on_input,
        extra_finalize_args,
        defer_output_reduction=False,
    ):
        self.defer_calls.append(defer_output_reduction)
        return payload.fused_expert_output


class _LegacyRouter(_Router):
    """Router using the pre-deferred-reduction finalize ABI."""

    def __init__(self):
        super().__init__(reduction=None)

    def finalize(
        self,
        payload,
        topk_weights,
        topk_ids,
        apply_router_weight_on_input,
        extra_finalize_args,
    ):
        self.defer_calls.append(False)
        return payload.fused_expert_output


class _Executor:
    topk_ids_dtype = torch.int64

    def execute(
        self,
        payload,
        activation,
        expert_map,
        a2_scale,
        apply_router_weight_on_input,
        extra_expert_args,
    ):
        return CombineForwardPayload(payload.expert_x + 1)


def _make_layer(
    routed: torch.Tensor,
    shared: torch.Tensor,
    *,
    ep_size: int,
    reduction: DeferredOutputReduction | None,
    gate: torch.Tensor | None,
    ffn_tp_size: int = 2,
    ffn_tp_rank: int = 0,
    enable_unified: bool = True,
) -> GenericMoeLayer:
    layer = GenericMoeLayer.__new__(GenericMoeLayer)
    nn.Module.__init__(layer)
    layer.hidden_dim = routed.size(-1)
    layer.top_k = 1
    layer.gate = _Gate()
    layer.select_topk = _SelectTopK()
    layer.correction_bias = None
    layer.fake_balance_expert = None
    layer.fused_moe = _FusedMoe(routed, reduction)
    layer.shared_expert = _SharedExpert(shared)
    layer.shared_expert_gate = _SharedGate(gate) if gate is not None else None
    layer.sigmoid_gate_scale_add = _SigmoidGateScaleAdd()
    layer.ffn_tp_size = ffn_tp_size
    layer.ffn_tp_rank = ffn_tp_rank
    layer.ep_size = ep_size
    layer.enable_unified_output_allreduce = enable_unified
    return layer


class GenericMoeUnifiedAllreduceTest(TestCase):
    def setUp(self) -> None:
        self.hidden = torch.randn(3, 4)
        self.routed = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0], [3.0, 4.0, 5.0, 6.0]]
        )
        self.shared = torch.tensor(
            [[0.5, 1.0, 1.5, 2.0], [1.0, 1.5, 2.0, 2.5], [1.5, 2.0, 2.5, 3.0]]
        )
        self.gate = torch.tensor([[0.0], [1.0], [-1.0]])
        self.reduction = DeferredOutputReduction(Group.TP, 2, 0)

    def test_env_defaults_off_and_explicit_true_enables(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(_moe_unified_allreduce_enabled())

        for value in ("1", "true", "yes", "on", "TRUE"):
            with self.subTest(value=value), patch.dict(
                os.environ,
                {"ENABLE_MOE_UNIFIED_ALLREDUCE": value},
                clear=True,
            ):
                self.assertTrue(_moe_unified_allreduce_enabled())

        with patch.dict(os.environ, {"ENABLE_MOE_UNIFIED_ALLREDUCE": "0"}, clear=True):
            self.assertFalse(_moe_unified_allreduce_enabled())

    def _assert_unified(self, ep_size: int, gate: torch.Tensor | None) -> None:
        layer = _make_layer(
            self.routed,
            self.shared,
            ep_size=ep_size,
            reduction=self.reduction,
            gate=gate,
        )
        reduced_inputs = []

        def fake_all_reduce(tensor, group):
            self.assertEqual(group, Group.TP)
            reduced_inputs.append(tensor.clone())
            return tensor * 2

        with patch(
            "rtp_llm.models_py.model_desc.generic_moe.all_reduce",
            side_effect=fake_all_reduce,
        ) as all_reduce:
            output = layer(self.hidden)

        expected_local = self.routed.clone()
        if gate is None:
            expected_local += self.shared
        else:
            expected_local += torch.sigmoid(gate) * self.shared
        torch.testing.assert_close(reduced_inputs[0], expected_local)
        torch.testing.assert_close(output, expected_local * 2)
        self.assertEqual(all_reduce.call_count, 1)
        self.assertEqual(layer.fused_moe.defer_calls, [True])
        self.assertEqual(layer.shared_expert.skip_calls, [True])

    def test_ep1_gated_outputs_share_one_allreduce(self):
        self._assert_unified(ep_size=1, gate=self.gate)

    def test_ep2_gated_outputs_share_one_allreduce(self):
        self._assert_unified(ep_size=2, gate=self.gate)

    def test_ungated_outputs_share_one_allreduce(self):
        self._assert_unified(ep_size=2, gate=None)

    def test_dispatch_router_keeps_shared_only_allreduce(self):
        layer = _make_layer(
            self.routed,
            self.shared,
            ep_size=2,
            reduction=None,
            gate=self.gate,
        )
        reduced_inputs = []

        def fake_all_reduce(tensor, group):
            reduced_inputs.append(tensor.clone())
            return tensor * 2

        with patch(
            "rtp_llm.models_py.model_desc.generic_moe.all_reduce",
            side_effect=fake_all_reduce,
        ) as all_reduce:
            output = layer(self.hidden)

        shared_contribution = torch.sigmoid(self.gate) * self.shared
        torch.testing.assert_close(reduced_inputs[0], shared_contribution)
        torch.testing.assert_close(output, self.routed + 2 * shared_contribution)
        self.assertEqual(all_reduce.call_count, 1)
        self.assertEqual(layer.fused_moe.defer_calls, [False])
        self.assertEqual(layer.shared_expert.skip_calls, [True])

    def test_ungated_dispatch_router_keeps_shared_only_allreduce(self):
        layer = _make_layer(
            self.routed,
            self.shared,
            ep_size=2,
            reduction=None,
            gate=None,
        )

        with patch(
            "rtp_llm.models_py.model_desc.generic_moe.all_reduce",
            side_effect=lambda tensor, group: tensor * 2,
        ) as all_reduce:
            output = layer(self.hidden)

        torch.testing.assert_close(output, self.routed + 2 * self.shared)
        self.assertEqual(all_reduce.call_count, 1)
        self.assertEqual(layer.fused_moe.defer_calls, [False])
        self.assertEqual(layer.shared_expert.skip_calls, [True])

    def test_partition_mismatch_does_not_unify(self):
        layer = _make_layer(
            self.routed,
            self.shared,
            ep_size=1,
            reduction=DeferredOutputReduction(Group.TP, 4, 0),
            gate=None,
        )
        self.assertFalse(layer._use_unified_output_allreduce())

    def test_rank_mismatch_does_not_unify(self):
        layer = _make_layer(
            self.routed,
            self.shared,
            ep_size=1,
            reduction=DeferredOutputReduction(Group.TP, 2, 1),
            gate=None,
        )
        self.assertFalse(layer._use_unified_output_allreduce())

    def test_group_mismatch_uses_legacy_tp_fallback(self):
        layer = _make_layer(
            self.routed,
            self.shared,
            ep_size=1,
            reduction=DeferredOutputReduction(Group.DP, 2, 0),
            gate=None,
        )

        with patch("rtp_llm.models_py.model_desc.generic_moe.all_reduce") as all_reduce:
            output = layer(self.hidden)

        torch.testing.assert_close(output, self.routed + self.shared)
        all_reduce.assert_not_called()
        self.assertEqual(layer.fused_moe.defer_calls, [False])
        self.assertEqual(layer.shared_expert.skip_calls, [False])

    def test_ffn_tp1_uses_legacy_fallback(self):
        layer = _make_layer(
            self.routed,
            self.shared,
            ep_size=1,
            reduction=DeferredOutputReduction(Group.TP, 1, 0),
            gate=None,
            ffn_tp_size=1,
        )

        with patch("rtp_llm.models_py.model_desc.generic_moe.all_reduce") as all_reduce:
            output = layer(self.hidden)

        torch.testing.assert_close(output, self.routed + self.shared)
        all_reduce.assert_not_called()
        self.assertEqual(layer.fused_moe.defer_calls, [False])
        self.assertEqual(layer.shared_expert.skip_calls, [False])

    def test_no_shared_expert_keeps_routed_finalize_enabled(self):
        layer = _make_layer(
            self.routed,
            self.shared,
            ep_size=1,
            reduction=self.reduction,
            gate=None,
        )
        layer.shared_expert = None
        layer.shared_expert_gate = None
        layer.sigmoid_gate_scale_add = None

        with patch("rtp_llm.models_py.model_desc.generic_moe.all_reduce") as all_reduce:
            output = layer(self.hidden)

        torch.testing.assert_close(output, self.routed)
        all_reduce.assert_not_called()
        self.assertEqual(layer.fused_moe.defer_calls, [False])

    def test_kill_switch_uses_legacy_path(self):
        layer = _make_layer(
            self.routed,
            self.shared,
            ep_size=1,
            reduction=self.reduction,
            gate=None,
            enable_unified=False,
        )
        self.assertFalse(layer._use_unified_output_allreduce())

        with patch("rtp_llm.models_py.model_desc.generic_moe.all_reduce") as all_reduce:
            output = layer(self.hidden)
        torch.testing.assert_close(output, self.routed + self.shared)
        all_reduce.assert_not_called()
        self.assertEqual(layer.fused_moe.defer_calls, [False])
        self.assertEqual(layer.shared_expert.skip_calls, [False])


class FusedMoeDeferredOutputTest(TestCase):
    def test_defer_false_preserves_legacy_router_finalize_abi(self):
        router = _LegacyRouter()
        fused_moe = FusedMoe(router, _Executor(), expert_num=2)
        hidden = torch.randn(3, 4)

        output = fused_moe(
            hidden_states=hidden,
            topk_weights=torch.ones(3, 1),
            topk_ids=torch.zeros(3, 1, dtype=torch.int64),
        )

        torch.testing.assert_close(output, hidden + 1)
        self.assertEqual(router.defer_calls, [False])

    def test_forwards_defer_request_only_to_capable_router(self):
        router = _Router(DeferredOutputReduction(Group.TP, 2, 0))
        fused_moe = FusedMoe(router, _Executor(), expert_num=2)
        hidden = torch.randn(3, 4)
        caller_args = {"sentinel": 1}

        output = fused_moe(
            hidden_states=hidden,
            topk_weights=torch.ones(3, 1),
            topk_ids=torch.zeros(3, 1, dtype=torch.int64),
            extra_finalize_args=caller_args,
            defer_output_reduction=True,
        )

        torch.testing.assert_close(output, hidden + 1)
        self.assertEqual(router.defer_calls, [True])
        self.assertEqual(caller_args, {"sentinel": 1})

    def test_rejects_defer_request_for_dispatch_router(self):
        fused_moe = FusedMoe(_Router(None), _Executor(), expert_num=2)

        with self.assertRaisesRegex(ValueError, "does not expose"):
            fused_moe(
                hidden_states=torch.randn(3, 4),
                topk_weights=torch.ones(3, 1),
                topk_ids=torch.zeros(3, 1, dtype=torch.int64),
                defer_output_reduction=True,
            )


class DispatchRouterCapabilityTest(TestCase):
    def test_deepep_and_mori_keep_default_non_deferred_capability(self):
        from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.routers.deepep_normal_router import (
            DeepepNormalRouter,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.routers.mori_ep_intranode_router import (
            MoriEpIntranodeRouter,
        )

        for router_cls in (DeepepNormalRouter, MoriEpIntranodeRouter):
            with self.subTest(router=router_cls.__name__):
                router = router_cls.__new__(router_cls)
                self.assertIsNone(router.deferred_output_reduction)


class BatchedDataDeferredOutputTest(TestCase):
    @patch(
        "rtp_llm.models_py.modules.factory.fused_moe.impl.common.router."
        "batched_data_router.all_reduce"
    )
    @patch(
        "rtp_llm.models_py.modules.factory.fused_moe.impl.common.router."
        "batched_data_router.TopKWeightAndReduceNaiveBatched.apply"
    )
    def test_finalize_can_return_local_partial(self, reduce_apply, all_reduce):
        router = BatchedDataRouter.__new__(BatchedDataRouter)
        router.ep_rank = 0
        router.tp_size = 2
        router.tp_rank = 1
        local_partial = torch.randn(3, 4)
        reduce_apply.return_value = local_partial

        result = router.finalize(
            payload=CombineForwardPayload(torch.randn(2, 3, 4)),
            topk_weights=torch.ones(3, 1),
            topk_ids=torch.zeros(3, 1, dtype=torch.int64),
            apply_router_weight_on_input=False,
            extra_finalize_args=None,
            defer_output_reduction=True,
        )

        self.assertIs(result, local_partial)
        all_reduce.assert_not_called()
        self.assertEqual(
            router.deferred_output_reduction,
            DeferredOutputReduction(Group.TP, 2, 1),
        )

    @patch(
        "rtp_llm.models_py.modules.factory.fused_moe.impl.common.router."
        "batched_data_router.all_reduce"
    )
    @patch(
        "rtp_llm.models_py.modules.factory.fused_moe.impl.common.router."
        "batched_data_router.TopKWeightAndReduceNaiveBatched.apply"
    )
    def test_finalize_default_keeps_legacy_allreduce(self, reduce_apply, all_reduce):
        router = BatchedDataRouter.__new__(BatchedDataRouter)
        router.ep_rank = 0
        router.tp_size = 2
        router.tp_rank = 0
        local_partial = torch.randn(3, 4)
        reduced = torch.randn(3, 4)
        reduce_apply.return_value = local_partial
        all_reduce.return_value = reduced

        result = router.finalize(
            payload=CombineForwardPayload(torch.randn(2, 3, 4)),
            topk_weights=torch.ones(3, 1),
            topk_ids=torch.zeros(3, 1, dtype=torch.int64),
            apply_router_weight_on_input=False,
            extra_finalize_args=None,
        )

        self.assertIs(result, reduced)
        all_reduce.assert_called_once_with(local_partial, Group.TP)


class CudaPureTpDeferredOutputTest(TestCase):
    def test_descriptor_and_deferred_finalize_match_cuda_tp_group(self):
        # The ROCm test build intentionally omits this CUDA-only binding.  Stub
        # it only while importing the otherwise device-independent router.
        from rtp_llm.ops import compute_ops

        with patch.object(compute_ops, "trt_fp8_quantize_128", create=True), patch(
            "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers."
            "pure_tp_router.all_reduce"
        ) as all_reduce:
            from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.pure_tp_router import (
                PureTpRouterNoQuant,
            )

            router = PureTpRouterNoQuant.__new__(PureTpRouterNoQuant)
            router.tp_size = 2
            router.tp_rank = 1
            local_partial = torch.randn(3, 4)
            payload = CombineForwardPayload(local_partial)

            deferred = router.finalize(
                payload,
                torch.ones(3, 1),
                torch.zeros(3, 1, dtype=torch.int64),
                False,
                None,
                defer_output_reduction=True,
            )

            self.assertIs(deferred, local_partial)
            all_reduce.assert_not_called()
            self.assertEqual(
                router.deferred_output_reduction,
                DeferredOutputReduction(Group.TP, 2, 1),
            )

            reduced = torch.randn(3, 4)
            all_reduce.return_value = reduced
            legacy = router.finalize(
                payload,
                torch.ones(3, 1),
                torch.zeros(3, 1, dtype=torch.int64),
                False,
                None,
            )
            self.assertIs(legacy, reduced)
            all_reduce.assert_called_once_with(local_partial, group=Group.TP)


if __name__ == "__main__":
    main()
