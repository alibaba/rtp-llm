import os
import sys
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    ExpertForwardPayload,
    ExpertGatePayload,
    FusedMoe,
    MoeWorkspace,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.fp8_fp4_base import (
    Fp8Fp4ExecutorBase,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.fp8_fp4_router import (
    Fp8Fp4Router,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4 import (
    gate as gate_module,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4 import (
    layer as layer_module,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.gate import Gate
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.layer import (
    Fp8Fp4MoeLayer,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.shared_expert import (
    _run_shared_expert,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.sm120 import (
    workspace as workspace_module,
)
from rtp_llm.utils.model_weight import W


class _FakeGate:
    def __init__(self, events, gate_pack_enabled):
        self.events = events
        self.gate_pack_enabled = gate_pack_enabled

    def can_prepare_gate_payload(self, x, input_ids):
        self.events.append("gate.can_prepare")
        return self.gate_pack_enabled

    def prepare_gate_payload(self, x, input_ids):
        self.events.append("gate.prepare")
        return object()

    def __call__(self, x, input_ids):
        self.events.append("gate.forward")
        tokens = x.size(0)
        return (
            torch.ones((tokens, 1), dtype=torch.float32),
            torch.zeros((tokens, 1), dtype=torch.long),
        )


class _FakeFusedMoe:
    supports_gate_pack = True

    def __init__(self, events):
        self.events = events

    def forward_gate_pack(self, *, hidden_states, gate_payload, activation):
        self.events.append("fused_moe.forward_gate_pack")
        return torch.ones_like(hidden_states)

    def __call__(self, *, hidden_states, topk_weights, topk_ids, activation):
        self.events.append("fused_moe.forward")
        return torch.ones_like(hidden_states)


class _FakeSharedExecutor:
    def __init__(self, events):
        self.events = events
        self.x = None

    def start(self, shared_experts, x):
        self.events.append("shared.start")
        self.x = x

    def finish(self):
        self.events.append("shared.finish")
        return torch.full_like(self.x, getattr(self, "value", 1.0))


class _Fp32Executor(Fp8Fp4ExecutorBase):
    def setup_weights(self, weights):
        pass

    def forward(self, x, topk_weights, topk_ids):
        return torch.full_like(x, getattr(self, "value", 1.0), dtype=torch.float32)

    def forward_gate_pack(self, x, gate_payload):
        tokens = x.size(0)
        return (
            torch.full_like(x, getattr(self, "value", 1.0), dtype=torch.float32),
            torch.ones((tokens, 1), dtype=torch.float32),
            torch.zeros((tokens, 1), dtype=torch.long),
        )


class Dsv4GateSm120PolicyTest(unittest.TestCase):
    def _gate(self, *, hash_layer=False, sm120_dsv4_numerics=True):
        if hash_layer:
            layer_weights = {
                W.moe_gate: torch.zeros((4, 8), dtype=torch.float32),
                W.moe_gate_tid2eid: torch.tensor(
                    [[0, 1], [1, 0], [2, 3], [3, 2]], dtype=torch.long
                ),
            }
        else:
            layer_weights = {
                W.moe_gate: torch.zeros((4, 8), dtype=torch.float32),
                W.moe_gate_bias: torch.zeros((4,), dtype=torch.float32),
            }
        return Gate(
            layer_id=0,
            dim=8,
            n_routed_experts=4,
            n_activated_experts=2,
            score_func="sqrtsoftplus",
            n_hash_layers=1 if hash_layer else 0,
            vocab_size=4 if hash_layer else 0,
            layer_weights=layer_weights,
            sm120_dsv4_numerics=sm120_dsv4_numerics,
        )

    def test_policy_is_capability_and_sm120_scoped(self):
        dsv4_gate = self._gate()
        generic_gate = self._gate(sm120_dsv4_numerics=False)
        x_cuda = SimpleNamespace(is_cuda=True, device=torch.device("cuda", 0))
        x_cpu = torch.ones((2, 8), dtype=torch.bfloat16)
        with patch.object(gate_module, "is_sm120", return_value=True):
            self.assertTrue(dsv4_gate._sm120_dsv4_policy(x_cuda))
            self.assertFalse(generic_gate._sm120_dsv4_policy(x_cuda))
            self.assertFalse(dsv4_gate._sm120_dsv4_policy(x_cpu))
            dsv4_gate._ced_row_projection = lambda x, w: x
            self.assertFalse(dsv4_gate._sm120_dsv4_policy(x_cuda))
            del dsv4_gate._ced_row_projection
        with patch.object(gate_module, "is_sm120", return_value=False):
            self.assertFalse(dsv4_gate._sm120_dsv4_policy(x_cuda))

    def test_ced_projection_precedes_padding(self):
        gate = self._gate()
        calls = []

        def projection(x, weight):
            calls.append(x.shape[0])
            return torch.zeros((x.shape[0], 4), dtype=x.dtype)

        gate._ced_row_projection = projection
        x = torch.ones((2, 8), dtype=torch.bfloat16)
        input_ids = torch.tensor([0, 1], dtype=torch.long)
        with (
            patch.object(gate_module, "_use_fused_gate", return_value=False),
            patch.dict(os.environ, {"MOE_GATE_FP32": "0"}),
        ):
            gate(x, input_ids)
        # CED received the original two rows, never the small-M padded batch,
        # and the SM120 policy stays disabled while it is installed.
        self.assertEqual(calls, [2])
        self.assertFalse(gate._sm120_dsv4_policy(x))

    def test_gate_new_4row_cpu_forward_disables_sm120_policy(self):
        bare_gate = Gate.__new__(Gate)
        self.assertFalse(
            bare_gate._sm120_dsv4_policy(torch.ones((4, 8), dtype=torch.bfloat16))
        )


class Fp8Fp4MoeLayerSchedulingTest(unittest.TestCase):
    def _layer(self, events, gate_pack_enabled, with_shared=True):
        layer = Fp8Fp4MoeLayer.__new__(Fp8Fp4MoeLayer)
        torch.nn.Module.__init__(layer)
        layer.dim = 4
        layer.gate = _FakeGate(events, gate_pack_enabled)
        layer.fused_moe = _FakeFusedMoe(events)
        layer.shared_experts = object() if with_shared else None
        layer._shared_executor = _FakeSharedExecutor(events) if with_shared else None
        return layer

    @staticmethod
    def _combine(events):
        def combine(routed, shared, out_dtype, out=None):
            events.append("combine")
            result = (routed.float() + shared.float()).to(out_dtype)
            if out is None:
                return result
            out.copy_(result)
            return out

        return combine

    def test_gate_pack_starts_shared_expert_before_gate(self):
        events = []
        layer = self._layer(events, gate_pack_enabled=True)
        x = torch.ones((2, 4), dtype=torch.bfloat16)

        with patch.object(
            layer_module,
            "combine_routed_and_shared",
            side_effect=self._combine(events),
        ):
            output = layer(x, None)

        self.assertEqual(
            events,
            [
                "gate.can_prepare",
                "shared.start",
                "gate.prepare",
                "fused_moe.forward_gate_pack",
                "shared.finish",
                "combine",
            ],
        )
        torch.testing.assert_close(output, torch.full_like(x, 2))

    def test_ordinary_gate_stays_before_shared_expert(self):
        events = []
        layer = self._layer(events, gate_pack_enabled=False)
        x = torch.ones((2, 4), dtype=torch.bfloat16)

        with patch.object(
            layer_module,
            "combine_routed_and_shared",
            side_effect=self._combine(events),
        ):
            layer(x, None)

        self.assertEqual(
            events,
            [
                "gate.can_prepare",
                "gate.forward",
                "shared.start",
                "fused_moe.forward",
                "shared.finish",
                "combine",
            ],
        )

    def test_fused_shared_expert_keeps_gate_first(self):
        events = []
        layer = self._layer(events, gate_pack_enabled=True, with_shared=False)
        x = torch.ones((2, 4), dtype=torch.bfloat16)

        layer(x, None)

        self.assertEqual(
            events,
            [
                "gate.can_prepare",
                "gate.prepare",
                "fused_moe.forward_gate_pack",
            ],
        )

    def test_other_executors_keep_routed_output_rounding(self):
        events = []
        layer = self._layer(events, gate_pack_enabled=False)
        executor = _Fp32Executor.__new__(_Fp32Executor)
        torch.nn.Module.__init__(executor)
        executor.value = 1.003
        self.assertFalse(executor.preserve_output_dtype)
        layer.fused_moe = FusedMoe(Fp8Fp4Router.__new__(Fp8Fp4Router), executor, 1)
        layer._shared_executor.value = 2**-8
        x = torch.ones((2, 4), dtype=torch.bfloat16)
        with patch.object(
            layer_module, "combine_routed_and_shared", side_effect=self._combine(events)
        ):
            output = layer(x, None)
        expected = (torch.full_like(x, executor.value).float() + 2**-8).to(x.dtype)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)

    def test_shared_expert_sum_rounds_only_after_fp32_routed_output(self):
        events = []
        layer = self._layer(events, gate_pack_enabled=False)
        executor = _Fp32Executor.__new__(_Fp32Executor)
        torch.nn.Module.__init__(executor)
        executor.value = 1.003
        executor.preserve_output_dtype = True
        layer.fused_moe = FusedMoe(Fp8Fp4Router.__new__(Fp8Fp4Router), executor, 1)
        layer._shared_executor.value = 2**-8
        x = torch.ones((2, 4), dtype=torch.bfloat16)
        with patch.object(
            layer_module, "combine_routed_and_shared", side_effect=self._combine(events)
        ):
            output = layer(x, None)
        expected = (
            torch.full_like(x, executor.value, dtype=torch.float32) + 2**-8
        ).to(x.dtype)
        early_rounded = (torch.full_like(x, executor.value).float() + 2**-8).to(x.dtype)
        self.assertFalse(torch.equal(expected, early_rounded))
        torch.testing.assert_close(output, expected, rtol=0, atol=0)

    def test_single_round_dispatch_chunks_local_gate_and_shared_expert_after_routed(
        self,
    ):
        events = []
        layer = self._layer(events, gate_pack_enabled=False)
        layer.max_tokens_per_rank = 4
        gate_sizes = []
        shared_sizes = []

        class _RecordingGate(_FakeGate):
            def __call__(self, gx, gids):
                gate_sizes.append(gx.size(0))
                return super().__call__(gx, gids)

        layer.gate = _RecordingGate(events, gate_pack_enabled=False)

        orig_start = layer._shared_executor.start

        def record_start(shared_experts, sx):
            shared_sizes.append(sx.size(0))
            return orig_start(shared_experts, sx)

        layer._shared_executor.start = record_start
        layer._shared_executor.value = 0.25

        executor = _Fp32Executor.__new__(_Fp32Executor)
        torch.nn.Module.__init__(executor)
        executor.value = 1.5
        executor.preserve_output_dtype = True
        executor.single_round_dispatch = True
        routed_sizes = []
        orig_forward = executor.forward

        def record_routed(rx, rw, ri):
            events.append("routed.forward")
            routed_sizes.append(rx.size(0))
            return orig_forward(rx, rw, ri)

        executor.forward = record_routed
        layer.fused_moe = FusedMoe(Fp8Fp4Router.__new__(Fp8Fp4Router), executor, 1)

        x = torch.ones((11, 4), dtype=torch.bfloat16)
        with patch.object(
            layer_module, "combine_routed_and_shared", side_effect=self._combine(events)
        ):
            output = layer(x, None)

        self.assertEqual(gate_sizes, [4, 4, 3])
        self.assertEqual(routed_sizes, [11])
        self.assertEqual(shared_sizes, [4, 4, 3])
        self.assertLess(events.index("routed.forward"), events.index("shared.start"))
        expected = torch.full_like(x, 1.75)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)


class Fp8Fp4ExecutorOutputDtypeTest(unittest.TestCase):
    def setUp(self):
        self.executor = _Fp32Executor.__new__(_Fp32Executor)
        torch.nn.Module.__init__(self.executor)
        self.x = torch.ones((2, 4), dtype=torch.bfloat16)

    def _execute(self, payload):
        return self.executor.execute(
            payload,
            activation="silu",
            expert_map=None,
            a2_scale=None,
            apply_router_weight_on_input=False,
            extra_expert_args=None,
        ).fused_expert_output

    def test_ordinary_output_restores_router_input_dtype(self):
        output = self._execute(
            ExpertForwardPayload(
                expert_x=self.x,
                expert_x_origin_dtype=self.x.dtype,
                expert_topk_weights=torch.ones((2, 1), dtype=torch.float32),
                expert_topk_ids=torch.zeros((2, 1), dtype=torch.long),
            )
        )

        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertEqual((output + self.x).dtype, torch.bfloat16)

    def test_gate_pack_output_restores_router_input_dtype(self):
        output = self._execute(
            ExpertForwardPayload(
                expert_x=self.x,
                expert_x_origin_dtype=self.x.dtype,
                gate_payload=ExpertGatePayload(
                    scores=torch.ones((2, 1), dtype=torch.float32),
                    topk=1,
                    score_func="sigmoid",
                    route_scale=1.0,
                ),
            )
        )

        self.assertEqual(output.dtype, torch.bfloat16)

    def test_gate_pack_can_preserve_precision_for_layer_combine(self):
        self.executor.value = 1.003
        self.executor.preserve_output_dtype = True
        output = self._execute(
            ExpertForwardPayload(
                expert_x=self.x,
                expert_x_origin_dtype=self.x.dtype,
                gate_payload=ExpertGatePayload(
                    scores=torch.ones((2, 1), dtype=torch.float32),
                    topk=1,
                    score_func="sigmoid",
                    route_scale=1.0,
                ),
            )
        )
        self.assertEqual(output.dtype, torch.float32)
        torch.testing.assert_close(
            output, torch.full_like(output, 1.003), rtol=0, atol=0
        )


class W13SharedExpertDirectForwardTest(unittest.TestCase):
    def test_prefers_direct_shared_expert_bypasses_fast_path_under_strict_fused(self):
        x = torch.ones((2, 4), dtype=torch.bfloat16)

        class _FastPath:
            def __init__(self, runnable):
                self._runnable = runnable

            def can_run(self, _shared, _x):
                return self._runnable

            def _run_prepared(self, _shared, _x):
                raise AssertionError("fast_path must be bypassed")

        generic_expert = lambda t: t * 2.0
        with patch.dict(os.environ, {"MOE_STRICT_FUSED": "1"}):
            with self.assertRaisesRegex(RuntimeError, "MOE_STRICT_FUSED=1 forbids"):
                _run_shared_expert(generic_expert, x, _FastPath(runnable=False))


class MoeWorkspaceIdentityTest(unittest.TestCase):
    def test_shared_moe_workspace_resolves_to_single_sm120_instance(self):
        ws = MoeWorkspace()
        cfg1 = SimpleNamespace(moe_workspace=ws)
        cfg2 = SimpleNamespace(moe_workspace=ws)
        resolved1 = workspace_module.workspace_for(cfg1)
        resolved2 = workspace_module.workspace_for(cfg2)
        self.assertIsInstance(resolved1, workspace_module.Sm120MoeWorkspace)
        self.assertIs(resolved1, resolved2)

    def test_concurrent_bare_config_workspace_for_constructs_once(self):
        bare_cfg = SimpleNamespace()
        barrier = threading.Barrier(2)
        created = 0
        real_cls = workspace_module.Sm120MoeWorkspace

        class _CountingWorkspace(real_cls):
            def __init__(self):
                nonlocal created
                created += 1
                super().__init__()

        results = []
        with patch.object(
            workspace_module, "Sm120MoeWorkspace", new=_CountingWorkspace
        ):

            def worker():
                barrier.wait()
                results.append(workspace_module.workspace_for(bare_cfg))

            threads = [threading.Thread(target=worker) for _ in range(2)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        self.assertEqual(created, 1)
        self.assertEqual(len(results), 2)
        self.assertIs(results[0], results[1])

    def test_fp8_fp4_moe_layer_init_shares_moe_workspace_with_shared_executor(self):
        for provided_ws in (MoeWorkspace(), None):
            with self.subTest(explicit_workspace=provided_ws is not None):
                captured = {}

                def fake_create(runtime_config, _weights):
                    captured["runtime_config"] = runtime_config
                    return SimpleNamespace(
                        strategy_name="sm120_fused_moe",
                        includes_shared_expert=False,
                    )

                layer_weights = {
                    W.ffn_w13: torch.zeros((4, 4)),
                    W.ffn_s13: torch.ones((1, 1)),
                    W.ffn_w2: torch.zeros((4, 4)),
                    W.ffn_s2: torch.ones((1, 1)),
                }
                with (
                    patch.object(layer_module, "Gate", return_value=SimpleNamespace()),
                    patch.object(
                        layer_module, "_shared_fp8_block_size", return_value=128
                    ),
                    patch.object(layer_module, "FusedMoeFactory") as factory_cls,
                    patch.object(
                        layer_module, "W13SharedExpert", return_value=SimpleNamespace()
                    ),
                    patch.object(
                        layer_module, "get_shared_expert_executor"
                    ) as get_shared_exec,
                ):
                    factory_cls.return_value.create_fused_moe.side_effect = fake_create
                    layer = Fp8Fp4MoeLayer(
                        layer_id=0,
                        dim=4,
                        moe_inter_dim=4,
                        n_routed_experts=2,
                        n_activated_experts=1,
                        n_shared_experts=1,
                        score_func="softmax",
                        route_scale=1.0,
                        swiglu_limit=0.0,
                        n_hash_layers=0,
                        vocab_size=8,
                        layer_weights=layer_weights,
                        moe_workspace=provided_ws,
                    )
                self.assertIsInstance(layer.moe_workspace, MoeWorkspace)
                if provided_ws is not None:
                    self.assertIs(layer.moe_workspace, provided_ws)
                self.assertIs(
                    captured["runtime_config"].moe_workspace, layer.moe_workspace
                )
                self.assertIs(
                    get_shared_exec.call_args.kwargs["moe_workspace"],
                    layer.moe_workspace,
                )


class CreateFp8LinearUe8m0LayoutTest(unittest.TestCase):
    def test_ue8m0_scale_packs_for_deepgemm(self):
        from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.quantized_linear import (
            create_fp8_linear,
        )
        from rtp_llm.models_py.modules.factory.linear import LinearFactory

        weight = torch.zeros((130, 128), dtype=torch.float8_e4m3fn)
        e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
        scale = torch.zeros((2, 1), dtype=torch.float32)
        captured = {}
        sentinel = object()

        def fake_create(weights, weight_key, scale_key, quant_config=None):
            captured["scale"] = weights[scale_key]
            captured["quant_config"] = quant_config
            return sentinel

        with patch.object(
            LinearFactory,
            "create_linear_from_weights",
            side_effect=fake_create,
        ):
            out = create_fp8_linear(weight, scale)

        self.assertIs(out, sentinel)
        self.assertIs(captured["scale"], scale)
        self.assertEqual(captured["scale"].dtype, torch.float32)
        self.assertEqual(tuple(captured["scale"].shape), (2, 1))

        if e8m0_dtype is not None:
            e8m0_scale = torch.ones((2, 1), dtype=e8m0_dtype)
            fake_packed = torch.zeros((130, 1), dtype=torch.int32)
            pack = Mock(return_value=fake_packed)
            layout = SimpleNamespace(
                get_mn_major_tma_aligned_packed_ue8m0_tensor=pack
            )
            with (
                patch.dict(sys.modules, {"deep_gemm.utils.layout": layout}),
                patch.object(
                    LinearFactory,
                    "create_linear_from_weights",
                    side_effect=fake_create,
                ),
            ):
                out = create_fp8_linear(weight, e8m0_scale)
            self.assertIs(out, sentinel)
            self.assertIs(captured["scale"], fake_packed)
            self.assertEqual(pack.call_args.args[0].shape[0], weight.shape[0])

    def test_ue8m0_rejects_malformed_scale_or_k(self):
        from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.quantized_linear import (
            create_fp8_linear,
        )

        e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
        if e8m0_dtype is None:
            return

        weight = torch.zeros((128, 128), dtype=torch.float8_e4m3fn)
        bad_scale = torch.zeros((2, 1), dtype=e8m0_dtype)
        with self.assertRaises(ValueError):
            create_fp8_linear(weight, bad_scale)

        bad_k_weight = torch.zeros((128, 130), dtype=torch.float8_e4m3fn)
        bad_k_scale = torch.zeros((1, 1), dtype=e8m0_dtype)
        with self.assertRaises(ValueError):
            create_fp8_linear(bad_k_weight, bad_k_scale)


if __name__ == "__main__":
    unittest.main()
