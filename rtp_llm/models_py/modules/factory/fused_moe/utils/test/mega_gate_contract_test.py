"""CPU contract tests for ``deep_gemm.bf16_mega_gate`` integration."""

from __future__ import annotations

import ast
import os
import sys
import types
import unittest
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    ExpertGatePayload,
    FusedMoe,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe import (
    _MEGA_MOE_JIT_WARMED_KEYS,
    MegaMoeExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_se import (
    MegaMoeSEExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.fp8_fp4_router import (
    MegaMoeRouter,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4 import (
    gate as gate_module,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4 import (
    layer as layer_module,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4 import (
    mega_gate as mega_gate_module,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.gate import Gate
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.layer import (
    Fp8Fp4MoeLayer,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.mega_gate import (
    MegaGateSpec,
    can_use_bf16_mega_gate,
    deduplicate_mega_gate_token_counts,
    prepare_capture_stream_barrier,
    run_bf16_mega_gate,
    try_bf16_mega_gate,
    warmup_bf16_mega_gate_specs,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.input_packer import (
    FusedMegaMoEInputPacker,
)
from rtp_llm.models_py.triton_kernels.moe import (
    mega_moe_input_pack as mega_pack_module,
)
from rtp_llm.utils.model_weight import W


@contextmanager
def _env_vars(**pairs: str | None):
    saved = {k: os.environ.get(k) for k in pairs}
    try:
        for k, v in pairs.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _make_gate(
    *,
    hidden: int = 512,
    experts: int = 16,
    topk: int = 4,
    score_func: str = "sqrtsoftplus",
    is_hash: bool = False,
    with_bias_vl: bool = False,
    has_visual_tokens: bool = False,
    vocab_size: int = 100,
    route_scale: float = 2.5,
) -> Gate:
    layer_weights = {
        W.moe_gate: torch.randn((experts, hidden), dtype=torch.bfloat16),
        W.moe_gate_bias: torch.randn((experts,), dtype=torch.float32) * 0.1,
        W.moe_gate_tid2eid: torch.stack(
            [torch.randperm(experts)[:topk] for _ in range(vocab_size)]
        ),
    }
    if with_bias_vl:
        layer_weights[W.moe_gate_bias_vl] = torch.randn((experts,), dtype=torch.float32)
    gate = Gate(
        layer_id=0,
        dim=hidden,
        n_routed_experts=experts,
        n_activated_experts=topk,
        score_func=score_func,
        route_scale=route_scale,
        n_hash_layers=1 if is_hash else 0,
        vocab_size=vocab_size,
        layer_weights=layer_weights,
    )
    gate._has_visual_tokens = has_visual_tokens
    return gate


class _DummySymBuf:
    def __init__(self, max_tokens: int, hidden: int, topk: int, *, with_se: bool = False):
        self.num_max_tokens_per_rank = max_tokens
        self.x = torch.empty((max_tokens, hidden), dtype=torch.float8_e4m3fn)
        self.x_sf = torch.empty((max_tokens, hidden // 128), dtype=torch.int32)
        self.topk_idx = torch.full((max_tokens, topk), -1, dtype=torch.int64)
        self.topk_weights = torch.full((max_tokens, topk), -1.0, dtype=torch.float32)
        self.shared_l1_acts_sf = (
            torch.empty((max_tokens * 2, hidden // 128), dtype=torch.int32)
            if with_se
            else None
        )


class MegaGateContractTest(unittest.TestCase):
    def setUp(self):
        mega_gate_module._WARMED_STREAMS.clear()
        mega_gate_module._LAST_SPEC_BY_DEVICE.clear()
        mega_gate_module._WARMED_JIT_KEYS.clear()
        mega_gate_module._ALL_TRUE_MASK_CACHE.clear()
        self.addCleanup(mega_gate_module._WARMED_STREAMS.clear)
        self.addCleanup(mega_gate_module._LAST_SPEC_BY_DEVICE.clear)
        self.addCleanup(mega_gate_module._WARMED_JIT_KEYS.clear)
        self.addCleanup(mega_gate_module._ALL_TRUE_MASK_CACHE.clear)

    @contextmanager
    def _mock_sm100_cpu_env(self, *, bf16_fn=None, config_fn=None, capturing: bool = False, stream_id: int = 101):
        calls = []

        def _default_bf16_mega_gate(
            x,
            weight,
            num_topk,
            use_shared_as_routed,
            num_shared_experts,
            routed_scaling_factor,
            ep_rank,
            scoring_func="identity",
            mask=None,
            bias=None,
            image_bias=None,
            image_token_mask=None,
            fix_routing_mask=None,
            to_physical_map=None,
            logical_count=None,
            unmapped_topk_idx=None,
            force_random=None,
            out=None,
        ):
            calls.append(
                {
                    "tokens": int(x.shape[0]),
                    "hidden": int(x.shape[1]),
                    "num_experts": int(weight.shape[0]),
                    "num_topk": int(num_topk),
                    "use_shared_as_routed": use_shared_as_routed,
                    "num_shared_experts": num_shared_experts,
                    "routed_scaling_factor": routed_scaling_factor,
                    "ep_rank": ep_rank,
                    "scoring_func": scoring_func,
                    "bias": bias,
                    "image_bias": image_bias,
                    "image_token_mask": image_token_mask,
                    "fix_routing_mask": fix_routing_mask,
                    "to_physical_map": to_physical_map,
                    "logical_count": logical_count,
                    "unmapped_topk_idx": unmapped_topk_idx,
                    "out": out,
                }
            )
            T = int(x.shape[0])
            K = int(num_topk)
            idx = (
                unmapped_topk_idx.clone()
                if unmapped_topk_idx is not None
                else torch.arange(K, dtype=torch.int64).unsqueeze(0).expand(T, K).clone()
            )
            weights = torch.full((T, K), float(routed_scaling_factor) / K, dtype=torch.float32)
            if out is not None:
                out[0].copy_(idx)
                out[1].copy_(weights)
                return out
            return idx, weights

        impl = bf16_fn or _default_bf16_mega_gate
        fake_dg = types.SimpleNamespace(bf16_mega_gate=impl)
        if config_fn is not None:
            fake_dg.get_bf16_mega_gate_config = config_fn

        with (
            mock.patch.dict(sys.modules, {"deep_gemm": fake_dg}),
            mock.patch.object(mega_gate_module, "_is_on_cuda", return_value=True),
            mock.patch.object(mega_gate_module, "_is_sm100_device", return_value=True),
            mock.patch.object(
                mega_gate_module, "_stream_key", side_effect=lambda dev: (0, stream_id)
            ),
            mock.patch("torch.cuda.is_current_stream_capturing", return_value=capturing),
            mock.patch(
                "torch.cuda.current_stream",
                return_value=SimpleNamespace(cuda_stream=stream_id, synchronize=lambda: None),
            ),
            mock.patch("torch.cuda.device", side_effect=lambda *a: nullcontext()),
        ):
            yield calls

    def test_can_use_eligibility_and_bypasses(self):
        gate = _make_gate(hidden=512, experts=16, topk=4, score_func="sqrtsoftplus")
        x = torch.randn((8, 512), dtype=torch.bfloat16)

        with self._mock_sm100_cpu_env():
            self.assertTrue(can_use_bf16_mega_gate(gate, x))

            with _env_vars(MOE_GATE_FP32="1"):
                self.assertFalse(can_use_bf16_mega_gate(gate, x))
            with _env_vars(MOE_GATE_FUSED="0"):
                self.assertFalse(can_use_bf16_mega_gate(gate, x))

            self.assertFalse(can_use_bf16_mega_gate(gate, x[:0]))
            self.assertFalse(can_use_bf16_mega_gate(gate, x.float()))
            self.assertFalse(can_use_bf16_mega_gate(gate, x.t()))

            gate_softmax = _make_gate(score_func="softmax")
            self.assertFalse(can_use_bf16_mega_gate(gate_softmax, x))

            gate_bad_h = _make_gate(hidden=384)
            self.assertFalse(can_use_bf16_mega_gate(gate_bad_h, torch.randn((4, 384), dtype=torch.bfloat16)))

            gate_bad_e = _make_gate(experts=14)
            self.assertFalse(can_use_bf16_mega_gate(gate_bad_e, x))

            gate_big_k = _make_gate(experts=64, topk=33)
            self.assertFalse(can_use_bf16_mega_gate(gate_big_k, x))

            gate._ced_row_projection = lambda a, b: a
            self.assertFalse(can_use_bf16_mega_gate(gate, x))
            del gate._ced_row_projection

        # Non-SM100 device capability
        with (
            mock.patch.dict(sys.modules, {"deep_gemm": types.SimpleNamespace(bf16_mega_gate=lambda *a, **k: None)}),
            mock.patch.object(mega_gate_module, "_is_on_cuda", return_value=True),
            mock.patch.object(mega_gate_module, "_device_index", return_value=0),
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.get_device_capability", return_value=(9, 0)),
        ):
            self.assertFalse(can_use_bf16_mega_gate(gate, x))

        # Unwarmed vs warmed capture stream
        with self._mock_sm100_cpu_env(capturing=True, stream_id=777):
            self.assertFalse(can_use_bf16_mega_gate(gate, x))
            mega_gate_module._WARMED_STREAMS.add((0, 777))
            self.assertTrue(can_use_bf16_mega_gate(gate, x))

    def test_run_and_try_bf16_mega_gate_return_order_hash_vl_and_chunking(self):
        # Non-hash with out= and chunk_rows
        gate = _make_gate(hidden=512, experts=16, topk=4, score_func="sigmoid", route_scale=3.0)
        x = torch.randn((5, 512), dtype=torch.bfloat16)
        with self._mock_sm100_cpu_env() as calls:
            out_idx = torch.empty((5, 4), dtype=torch.int64)
            out_w = torch.empty((5, 4), dtype=torch.float32)
            idx, weights = run_bf16_mega_gate(gate, x, out=(out_idx, out_w), chunk_rows=2)
            self.assertIs(idx, out_idx)
            self.assertIs(weights, out_w)
            self.assertEqual([c["tokens"] for c in calls], [2, 2, 1])
            self.assertFalse(calls[0]["use_shared_as_routed"])
            self.assertEqual(calls[0]["num_shared_experts"], 0)
            self.assertEqual(calls[0]["scoring_func"], "sigmoid")

            # Gate.forward() swaps DeepGEMM's (idx, weights) -> (weights, idx)
            fw, fi = gate(x)
            self.assertEqual(fw.dtype, torch.float32)
            self.assertEqual(fi.dtype, torch.int64)
            torch.testing.assert_close(fw, torch.full((5, 4), 3.0 / 4))

        # Hash + visual routing contracts
        hgate = _make_gate(
            hidden=512,
            experts=16,
            topk=4,
            is_hash=True,
            with_bias_vl=True,
            has_visual_tokens=True,
            vocab_size=50,
            route_scale=2.0,
        )
        orig_tid2eid = hgate.tid2eid.clone()
        input_ids = torch.tensor([1, 55, 2, 60], dtype=torch.long)
        with self._mock_sm100_cpu_env() as calls:
            w, i = try_bf16_mega_gate(hgate, x[:4], input_ids)
            self.assertIsNotNone(w)
            self.assertEqual(len(calls), 1)
            c = calls[0]
            self.assertIsNone(c["bias"])
            self.assertIsNotNone(c["image_bias"])
            self.assertEqual(c["image_token_mask"].tolist(), [False, True, False, True])
            self.assertEqual(c["fix_routing_mask"].tolist(), [True, False, True, False])
            self.assertTrue(torch.equal(hgate.tid2eid, orig_tid2eid))

    def test_gate_ast_extraction_compatibility(self):
        gate_path = (
            Path(gate_module.__file__).resolve()
        )
        tree = ast.parse(gate_path.read_text())
        nodes = [
            node
            for node in tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef))
            and node.name in ("Gate", "_select_routes_with_nonfinite_fallback")
        ]
        ns = dict(
            torch=torch,
            nn=torch.nn,
            F=F,
            os=os,
            _use_fused_gate=lambda *a: False,
            ExpertGatePayload=lambda **kw: kw,
        )
        prefix = ast.parse("from __future__ import annotations").body
        exec(compile(ast.Module(body=prefix + nodes, type_ignores=[]), str(gate_path), "exec"), ns)
        gate = ns["Gate"].__new__(ns["Gate"])
        torch.nn.Module.__init__(gate)
        gate.weight = torch.randn((8, 16), dtype=torch.bfloat16)
        gate.bias = torch.zeros(8, dtype=torch.float32)
        gate.bias_vl = None
        gate.hash = False
        gate.topk = 2
        gate.score_func = "sqrtsoftplus"
        gate.route_scale = 1.0
        w, i = gate(torch.randn((3, 16), dtype=torch.bfloat16))
        self.assertEqual(w.shape, (3, 2))
        self.assertEqual(i.shape, (3, 2))

    def test_mega_moe_executor_and_se_forward_mega_gate_and_layer_scheduling(self):
        for exec_cls, with_se in ((MegaMoeExecutor, False), (MegaMoeSEExecutor, True)):
            with self.subTest(executor=exec_cls.__name__):
                ex = exec_cls.__new__(exec_cls)
                torch.nn.Module.__init__(ex)
                ex.cfg = SimpleNamespace(
                    layer_id=0,
                    n_activated_experts=4,
                    dim=512,
                )
                ex._input_packer = FusedMegaMoEInputPacker()
                ex._mega_buf = _DummySymBuf(16, 512, 4, with_se=with_se)
                ex._mega_y = torch.empty((16, 512), dtype=torch.bfloat16)
                if with_se:
                    ex._shared_l1_w = ex._shared_l1_sf = torch.empty(1)
                    ex._shared_l2_w = ex._shared_l2_sf = torch.empty(1)
                    ex._shared_mid_fp8 = torch.empty((16, 128), dtype=torch.float8_e4m3fn)
                    ex._shared_mid_sf = torch.empty((16, 1), dtype=torch.int32)
                    ex._block_m = lambda tokens: 128

                gate = _make_gate(hidden=512, experts=16, topk=4, route_scale=2.5)
                x = torch.randn((3, 512), dtype=torch.bfloat16)
                with (
                    self._mock_sm100_cpu_env(),
                    mock.patch.object(
                        mega_gate_module, "is_bf16_mega_gate_available", return_value=True
                    ),
                    mock.patch.object(
                        mega_pack_module, "fused_pack_mega_moe_x_inputs"
                    ) as x_pack_mock,
                    mock.patch.object(ex, "_launch") as launch_mock,
                ):
                    self.assertTrue(ex.supports_mega_gate)
                    out_buf = torch.empty((3, 512), dtype=torch.bfloat16)
                    y, w, idx = ex.forward_mega_gate(x, gate, out=out_buf)
                    self.assertIs(y, out_buf)
                    self.assertEqual(w.shape, (3, 4))
                    self.assertEqual(idx.shape, (3, 4))
                    self.assertTrue(torch.allclose(ex._mega_buf.topk_weights[:3], w))
                    self.assertTrue(torch.equal(ex._mega_buf.topk_idx[:3], idx))
                    x_pack_mock.assert_called_once()
                    if with_se:
                        self.assertIs(
                            x_pack_mock.call_args.kwargs["out_shared_l1_sf"],
                            ex._mega_buf.shared_l1_acts_sf,
                        )
                        self.assertEqual(
                            x_pack_mock.call_args.kwargs["shared_l1_block_m"], 128
                        )
                    launch_mock.assert_called_once_with(out_buf, 3, x.device)

                    # Zero-token EP rank still enters _launch without running bf16_mega_gate
                    x_pack_mock.reset_mock()
                    launch_mock.reset_mock()
                    y0, w0, idx0 = ex.forward_mega_gate(x[:0], gate)
                    self.assertEqual(y0.shape, (0, 512))
                    self.assertEqual(w0.shape, (0, 4))
                    self.assertEqual(idx0.shape, (0, 4))
                    launch_mock.assert_called_once()

        # FusedMoe.supports_mega_gate respects supports_gate_pack overrides
        class _NoGatePackFusedMoe(FusedMoe):
            @property
            def supports_gate_pack(self) -> bool:
                return False

        router = MegaMoeRouter.__new__(MegaMoeRouter)
        dummy_exec = SimpleNamespace(
            supports_gate_pack=True,
            supports_mega_gate=True,
            includes_shared_expert=False,
            topk_ids_dtype=torch.int64,
        )
        with mock.patch.object(mega_gate_module, "is_bf16_mega_gate_available", return_value=True):
            self.assertTrue(router.supports_mega_gate)
            fm = FusedMoe(router, dummy_exec, 16, strategy_name="mega_moe")
            self.assertTrue(fm.supports_mega_gate)
            fm_disabled = _NoGatePackFusedMoe(
                router, dummy_exec, 16, strategy_name="mega_moe"
            )
            self.assertFalse(fm_disabled.supports_mega_gate)

        # Layer scheduling: mega_gate skips Gate forward & starts shared executor before forward_mega_gate
        events = []
        layer = Fp8Fp4MoeLayer.__new__(Fp8Fp4MoeLayer)
        torch.nn.Module.__init__(layer)
        layer.dim = 512
        layer.gate = _make_gate(hidden=512, experts=16, topk=4)
        layer.shared_experts = object()
        layer._shared_executor = SimpleNamespace(
            start=lambda se, inp: events.append("shared.start"),
            finish=lambda: events.append("shared.finish") or torch.zeros_like(x),
        )
        layer.fused_moe = SimpleNamespace(
            supports_mega_gate=True,
            supports_gate_pack=True,
            forward_mega_gate=lambda **kw: events.append("fused_moe.forward_mega_gate")
            or torch.zeros_like(kw["hidden_states"]),
        )
        with (
            self._mock_sm100_cpu_env(),
            mock.patch.object(
                layer_module,
                "combine_routed_and_shared",
                side_effect=lambda routed, shared, dtype, out=None: (routed + shared).to(dtype),
            ),
        ):
            out = layer(x, None)
        self.assertEqual(out.shape, x.shape)
        self.assertEqual(
            events, ["shared.start", "fused_moe.forward_mega_gate", "shared.finish"]
        )

    def test_warmup_dedup_and_capture_stream_barrier_init(self):
        def fake_config(num_tokens, hidden, num_experts, topk):
            return {"block_tokens": 32 if num_tokens <= 4 else 64, "num_split_k": 1}

        reps = deduplicate_mega_gate_token_counts(
            hidden=512,
            num_experts=16,
            topk=4,
            max_tokens=16,
            config_fn=fake_config,
        )
        self.assertEqual(reps, [1, 5])

        # Identical specs across 10 layers compile only 2 token buckets total
        model = torch.nn.ModuleList(
            [_make_gate(hidden=512, experts=16, topk=4) for _ in range(10)]
        )
        with self._mock_sm100_cpu_env(config_fn=fake_config, stream_id=501) as calls:
            # Non-DSV4 (no prior spec) capture barrier is a no-op
            self.assertFalse(prepare_capture_stream_barrier(torch.device("cpu")))

            compiled = warmup_bf16_mega_gate_specs(
                [g for g in model],
                max_tokens=16,
                device=torch.device("cpu"),
            )
            self.assertEqual(compiled, 2)
            self.assertEqual(len(calls), 2)
            self.assertIn((0, 501), mega_gate_module._WARMED_STREAMS)

        # Now capture stream 999 has not run bf16_mega_gate -> barrier init warms it once
        with self._mock_sm100_cpu_env(config_fn=fake_config, stream_id=999) as calls:
            self.assertTrue(prepare_capture_stream_barrier(torch.device("cpu")))
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["tokens"], 1)
            self.assertIn((0, 999), mega_gate_module._WARMED_STREAMS)
            # Idempotent on subsequent captures on the same capture stream
            self.assertFalse(prepare_capture_stream_barrier(torch.device("cpu")))
            self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
