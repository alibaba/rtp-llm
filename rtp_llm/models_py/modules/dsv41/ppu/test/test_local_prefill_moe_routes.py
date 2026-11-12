"""Local Prefill direct routes against the unchanged gate/pad/mask path.

CPU cases exercise admission and the torch fallback. Explicit PPU cases use
the actual fused gate and require byte-identical routes plus kernel evidence.
The MoE consumer records dispatch inputs; it does not emulate EP collectives.
"""

import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.modules.dsv41.ppu import layers, moe
from rtp_llm.models_py.modules.dsv41.ppu._quant import is_ppu


ENV = layers.LOCAL_PREFILL_DIRECT_MOE_ROUTES_ENV
CP_ENV = "RTP_V41_PPU_CP8_DIRECT_MOE_ROUTES"
EVIDENCE_ENV = "RTP_V41_SHARED_EVIDENCE_DIR"


def _state(real=None, *, context=False):
    model = NS(_is_speculative=False, _is_decode_role=False)
    state = NS(
        prefill=True, idle=False, is_graph=False, compacted=context,
        attn=NS(is_cuda_graph=False), cache=NS(sharded=False),
        context=NS(cp_size=1, kv_cache_sharded=False, local_is_real=real,
                   gather_restore_positions=None) if context else None,
        runtime=NS(model=model, policy=NS(mode="compact")), routed_rows={},
        use_compact_cp8_ops=lambda cp: False, prefill_mega_buffer=None,
    )
    if not context:
        state.active_token_mask = real
    state.runtime.current = state
    return state


def _metadata(rows=4096, *, width=5120, device="cuda:0", dtype=torch.bfloat16,
              contiguous=True):
    return NS(ndim=2, shape=(rows, width), dtype=dtype, device=torch.device(device),
              is_contiguous=lambda: contiguous)


def _mask_metadata(rows=4096, *, dtype=torch.bool, device="cuda:0", contiguous=True):
    value = Mock(spec=torch.Tensor)
    value.shape, value.dtype, value.device = (rows,), dtype, torch.device(device)
    value.is_contiguous.return_value = contiguous
    return value


class _Gate(torch.nn.Module):
    def __init__(self, width, experts, device):
        super().__init__()
        generator = torch.Generator().manual_seed(610)
        self.register_buffer("weight", torch.randn(experts, width, generator=generator).to(
            device=device, dtype=torch.bfloat16,
        ))
        self.register_buffer("bias", torch.randn(experts, generator=generator).to(device))
        self.graph_routes = None

    def forward(self, hidden, *, graph_routes=None, allow_top3_graph_routes=False):
        self.graph_routes = graph_routes
        return moe.sqrtsoftplus_gate(
            hidden, self.weight, self.bias, top_k=6, routed_scaling_factor=1.5,
            graph_routes=graph_routes, allow_top3_graph_routes=allow_top3_graph_routes,
        )


class _Consumer:
    def __call__(self, hidden, ids, weights, **kwargs):
        self.ids, self.weights, self.kwargs = ids, weights, kwargs
        return hidden


def _unit(state, device, *, width=32, experts=16):
    # Use the real forward without constructing checkpoint/EP resources.
    unit = layers.PPUSharedMoE.__new__(layers.PPUSharedMoE)
    torch.nn.Module.__init__(unit)
    unit.routed_enabled, unit.calls, unit.layer_id = True, 0, 0
    unit.shared_expert = None
    unit.gate = _Gate(width, experts, device)
    unit.ops = NS(moe=_Consumer())
    unit.runtime = state.runtime
    return unit


def _hidden(rows, width, device):
    return torch.randn(rows, width, generator=torch.Generator().manual_seed(611)).to(
        device=device, dtype=torch.bfloat16,
    )


def _assert_bytes(test, actual, expected):
    test.assertEqual(actual.shape, expected.shape)
    test.assertEqual(actual.dtype, expected.dtype)
    test.assertEqual(actual.device, expected.device)
    test.assertTrue(torch.equal(actual.contiguous().view(torch.uint8),
                                expected.contiguous().view(torch.uint8)))


class LocalPrefillMoEAdmission(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {ENV: "1", CP_ENV: "0"})
        environment.start()
        self.addCleanup(environment.stop)
        for target, value in (
            ("rtp_llm.models_py.modules.dsv41.ppu._quant.is_ppu", True),
            ("torch.cuda.get_device_capability", (8, 9)),
            ("torch.cuda.is_current_stream_capturing", False),
        ):
            mocked = patch(target, return_value=value)
            mocked.start()
            self.addCleanup(mocked.stop)

    @torch.inference_mode()
    def test_default_off_and_actual_full_compact_and_active_masks(self):
        real = _mask_metadata()
        state = _state(real, context=True)
        with patch.dict(os.environ):
            os.environ.pop(ENV, None)
            self.assertIsNone(layers._local_prefill_direct_moe_real(state, _metadata()))
        with patch.dict(os.environ, {ENV: "0"}):
            self.assertIsNone(layers._local_prefill_direct_moe_real(state, _metadata()))
        for rows in (0, 7, 140, 4096):
            real = _mask_metadata(rows)
            for context in (False, True):
                state = _state(real, context=context)
                self.assertIs(layers._local_prefill_direct_moe_real(state, _metadata(rows)), real)
        state = _state()
        with patch.object(layers.torch, "ones", return_value=real) as ones:
            self.assertIs(layers._local_prefill_direct_moe_real(state, _metadata()), real)
        ones.assert_called_once_with(4096, dtype=torch.bool, device=torch.device("cuda:0"))
        state = _state(_mask_metadata(), context=True)
        state.active_token_mask = _mask_metadata()
        self.assertIs(layers._local_prefill_direct_moe_real(state, _metadata()),
                      state.context.local_is_real, "context mask precedes active-token mask")
        replacement = _mask_metadata()
        state.context.local_is_real = replacement
        self.assertIs(layers._local_prefill_direct_moe_real(state, _metadata()), replacement)

    def test_grad_idle_decode_graph_sharding_and_other_devices_fall_back(self):
        state = _state(_mask_metadata(), context=True)
        self.assertIsNone(layers._local_prefill_direct_moe_real(state, _metadata()))
        with torch.inference_mode():
            self.assertIsNone(layers._local_prefill_direct_moe_real(None, _metadata()))
            for name in ("idle", "is_graph", "is_graph_commit", "verify", "proposal"):
                state = _state(_mask_metadata(), context=True)
                setattr(state, name, True)
                self.assertIsNone(layers._local_prefill_direct_moe_real(state, _metadata()))
            state = _state()
            state.prefill = False
            self.assertIsNone(layers._local_prefill_direct_moe_real(state, _metadata()))
            state = _state()
            state.attn.is_cuda_graph = True
            self.assertIsNone(layers._local_prefill_direct_moe_real(state, _metadata()))
            for name in ("_is_speculative", "_is_decode_role"):
                state = _state()
                setattr(state.runtime.model, name, True)
                self.assertIsNone(layers._local_prefill_direct_moe_real(state, _metadata()))
            for cache in (None, NS(sharded=True)):
                state = _state()
                state.cache = cache
                self.assertIsNone(layers._local_prefill_direct_moe_real(state, _metadata()))
            for cp in (NS(cp_size=8, kv_cache_sharded=True), NS(cp_size=1, kv_cache_sharded=True)):
                state = _state()
                state.context = cp
                self.assertIsNone(layers._local_prefill_direct_moe_real(state, _metadata()))
            for flat in (_metadata(device="cpu"), _metadata(dtype=torch.float32),
                         _metadata(width=4096), _metadata(contiguous=False)):
                self.assertIsNone(layers._local_prefill_direct_moe_real(_state(), flat))
            for real in (_mask_metadata(139), _mask_metadata(140, dtype=torch.int32),
                         _mask_metadata(140, device="cpu"), _mask_metadata(140, contiguous=False), None):
                self.assertIsNone(layers._local_prefill_direct_moe_real(
                    _state(real, context=True), _metadata(140),
                ))
            for target, value in (
                ("rtp_llm.models_py.modules.dsv41.ppu._quant.is_ppu", False),
                ("torch.cuda.get_device_capability", (8, 0)),
                ("torch.cuda.is_current_stream_capturing", True),
            ):
                with patch(target, return_value=value):
                    self.assertIsNone(layers._local_prefill_direct_moe_real(_state(), _metadata()))


class _RouteCases:
    def _compare(self, rows, *, context=False, all_invalid=False, special=None,
                 evidence=False, fused=True):
        real = torch.arange(rows, device=self.device) % 5 != 3
        if all_invalid:
            real.zero_()
        state = _state(real, context=context)
        unit = _unit(state, self.device, width=self.width, experts=self.experts)
        hidden = _hidden(rows, self.width, self.device)
        if special == "nonfinite" and rows:
            hidden[0] = float("nan")
            if rows > 1:
                hidden[1] = float("inf")
        elif special == "ties":
            unit.gate.weight.zero_()
            unit.gate.bias.zero_()
        elif special == "nonfinite_bias":
            unit.gate.bias.fill_(-float("inf"))
        evidence_value = "requested-test-evidence" if evidence else ""
        with patch.dict(os.environ, {CP_ENV: "0", EVIDENCE_ENV: evidence_value,
                                     "RTP_V41_PPU_GATE_TRITON": str(int(fused))}):
            with patch.dict(os.environ, {ENV: "0"}):
                expected_output = unit(hidden)
            expected_ids, expected_weights = unit.ops.moe.ids, unit.ops.moe.weights
            expected_count = state.routed_rows[0]
            state.routed_rows.clear()
            with patch.dict(os.environ, {ENV: "1"}):
                if self.device.type == "cpu":
                    # CPU exercises the same public gate fallback; hardware
                    # admission itself is tested separately above.
                    with patch.object(layers, "_local_prefill_direct_moe_real", return_value=real):
                        actual_output = unit(hidden)
                else:
                    actual_output = unit(hidden)
            routes = unit.gate.graph_routes
            self.assertIsNotNone(routes, "exercise the direct gate API")
            self.assertIs(routes[0], real)
            _assert_bytes(self, unit.ops.moe.ids, expected_ids)
            _assert_bytes(self, unit.ops.moe.weights, expected_weights)
            _assert_bytes(self, actual_output, expected_output)
            self.assertEqual(unit.ops.moe.ids.shape, (rows, 8))
            self.assertEqual(unit.ops.moe.ids.dtype, torch.int64)
            self.assertEqual(unit.ops.moe.weights.dtype, torch.float32)
            self.assertEqual(int(routes[1]), expected_count)
            self.assertEqual(expected_count, int(real.sum()))
            self.assertEqual(state.routed_rows, {0: expected_count} if evidence else {})
            self.assertNotIn("prepared_buffer", unit.ops.moe.kwargs)

    @torch.inference_mode()
    def test_full4096_compact140_ragged_padding_empty_and_all_invalid(self):
        for rows, context, all_invalid in ((4096, False, False), (140, True, False),
                                           (7, True, False), (0, False, False),
                                           (9, True, True)):
            with self.subTest(rows=rows, context=context, all_invalid=all_invalid):
                self._compare(rows, context=context, all_invalid=all_invalid)

    @torch.inference_mode()
    def test_ties_nonfinite_and_existing_torch_fallback_are_exact(self):
        for special in ("ties", "nonfinite", "nonfinite_bias"):
            for fused in (False, True):
                with self.subTest(special=special, fused=fused):
                    self._compare(7, context=True, special=special, fused=fused)

    @torch.inference_mode()
    def test_evidence_keeps_accurate_device_count_and_host_reporting(self):
        self._compare(140, context=True, evidence=True)
        self._compare(7, context=True, all_invalid=True, evidence=True, fused=False)


class LocalPrefillMoERoutesCPU(_RouteCases, unittest.TestCase):
    device, width, experts = torch.device("cpu"), 32, 16

    @torch.inference_mode()
    def test_cpu_idle_decode_and_cp8_keep_legacy_route_and_capacity(self):
        real = torch.tensor((True, False, True))
        for kind in ("cpu", "idle", "decode", "cp8"):
            state = _state(real, context=True)
            if kind == "idle":
                state.idle = True
            elif kind == "decode":
                state.prefill = False
            elif kind == "cp8":
                state.context.cp_size, state.context.kv_cache_sharded = 8, True
                state.prefill_mega_buffer = object()
            unit = _unit(state, self.device)
            hidden = _hidden(3, self.width, self.device)
            with patch.dict(os.environ, {ENV: "0", CP_ENV: "0", EVIDENCE_ENV: ""}):
                unit(hidden)
            expected_ids, expected_weights = unit.ops.moe.ids, unit.ops.moe.weights
            expected_kwargs = unit.ops.moe.kwargs
            with patch.dict(os.environ, {ENV: "1", CP_ENV: "0", EVIDENCE_ENV: ""}):
                unit(hidden)
            self.assertIsNone(unit.gate.graph_routes)
            _assert_bytes(self, unit.ops.moe.ids, expected_ids)
            _assert_bytes(self, unit.ops.moe.weights, expected_weights)
            self.assertEqual(unit.ops.moe.kwargs, expected_kwargs)
            if kind == "idle":
                self.assertEqual(state.routed_rows, {0: 0})
            if kind == "cp8":
                self.assertIs(unit.ops.moe.kwargs["prepared_buffer"], state.prefill_mega_buffer)

    @torch.inference_mode()
    def test_graph_exits_before_local_admission(self):
        real = torch.tensor((True, False, True))
        state = _state()
        state.is_graph, state.grouped_moe, state.bulk_counters = True, None, True
        state.metadata = NS(valid_mask=real)
        state.routed_rows_device = {0: torch.zeros((), dtype=torch.int64)}
        unit = _unit(state, self.device)
        consumer = _Consumer()
        unit.ops.moe_engine = NS(forward_shared_graph=consumer)
        with patch.dict(os.environ, {ENV: "1"}), patch.object(
            layers, "_local_prefill_direct_moe_real", side_effect=AssertionError("graph admission"),
        ):
            unit(_hidden(3, self.width, self.device))
        self.assertIs(unit.gate.graph_routes[0], real)
        self.assertEqual(int(state.routed_rows_device[0]), 2)
        self.assertTrue(consumer.kwargs["fused_expert_load"])


@unittest.skipUnless(os.environ.get("RTP_V41_TEST_LOCAL_WORKSPACE_DEVICE") == "1",
                     "explicit PPU device test")
class LocalPrefillMoERoutesDevice(_RouteCases, unittest.TestCase):
    device, width, experts = torch.device("cuda:0"), 5120, 384

    def setUp(self):
        self.assertTrue(is_ppu(), "direct route qualification requires a ZW PPU")
        self.assertEqual(torch.cuda.get_device_capability(), (8, 9))

    @torch.inference_mode()
    def test_one_module_observes_current_layout_mask_and_inplace_updates(self):
        state = _state()
        unit = _unit(state, self.device, width=self.width, experts=self.experts)
        real = torch.arange(7, device=self.device) % 3 != 1
        hidden = _hidden(7, self.width, self.device)
        with patch.dict(os.environ, {CP_ENV: "0", EVIDENCE_ENV: "",
                                     "RTP_V41_PPU_GATE_TRITON": "1"}):
            for transition in ("full", "compact", "inplace", "rebind"):
                if transition == "compact":
                    state.context = NS(cp_size=1, kv_cache_sharded=False, local_is_real=real,
                                       gather_restore_positions=None)
                    state.compacted = True
                elif transition == "inplace":
                    real.logical_not_()
                elif transition == "rebind":
                    state.context.local_is_real = torch.zeros(7, dtype=torch.bool, device=self.device)
                with patch.dict(os.environ, {ENV: "0"}):
                    unit(hidden)
                saved = unit.ops.moe.ids, unit.ops.moe.weights, state.routed_rows[0]
                state.routed_rows.clear()
                with patch.dict(os.environ, {ENV: "1"}):
                    unit(hidden)
                _assert_bytes(self, unit.ops.moe.ids, saved[0])
                _assert_bytes(self, unit.ops.moe.weights, saved[1])
                self.assertEqual(int(unit.gate.graph_routes[1]), saved[2])
                self.assertEqual(state.routed_rows, {})
                if state.context is not None:
                    self.assertIs(unit.gate.graph_routes[0], state.context.local_is_real)

    @torch.inference_mode()
    def test_idle_decode_and_existing_cp8_direct_routes_are_unchanged(self):
        real = torch.tensor((True, False, True), device=self.device)
        for kind in ("idle", "decode", "cp8"):
            state = _state(real, context=True)
            if kind == "idle":
                state.idle = True
            elif kind == "decode":
                state.prefill = False
            else:
                state.context.cp_size, state.context.kv_cache_sharded = 8, True
                state.prefill_mega_buffer = object()
            unit = _unit(state, self.device, width=self.width, experts=self.experts)
            hidden = _hidden(3, self.width, self.device)
            with patch.dict(os.environ, {ENV: "0", CP_ENV: "1", EVIDENCE_ENV: ""}):
                unit(hidden)
            expected_ids, expected_weights = unit.ops.moe.ids, unit.ops.moe.weights
            expected_kwargs = unit.ops.moe.kwargs
            with patch.dict(os.environ, {ENV: "1", CP_ENV: "1", EVIDENCE_ENV: ""}):
                unit(hidden)
            _assert_bytes(self, unit.ops.moe.ids, expected_ids)
            _assert_bytes(self, unit.ops.moe.weights, expected_weights)
            self.assertEqual(unit.ops.moe.kwargs, expected_kwargs)
            self.assertEqual(unit.gate.graph_routes is not None, kind == "cp8")
            if kind == "cp8":
                self.assertIs(unit.gate.graph_routes[0], real)
                self.assertIs(unit.ops.moe.kwargs["prepared_buffer"], state.prefill_mega_buffer)
            elif kind == "idle":
                self.assertEqual(state.routed_rows, {0: 0})

    @torch.inference_mode()
    def test_native_gate_profile_eliminates_external_pad_mask_and_host_count(self):
        state = _state()
        unit = _unit(state, self.device, width=self.width, experts=self.experts)
        hidden = _hidden(4096, self.width, self.device)
        summaries = {}
        for enabled in ("0", "1"):
            with patch.dict(os.environ, {ENV: enabled, CP_ENV: "0", EVIDENCE_ENV: "",
                                         "RTP_V41_PPU_GATE_TRITON": "1"}):
                unit(hidden)  # Warm each existing kernel variant before profiling.
                torch.cuda.synchronize()
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                         torch.profiler.ProfilerActivity.CUDA]) as profile:
                    unit(hidden)
                    torch.cuda.synchronize()
            events = profile.events()
            counts = {name: sum(event.name == name for event in events) for name in (
                "aten::cat", "aten::where", "aten::any", "aten::sum",
                "aten::item", "aten::_local_scalar_dense",
            )}
            kernels = sorted({event.name for event in events
                              if event.device_type == torch.autograd.DeviceType.CUDA})
            self.assertTrue(any("_router_gate_kernel" in name for name in kernels), kernels)
            summaries[enabled] = dict(counts=counts, kernels=kernels)
            if enabled == "0":
                saved = unit.ops.moe.ids, unit.ops.moe.weights
                self.assertGreaterEqual(counts["aten::cat"], 2)
                self.assertGreaterEqual(counts["aten::where"], 2)
                self.assertGreaterEqual(counts["aten::item"], 1)
            else:
                self.assertEqual(sum(counts.values()), 0, counts)
                self.assertIsNotNone(unit.gate.graph_routes)
                _assert_bytes(self, unit.ops.moe.ids, saved[0])
                _assert_bytes(self, unit.ops.moe.weights, saved[1])
                self.assertEqual(int(unit.gate.graph_routes[1]), 4096)
                self.assertNotIn("prepared_buffer", unit.ops.moe.kwargs)
            directory = os.environ.get("RTP_V41_TEST_LOCAL_MOE_ROUTES_TRACE_DIR") or os.environ.get(
                "TEST_UNDECLARED_OUTPUTS_DIR"
            )
            if directory:
                Path(directory).mkdir(parents=True, exist_ok=True)
                profile.export_chrome_trace(str(Path(directory) / f"local-moe-routes-{enabled}.json"))
        print(json.dumps({"local_prefill_direct_moe_routes": summaries}, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
