"""Exact local shared SwiGLU activation reuse; no new math or tolerance.

Registration requires this module, test_local_prefill_operator_flags.py and
its existing dependencies, and v41_ppu_shared_sources (including the existing
moe/_silu_mul_bf16_triton.py provider). CPU metadata doubles prove admission
and caller wiring only. Device tests require an explicit real PPU89 and
compare current ATen bytes, including NaN payloads; they do not use the
different silu_mul_split_bf16 (non-exact activation) oracle.
"""

from contextlib import contextmanager, ExitStack
from importlib.machinery import ModuleSpec
import json
import os
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn
from torch.nn import functional as F

from rtp_llm.models_py.modules.dsv4 import _record_tensor, _forward_tensor_debug
from rtp_llm.models_py.modules.dsv41.moe import _silu_mul_bf16_triton as provider
from rtp_llm.models_py.modules.dsv41.ppu import _quant, attention_layer, layers, moe
from rtp_llm.models_py.modules.dsv41.ppu.linear import PPUPTPCLinear, PPUResidentBF16Linear
from rtp_llm.models_py.modules.dsv41.ppu.native_cache_adapter import SharedNativeCache
from rtp_llm.models_py.modules.dsv41.ppu.ptpc_quant import quantize_ptpc_activation
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_operator_flags import _trace_events


ENV = "RTP_V41_PPU_LOCAL_PREFILL_EXACT_SHARED_SWIGLU"
WIDTH, N = 4608, 2304


def raw_bytes(value):
    # A size-one stride-two tensor can report contiguous yet cannot be viewed
    # as bytes. Explicit contiguous-format clone also handles that observation.
    return (value.resolve_conj().resolve_neg().reshape(-1)
            .clone(memory_format=torch.contiguous_format).view(torch.uint8))


def view_metadata(value):
    return (value._cdata, value.untyped_storage().data_ptr(), value.storage_offset(),
            value.shape, value.stride(), value.dtype, value.is_neg(), value.is_conj())


def assert_bits(test, actual, expected, label=""):
    test.assertEqual(actual.shape, expected.shape, label)
    test.assertEqual(actual.dtype, expected.dtype, label)
    test.assertEqual(actual.device, expected.device, label)
    left, right = raw_bytes(actual), raw_bytes(expected)
    if not torch.equal(left, right):
        indices = torch.nonzero(left != right).flatten()[:12]
        detail = {"label": label, "shape": list(actual.shape),
                  "byte_indices": indices.cpu().tolist(),
                  "actual": left[indices].cpu().tolist(),
                  "expected": right[indices].cpu().tolist()}
        test.fail("exact ATen byte mismatch " + json.dumps(detail, sort_keys=True))


def old_activation(gate_up, limit=10.0):
    # The current source authority: clamp BF16 first, then widen, ATen SiLU,
    # FP32 multiply and exactly one BF16 landing. No replacement sigmoid.
    gate, up = gate_up.chunk(2, dim=-1)
    if limit and limit > 0:
        up = up.clamp(min=-float(limit), max=float(limit))
        gate = gate.clamp(max=float(limit))
    return (F.silu(gate.float()) * up.float()).to(torch.bfloat16)


def call(gate_up, *, enabled=False, limit=10.0, **kwargs):
    return moe.shared_expert_forward(
        gate_up, lambda value: value, lambda value: value, limit,
        local_prefill_exact=enabled, **kwargs,
    )


@contextmanager
def native_metadata_double(**changes):
    """CPU admission double only; it never executes a native device kernel."""
    real_find_spec = moe.importlib.util.find_spec
    triton_spec = real_find_spec("triton") or ModuleSpec("triton", loader=None)
    configuration = dict(ppu=True, capability=(8, 9), capture=False, triton=triton_spec)
    configuration.update(changes)

    def find_spec(name, *args, **kwargs):
        if name == "triton":
            return configuration["triton"]
        return real_find_spec(name, *args, **kwargs)

    with ExitStack() as stack:
        stack.enter_context(patch.object(torch.Tensor, "is_cuda", property(lambda self: True)))
        stack.enter_context(patch.object(_quant, "is_ppu", return_value=configuration["ppu"]))
        stack.enter_context(patch.object(torch.cuda, "get_device_capability", return_value=configuration["capability"]))
        stack.enter_context(patch.object(torch.cuda, "is_current_stream_capturing", return_value=configuration["capture"]))
        stack.enter_context(patch.object(moe.importlib.util, "find_spec", side_effect=find_spec))
        yield


def all_encodings_cases():
    # Each half independently visits all 65536 BF16 encodings. These are four
    # partner families, not an enumeration of 65536**2 Cartesian pairs.
    rows = (65536 + N - 1) // N
    ids = torch.arange(rows * N, dtype=torch.int32).reshape(rows, N) & 0xffff
    one = torch.full_like(ids, 0x3f80)
    cases = (("all-gate-unit-up", ids, one),
             ("unit-gate-all-up", one, ids),
             ("all-gate-permuted-up", ids, (ids * 257 + 0x1234) & 0xffff),
             ("permuted-gate-all-up", (ids * 17 + 0x4321) & 0xffff, ids))
    for name, gate, up in cases:
        yield name, torch.cat((gate, up), -1).to(torch.int16).view(torch.bfloat16)


def edge_input(rows, device="cpu"):
    generator = torch.Generator().manual_seed(811 + rows)
    value = (torch.randn(rows, WIDTH, generator=generator) * 4).bfloat16()
    edge = torch.tensor([
        0x0000, 0x8000, 0x0001, 0x8001, 0x007f, 0x807f, 0x0080, 0x8080,
        0x411f, 0x4120, 0x4121, 0xc11f, 0xc120, 0xc121, 0x7f7f, 0xff7f,
        0x7f80, 0xff80, 0x7f81, 0xff81, 0x7fc0, 0xffc0, 0x7fff, 0xffff,
    ], dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    if rows:
        value[0, :edge.numel()] = edge
        value[0, N:N + edge.numel()] = edge.flip(0)
    return value.to(device)


def local_state():
    cache = SharedNativeCache.__new__(SharedNativeCache)
    cache._is_prefill, cache.sharded = True, False
    model = NS(_is_decode_role=False, _is_speculative=False, _prefill_cp_size=1,
               parallelism_config=NS(tp_size=1))
    state = NS(prefill=True, idle=False, lengths=[4096], context=None,
               cache=cache, attn=NS(is_cuda_graph=False), routed_rows={},
               runtime=NS(model=model), compacted=False)
    state.runtime.current = state
    return state


@contextmanager
def local_phase_double():
    with ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, {ENV: "1"}))
        stack.enter_context(patch.object(layers, "is_ppu", return_value=True))
        stack.enter_context(patch.object(torch.cuda, "is_current_stream_capturing", return_value=False))
        stack.enter_context(patch.object(_record_tensor, "ENABLED", False))
        stack.enter_context(patch.object(_forward_tensor_debug, "enabled", return_value=False))
        stack.enter_context(patch.object(attention_layer, "optimized_diagnostics_enabled", return_value=False))
        yield


class ProbeLinear:
    """Real PTPC quantization/GEMM with observable unchanged boundaries."""

    def __init__(self, linear):
        self.linear, self.records = linear, []

    def __call__(self, value):
        payload, scale = self.linear.quantize_input(value)
        output = self.linear.forward_quantized(payload, scale)
        self.records.append((value, payload, scale, output))
        return output


class LocalPrefillSharedSwiGLUCPU(unittest.TestCase):
    def test_default_cpu_unsupported_and_original_errors_keep_old_bytes(self):
        encodings = list(all_encodings_cases())
        for half in (0, 1):
            bits = torch.cat([value[:, half * N:(half + 1) * N]
                              .contiguous().view(torch.int16).flatten()
                              for _, value in encodings]).to(torch.int32) & 0xffff
            self.assertTrue(torch.equal(bits.unique(sorted=True), torch.arange(65536)))
        for value, limit in ((edge_input(3), 10.0), (edge_input(0), 10.0),
                             (edge_input(3).float(), 10.0),
                             (edge_input(3)[:, ::2], 10.0),
                             (edge_input(3), 0.0), (edge_input(3), 3.0)):
            with self.subTest(shape=tuple(value.shape), dtype=value.dtype, limit=limit):
                before = raw_bytes(value)
                with patch.object(provider, "silu_mul_split_bf16_torch_exact", side_effect=AssertionError("CPU native")):
                    assert_bits(self, call(value, enabled=True, limit=limit), old_activation(value, limit))
                    assert_bits(self, call(value), old_activation(value))
                self.assertTrue(torch.equal(raw_bytes(value), before))
        value = edge_input(2).to_sparse()
        for enabled in (False, True):
            with self.subTest(sparse=True, enabled=enabled):
                with self.assertRaises((RuntimeError, NotImplementedError)):
                    call(value, enabled=enabled)

    @torch.no_grad()
    def test_core_admission_reads_actual_w13_output_and_strict_control(self):
        value = edge_input(3)
        expected = old_activation(value)
        with native_metadata_double(), patch.object(
            provider, "silu_mul_split_bf16_torch_exact",
            side_effect=lambda value, *, clamp_limit: old_activation(value, clamp_limit),
        ) as exact:
            assert_bits(self, call(value, enabled=True), expected)
            exact.assert_called_once_with(value, clamp_limit=10.0)
        # This is an actual W13 result check, not admission from x's geometry.
        hidden = torch.ones(3, 64, dtype=torch.bfloat16)
        with native_metadata_double(), patch.object(
            provider, "silu_mul_split_bf16_torch_exact",
            side_effect=lambda value, *, clamp_limit: old_activation(value, clamp_limit),
        ) as exact:
            actual = moe.shared_expert_forward(hidden, lambda value: edge_input(value.shape[0]),
                                              lambda value: value, local_prefill_exact=True)
            self.assertEqual(exact.call_args.args[0].shape, (3, WIDTH))
            assert_bits(self, actual, expected)
        # Metadata-only boundary coverage avoids allocating 151 MB merely to
        # verify the M ceiling. Neither case executes a CPU/device math oracle.
        for rows in (16384, 16385):
            value = torch.empty(rows, WIDTH, device="meta", dtype=torch.bfloat16)
            with self.subTest(rows=rows), native_metadata_double(), patch.object(
                provider, "silu_mul_split_bf16_torch_exact",
                return_value=torch.empty(rows, N, device="meta", dtype=torch.bfloat16),
            ) as exact:
                self.assertEqual(call(value, enabled=True).shape, (rows, N))
                self.assertEqual(exact.call_count, int(rows == 16384))

    @torch.no_grad()
    def test_core_rollback_metadata_no_launch_mutation_and_original_flags(self):
        base = edge_input(3)
        # No native admission is inferred from hidden: W13's actual result is
        # checked independently, including a transformed or wrong-width output.
        cases = (("false", base, False, 10.0), ("integer-control", base, 1, 10.0),
                 ("empty", base[:0], True, 10.0),
                 ("wrong-width", base[:, :128].contiguous(), True, 10.0),
                 ("strided", base[:, ::2], True, 10.0),
                 ("negative-view", torch._neg_view(base), True, 10.0),
                 ("dtype", base.float(), True, 10.0),
                 ("clip", base, True, 9.0),
                 ("requires-grad", base.clone().requires_grad_(), True, 10.0))
        for name, value, enabled, limit in cases:
            with self.subTest(case=name), native_metadata_double(), patch.object(
                provider, "silu_mul_split_bf16_torch_exact", side_effect=AssertionError("unsupported native"),
            ) as exact:
                before = raw_bytes(value)
                metadata = view_metadata(value)
                assert_bits(self, call(value, enabled=enabled, limit=limit), old_activation(value, limit))
                exact.assert_not_called()
                self.assertTrue(torch.equal(raw_bytes(value), before))
                self.assertEqual(view_metadata(value), metadata)
        value = base.float().to(torch.complex64).conj()
        before, metadata = raw_bytes(value), view_metadata(value)
        with native_metadata_double(), patch.object(provider, "silu_mul_split_bf16_torch_exact", side_effect=AssertionError("conjugate native")):
            with self.assertRaises(RuntimeError) as original:
                call(value, enabled=False)
            with self.assertRaises(type(original.exception)) as actual:
                call(value, enabled=True)
            self.assertEqual(str(actual.exception), str(original.exception))
        self.assertTrue(torch.equal(raw_bytes(value), before))
        self.assertEqual(view_metadata(value), metadata)
        for configuration in (dict(ppu=False), dict(capability=(8, 0)),
                              dict(capture=True), dict(triton=None)):
            with self.subTest(configuration=configuration), native_metadata_double(**configuration), patch.object(
                provider, "silu_mul_split_bf16_torch_exact", side_effect=AssertionError("unsupported backend"),
            ):
                assert_bits(self, call(base, enabled=True), old_activation(base))
        with torch.enable_grad(), native_metadata_double(), patch.object(
            provider, "silu_mul_split_bf16_torch_exact", side_effect=AssertionError("grad native"),
        ):
            assert_bits(self, call(base, enabled=True), old_activation(base))
        with native_metadata_double(), patch.object(
            provider, "silu_mul_split_bf16_torch_exact", side_effect=lambda value, **kw: old_activation(value, kw["clamp_limit"]),
        ) as exact, patch.object(provider, "silu_mul_split_bf16", return_value=base[:, :N]) as fused:
            call(base, prefill_exact=True)
            exact.assert_called_once()
            call(base, fused_activation=True)
            fused.assert_called_once()
            # Unsupported local request retains explicit old flag precedence.
            call(base, enabled=True, limit=3.0, prefill_exact=True, fused_activation=True)
            self.assertEqual(exact.call_count, 2)
            self.assertEqual(fused.call_count, 1)
        with native_metadata_double(), patch.object(provider, "silu_mul_split_bf16_torch_exact", side_effect=RuntimeError("native launch failed")):
            with self.assertRaisesRegex(RuntimeError, "native launch failed"):
                call(base, enabled=True)

    @torch.no_grad()
    def test_local_phase_default_exact_cache_b1_and_all_phase_rollbacks(self):
        helper = layers._local_prefill_exact_shared_swiglu_enabled
        self.assertEqual(layers.LOCAL_PREFILL_EXACT_SHARED_SWIGLU_ENV, ENV)
        with local_phase_double():
            for flag in (None, "0", "true", "2"):
                with self.subTest(flag=flag), patch.dict(os.environ):
                    if flag is None:
                        os.environ.pop(ENV, None)
                    else:
                        os.environ[ENV] = flag
                    self.assertFalse(helper(local_state()))
            self.assertTrue(helper(local_state()))
            self.assertFalse(helper(None))
            for flag in ("idle", "is_graph", "is_graph_commit", "verify", "proposal"):
                state = local_state()
                setattr(state, flag, True)
                self.assertFalse(helper(state), flag)
            for attribute, value in (("prefill", False), ("lengths", []), ("lengths", [3, 4]),
                                     ("context", NS(cp_size=8, kv_cache_sharded=False)),
                                     ("context", NS(cp_size=1, kv_cache_sharded=True)),
                                     ("cache", NS(_is_prefill=True, sharded=False))):
                state = local_state()
                setattr(state, attribute, value)
                self.assertFalse(helper(state), attribute)
            for attribute, value in (("_is_decode_role", True), ("_is_speculative", True),
                                     ("_prefill_cp_size", 8), ("parallelism_config", NS(tp_size=8))):
                state = local_state()
                setattr(state.runtime.model, attribute, value)
                self.assertFalse(helper(state), attribute)
            for attribute in ("sharded", "_is_prefill"):
                state = local_state()
                setattr(state.cache, attribute, attribute == "sharded")
                self.assertFalse(helper(state), attribute)
            state = local_state()
            state.context = NS(cp_size=1, kv_cache_sharded=False)
            self.assertTrue(helper(state), "actual local compact CP1 stays eligible")
            state.attn.is_cuda_graph = True
            self.assertFalse(helper(state))
            for target, attribute, value in ((layers, "is_ppu", False),
                                              (torch.cuda, "is_current_stream_capturing", True),
                                              (attention_layer, "optimized_diagnostics_enabled", True),
                                              (_forward_tensor_debug, "enabled", True)):
                with patch.object(target, attribute, return_value=value):
                    self.assertFalse(helper(local_state()), attribute)
            with patch.object(_record_tensor, "ENABLED", True):
                self.assertFalse(helper(local_state()))
            with torch.enable_grad():
                self.assertFalse(helper(local_state()))

    @torch.no_grad()
    def test_actual_expert_and_moe_callers_only_add_strict_optional_keyword(self):
        expert = layers.PPUSharedExpert.__new__(layers.PPUSharedExpert)
        nn.Module.__init__(expert)
        expert.w13, expert.w2, expert.swiglu_limit = Mock(), Mock(), 10.0
        hidden = torch.ones(3, 32, dtype=torch.bfloat16)
        for local in (False, True, 1):
            with self.subTest(local=local), patch.object(layers, "shared_expert_forward", return_value=hidden) as forwarded:
                self.assertIs(expert(hidden, local_prefill_exact=local, prefill_exact=True, fused_activation=True), hidden)
                expected = dict(prefill_exact=True, fused_activation=True)
                if local is True:
                    expected["local_prefill_exact"] = True
                forwarded.assert_called_once_with(hidden, expert.w13, expert.w2, 10.0, **expected)
        state = local_state()
        state.lengths = [hidden.shape[0]]
        unit = layers.PPUSharedMoE.__new__(layers.PPUSharedMoE)
        nn.Module.__init__(unit)
        unit.routed_enabled, unit.calls, unit.layer_id = True, 0, 0
        unit.shared_expert = Mock(side_effect=lambda value, **kwargs: value)
        unit.gate = Mock(return_value=(torch.zeros(3, 6, dtype=torch.int64), torch.ones(3, 6)))
        unit.ops = NS(moe=Mock(side_effect=lambda value, *args, **kwargs: torch.zeros_like(value)))
        unit.runtime = state.runtime
        with local_phase_double(), patch.object(layers, "_full_cp8_prefill_context", return_value=None), patch.object(
            layers, "_local_prefill_direct_moe_real", return_value=None,
        ), patch.object(layers, "cp8_yoco_small_ops_enabled", return_value=False):
            for flag in ("0", "1"):
                with self.subTest(flag=flag), patch.dict(os.environ, {ENV: flag}):
                    unit.shared_expert.reset_mock()
                    assert_bits(self, unit(hidden), hidden)
                    expected = dict(prefill_exact=False, fused_activation=False)
                    if flag == "1":
                        expected["local_prefill_exact"] = True
                    unit.shared_expert.assert_called_once()
                    self.assertEqual(unit.shared_expert.call_args.kwargs, expected)
                    self.assertEqual(len(unit.shared_expert.call_args.args), 1)
                    actual = unit.shared_expert.call_args.args[0]
                    self.assertEqual(actual.data_ptr(), hidden.data_ptr())
                    self.assertEqual(view_metadata(actual)[1:], view_metadata(hidden)[1:])
                    assert_bits(self, actual, hidden, "run_shared flattened alias")


@unittest.skipUnless(os.environ.get("RTP_V41_TEST_LOCAL_WORKSPACE_DEVICE") == "1", "explicit PPU device test")
class LocalPrefillSharedSwiGLUDevice(unittest.TestCase):
    def setUp(self):
        self.assertTrue(_quant.is_ppu(), "requires an actual ZW PPU")
        self.assertEqual(torch.cuda.get_device_capability(), (8, 9))

    @torch.no_grad()
    def test_all_bf16_encodings_and_partner_families_exact_bytes(self):
        for name, host in all_encodings_cases():
            with self.subTest(case=name):
                value = host.to("cuda")
                self.assertTrue(torch.equal(raw_bytes(value).cpu(), raw_bytes(host)),
                                "the transfer must exercise every original NaN payload")
                before = raw_bytes(value)
                expected = old_activation(value)
                with patch.object(provider, "silu_mul_split_bf16_torch_exact", wraps=provider.silu_mul_split_bf16_torch_exact) as exact:
                    actual = call(value, enabled=True)
                exact.assert_called_once_with(value, clamp_limit=10.0)
                assert_bits(self, actual, expected, name)
                self.assertTrue(torch.equal(raw_bytes(value), before))
                self.assertNotEqual(actual.untyped_storage().data_ptr(), value.untyped_storage().data_ptr())

    @torch.no_grad()
    def test_actual_sizes_offset_canaries_views_and_empty_fallback(self):
        for rows in (0, 1, 140, 267, 4096, 4097):
            with self.subTest(rows=rows):
                offset, tail = 23, 37
                backing = torch.full((rows * WIDTH + offset + tail,), -123.0, device="cuda", dtype=torch.bfloat16)
                value = backing[offset:offset + rows * WIDTH].reshape(rows, WIDTH)
                value.copy_(edge_input(rows, "cuda"))
                before = raw_bytes(backing)
                with patch.object(provider, "silu_mul_split_bf16_torch_exact", wraps=provider.silu_mul_split_bf16_torch_exact) as exact:
                    actual = call(value, enabled=True)
                self.assertEqual(exact.call_count, int(rows > 0))
                assert_bits(self, actual, old_activation(value), f"M{rows}")
                self.assertTrue(torch.equal(raw_bytes(backing), before))
                if rows:
                    self.assertNotEqual(actual.untyped_storage().data_ptr(), backing.untyped_storage().data_ptr())
        base = edge_input(3, "cuda")
        for value, limit in ((torch._neg_view(base), 10.0), (base[:, ::2], 10.0),
                             (base.float(), 10.0), (base[:, :128].contiguous(), 10.0),
                             (base, 0.0), (base, 9.0)):
            with self.subTest(shape=tuple(value.shape), dtype=value.dtype, limit=limit), patch.object(
                provider, "silu_mul_split_bf16_torch_exact", side_effect=AssertionError("unsupported native"),
            ):
                before = raw_bytes(base)
                metadata = view_metadata(value)
                assert_bits(self, call(value, enabled=True, limit=limit), old_activation(value, limit))
                self.assertTrue(torch.equal(raw_bytes(base), before))
                self.assertEqual(view_metadata(value), metadata)
        value = base.float().to(torch.complex64).conj()
        before, metadata = raw_bytes(value), view_metadata(value)
        with patch.object(provider, "silu_mul_split_bf16_torch_exact", side_effect=AssertionError("conjugate native")):
            with self.assertRaises(RuntimeError) as original:
                call(value, enabled=False)
            with self.assertRaises(type(original.exception)) as actual:
                call(value, enabled=True)
            self.assertEqual(str(actual.exception), str(original.exception))
        self.assertTrue(torch.equal(raw_bytes(value), before))
        self.assertEqual(view_metadata(value), metadata)

    @torch.no_grad()
    def test_real_w13_w2_ptpc_and_bf16_projection_boundaries_exact(self):
        torch.manual_seed(812)
        weight13 = (torch.randn(WIDTH, 5120, device="cuda") * 0.25).to(torch.float8_e4m3fn)
        weight2 = (torch.randn(5120, N, device="cuda") * 0.25).to(torch.float8_e4m3fn)
        scale13 = torch.full((WIDTH, 1), 0.125, device="cuda")
        scale2 = torch.full((5120, 1), 0.125, device="cuda")
        w13, w2 = ProbeLinear(PPUPTPCLinear(weight13, scale13)), ProbeLinear(PPUPTPCLinear(weight2, scale2))
        expert = layers.PPUSharedExpert.__new__(layers.PPUSharedExpert)
        nn.Module.__init__(expert)
        expert.w13, expert.w2, expert.swiglu_limit = w13, w2, 10.0
        weight_snapshots = [raw_bytes(value) for value in (weight13, weight2, scale13, scale2)]
        for rows in (140, 4096, 4097):
            with self.subTest(rows=rows):
                value = (torch.randn(rows, 5120, device="cuda") * 0.5).bfloat16()
                input_before = raw_bytes(value)
                records = []
                outputs = []
                for enabled in (False, True):
                    w13.records.clear()
                    w2.records.clear()
                    with patch.object(provider, "silu_mul_split_bf16_torch_exact", wraps=provider.silu_mul_split_bf16_torch_exact) as exact:
                        outputs.append(expert(value, local_prefill_exact=enabled))
                    self.assertEqual(exact.call_count, int(enabled))
                    records.append((w13.records[0], w2.records[0]))
                for projection, label in enumerate(("W13", "W2")):
                    for boundary, kind in enumerate(("BF16 input", "FP8 payload", "FP32 scales", "BF16 output")):
                        assert_bits(self, records[1][projection][boundary], records[0][projection][boundary], f"{label} {kind} M{rows}")
                assert_bits(self, outputs[1], outputs[0], f"projected M{rows}")
                for actual, expected in zip(quantize_ptpc_activation(outputs[1]), quantize_ptpc_activation(outputs[0])):
                    assert_bits(self, actual, expected, "following PTPC boundary")
                self.assertTrue(torch.equal(raw_bytes(value), input_before))
                del records, outputs
        for value, before in zip((weight13, weight2, scale13, scale2), weight_snapshots):
            self.assertTrue(torch.equal(raw_bytes(value), before))
        # Also preserve the BF16-resident linear ABI rather than assuming all
        # shared projections use PTPC. These are real ATen device GEMMs.
        value = torch.randn(140, 64, device="cuda", dtype=torch.bfloat16)
        resident13 = PPUResidentBF16Linear(torch.randn(WIDTH, 64, device="cuda", dtype=torch.bfloat16))
        resident2 = PPUResidentBF16Linear(torch.randn(64, N, device="cuda", dtype=torch.bfloat16))
        assert_bits(self, moe.shared_expert_forward(value, resident13, resident2, local_prefill_exact=True),
                    moe.shared_expert_forward(value, resident13, resident2), "resident BF16 W13/W2")

    @torch.no_grad()
    def test_actual_profile_seven_to_four_keeps_aten_silu_mul_and_landing(self):
        for rows in (4096, 140):
            value = edge_input(rows, "cuda")
            call(value, enabled=False)
            call(value, enabled=True)
            outputs = []
            for enabled in (False, True):
                torch.cuda.synchronize()
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                        torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as profile:
                    outputs.append(call(value, enabled=enabled))
                    torch.cuda.synchronize()
                events = _trace_events(profile, f"shared-swiglu-M{rows}-flag{int(enabled)}")
                kernels = [event["name"] for event in events if event.get("cat") == "kernel"]
                cpu = [event for event in events if event.get("cat") == "cpu_op"]
                self.assertEqual(sum("_clamp_widen_gate_up_kernel" in name for name in kernels), int(enabled), kernels)
                self.assertEqual(len(kernels), 4 if enabled else 7, kernels)
                self.assertEqual(sum(event["name"] == "aten::clamp" for event in cpu), 0 if enabled else 2)
                self.assertEqual(sum(event["name"] == "aten::_to_copy" for event in cpu), 1 if enabled else 3)
                self.assertEqual(sum(event["name"] == "aten::silu" for event in cpu), 1)
                self.assertEqual(sum(event["name"] == "aten::mul" for event in cpu), 1)
                self.assertFalse(any("_silu_mul_split_bf16_kernel" in name for name in kernels))
            assert_bits(self, outputs[1], outputs[0], f"profile M{rows}")


if __name__ == "__main__":
    unittest.main()
