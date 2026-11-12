"""SG-aligned primitives: BF16 SiLU landing, BF16 multiply, row E4M3 quant.

The old FP32 activation is a timing baseline, not this opt-in's byte oracle.
No device tests run without the explicit PPU qualification environment flag.
"""

from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
import unittest
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F

from rtp_llm.models_py.kernels.ppu.jit import sglang_jit
from rtp_llm.models_py.modules.dsv41.ppu import local_prefill_sg_ops as ops
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_operator_flags import _trace_events
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_token_topk_variants import _artifact_directory


SG_COMMIT = "3bc4f87d5f9bb6f04b724540fb12a3acdc62d2fa"
SG_HEADER_SHA = "c2fe1a44a1309d05efc06c516d81839a0517254596ac2262bc9effdbe905c528"
BUNDLE = Path(sglang_jit.__file__).with_name("dsv4_ppu") / "prefill_swiglu"
HEADER = "compat/elementwise/silu_and_mul_post_per_token_quant_fp8.cuh"
SYMBOL = "sglang::SiluMulFp8TP<256,true,false>::run"


def assert_bits(case, actual, expected):
    case.assertEqual(actual.shape, expected.shape)
    case.assertEqual(actual.dtype, expected.dtype)
    case.assertTrue(torch.equal(actual.contiguous().view(torch.uint8),
                                expected.contiguous().view(torch.uint8)))


def sg_activation_reference(gateup, limit=10.0):
    """Independent finite-value reference for the SG header's two BF16 rounds.

    CPU sigmoid is independent of __ppu_sgmdf. IEEE clamp/min intrinsic
    behavior is compared to the actual SG native reference separately.
    """
    gate, up = gateup.chunk(2, dim=-1)
    gate = gate.clamp(max=limit)
    up = up.clamp(-limit, limit)
    silu_bf16 = F.silu(gate.float()).to(torch.bfloat16)
    return (silu_bf16.float() * up.float()).to(torch.bfloat16)


def direct_sg_reference(gateup):
    """Launch the pinned SG implementation directly, outside the RTP adapter.

    A distinct module binding verifies the adapter's ABI/outputs/parameters;
    this is the same SG algorithm, not an independent numerical algorithm.
    """
    module = sglang_jit.load_sglang_kernel(
        "qualification_sg_silu_mul_fp8", HEADER, SYMBOL,
        torch.cuda.get_device_capability(gateup.device), ("-use_fast_math",),
        bundle="prefill_swiglu",
    )
    payload = torch.empty((gateup.shape[0], 2304), dtype=torch.float8_e4m3fn, device=gateup.device)
    scale = torch.empty((gateup.shape[0], 1), dtype=torch.float32, device=gateup.device)
    if gateup.shape[0]:
        with torch.cuda.device(gateup.device):
            module.forward(gateup, payload, scale, 10.0, 0.0, 0.0, 1e-10)
    return payload, scale


class LocalPrefillSGOpsCPU(unittest.TestCase):
    def setUp(self):
        guard = torch.no_grad()
        guard.__enter__()
        self.addCleanup(guard.__exit__, None, None, None)

    def test_metadata_declines_before_provider_and_device_checks_are_real(self):
        good = torch.randn(3, 4608, dtype=torch.bfloat16)
        self.assertFalse(ops.supports_swiglu(good, 10.0))
        with patch.object(ops, "_supported_device", new=lambda value: True):
            self.assertTrue(ops.supports_swiglu(good, 10.0))
            for value, limit in (
                (good.float(), 10.0), (good[:, ::2], 10.0),
                (torch._neg_view(good), 10.0), (good.to_sparse(), 10.0),
                (torch.empty(16385, 4608, dtype=torch.bfloat16), 10.0),
                (good, 0.0), (good, None), (good, True), (good, float("nan")),
                (torch.empty(2 * 4608 + 1, dtype=torch.bfloat16)[1:].view(2, 4608), 10.0),
            ):
                with self.subTest(shape=value.shape, limit=limit), patch.object(
                    ops, "_swiglu_module", side_effect=AssertionError("unsupported native")
                ):
                    self.assertFalse(ops.supports_swiglu(value, limit))
                    with self.assertRaises(ValueError):
                        ops.swiglu_quant(value, limit)
            with torch.enable_grad():
                self.assertFalse(ops.supports_swiglu(good, 10.0))
            self.assertFalse(ops.supports_hc(good))
        from rtp_llm.models_py.modules.dsv41.ppu import _quant
        fake = type("DeviceMetadata", (), dict(is_cuda=True, device=torch.device("cuda:0")))()
        for arch, capture, ppu, expected in (((8, 9), False, True, True),
                                              ((8, 0), False, True, False),
                                              ((8, 9), True, True, False),
                                              ((8, 9), False, False, False)):
            with patch.object(_quant, "is_ppu", return_value=ppu), patch.object(
                torch.cuda, "device", new=lambda device: nullcontext()
            ), patch.object(torch.cuda, "get_device_capability", return_value=arch), patch.object(
                torch.cuda, "is_current_stream_capturing", return_value=capture
            ), patch.object(torch.version, "hip", None):
                self.assertIs(ops._supported_device(fake), expected)

    def test_abi_owned_outputs_empty_and_native_error_propagation(self):
        source = torch.randn(2, 4608, dtype=torch.bfloat16)
        snapshot = source.clone()
        module = Mock()
        module.forward.side_effect = lambda x, q, s, *args: (q.view(torch.uint8).zero_(), s.fill_(1))
        with patch.object(ops, "_supported_device", new=lambda value: True), patch.object(
            ops, "_device_context", new=lambda value: nullcontext()
        ), patch.object(torch.cuda, "get_device_capability", return_value=(8, 9)), patch.object(
            ops, "_swiglu_module", return_value=module
        ) as load:
            q, scale = ops.swiglu_quant(source, 10)
            self.assertEqual((q.shape, q.dtype), ((2, 2304), torch.float8_e4m3fn))
            self.assertEqual((scale.shape, scale.dtype), ((2, 1), torch.float32))
            self.assertEqual(module.forward.call_args.args[3:], (10.0, 0.0, 0.0, 1e-10))
            self.assertIs(module.forward.call_args.args[0], source)
            self.assertIs(module.forward.call_args.args[1], q)
            self.assertIs(module.forward.call_args.args[2], scale)
            self.assertNotEqual(source.untyped_storage().data_ptr(), q.untyped_storage().data_ptr())
            assert_bits(self, source, snapshot)
            empty_q, empty_s = ops.swiglu_quant(source[:0], 10.0)
            self.assertEqual((empty_q.shape, empty_s.shape), ((0, 2304), (0, 1)))
            load.assert_called_once()
            module.forward.side_effect = RuntimeError("native failure")
            with self.assertRaisesRegex(RuntimeError, "native failure"):
                ops.swiglu_quant(source, 10.0)
        hidden = torch.randn(2, 5120, dtype=torch.bfloat16)
        with patch.object(ops, "_supported_device", new=lambda value: True), patch.object(
            ops, "_device_context", new=lambda value: nullcontext()
        ), patch.object(ops, "_expand_hc", side_effect=lambda value: value[:, None].repeat(1, 4, 1)) as expand:
            actual = ops.hc_broadcast(hidden)
            assert_bits(self, actual, hidden[:, None].repeat(1, 4, 1))
            self.assertEqual(ops.hc_broadcast(hidden[:0]).shape, (0, 4, 5120))
            expand.assert_called_once_with(hidden)

    def test_frozen_bundle_binding_and_host_only_compatibility(self):
        manifest = json.loads((BUNDLE / "source-manifest.json").read_text())
        self.assertEqual(manifest["upstream"]["commit"], SG_COMMIT)
        for name, expected in manifest["sources"].items():
            self.assertEqual(hashlib.sha256((BUNDLE / name).read_bytes()).hexdigest(), expected)
        original = BUNDLE / "csrc/elementwise/silu_and_mul_post_per_token_quant_fp8.cuh"
        self.assertEqual(hashlib.sha256(original.read_bytes()).hexdigest(), SG_HEADER_SHA)
        runtime = (BUNDLE / HEADER).read_text()
        self.assertEqual(runtime.replace("reinterpret_cast<const void*>", "std::bit_cast<const void*>"), original.read_text())
        ops._swiglu_module.cache_clear()
        self.addCleanup(ops._swiglu_module.cache_clear)
        with patch.object(ops, "load_sglang_kernel", return_value=object()) as load:
            ops._swiglu_module((8, 9))
        load.assert_called_once_with("prefill_silu_mul_fp8", HEADER, SYMBOL, (8, 9),
                                     ("-use_fast_math",), bundle="prefill_swiglu")

    def test_sg_oracle_has_two_bf16_rounds_and_no_legacy_identity_requirement(self):
        gate = torch.linspace(-12, 12, 2304).to(torch.bfloat16)
        up = torch.linspace(-9, 9, 2304).roll(137).to(torch.bfloat16)
        source = torch.cat((gate, up))[None]
        expected = (F.silu(gate.clamp(max=10).float()).to(torch.bfloat16).float()
                    * up.clamp(-10, 10).float()).to(torch.bfloat16)[None]
        assert_bits(self, sg_activation_reference(source), expected)
        legacy = (F.silu(gate.clamp(max=10).float()) * up.clamp(-10, 10).float()).to(torch.bfloat16)[None]
        self.assertFalse(torch.equal(expected.view(torch.uint8), legacy.view(torch.uint8)))


@unittest.skipUnless(os.environ.get("RTP_V41_TEST_LOCAL_WORKSPACE_DEVICE") == "1", "explicit PPU device test")
class LocalPrefillSGOpsDevice(unittest.TestCase):
    def setUp(self):
        from rtp_llm.models_py.modules.dsv41.ppu._quant import is_ppu
        self.assertTrue(is_ppu())
        self.assertEqual(torch.cuda.get_device_capability(), (8, 9))
        self.assertEqual(torch.cuda.current_device(), 0)
        self.assertEqual(torch.cuda.current_stream(), torch.cuda.default_stream())
        guard = torch.no_grad()
        guard.__enter__()
        self.addCleanup(guard.__exit__, None, None, None)
        torch.manual_seed(121112)

    def _compare_native(self, source):
        before = source.clone()
        actual = ops.swiglu_quant(source, 10.0)
        expected = direct_sg_reference(source)
        for got, ref in zip(actual, expected):
            assert_bits(self, got, ref)
            self.assertTrue(got.is_contiguous())
            self.assertNotEqual(got.untyped_storage().data_ptr(), source.untyped_storage().data_ptr())
        assert_bits(self, source, before)
        return actual

    def test_actual_4096_4097_140_native_bytes_and_independent_cpu_sg_reference(self):
        for rows in (4096, 4097, 140):
            with self.subTest(rows=rows):
                source = (torch.randn(rows, 4608, dtype=torch.bfloat16, device="cuda:0") * 4).contiguous()
                payload, scale = self._compare_native(source)
                self.assertEqual(payload.shape, (rows, 2304))
                self.assertEqual(scale.shape, (rows, 1))
                reference = sg_activation_reference(source.cpu()).float()
                dequant = payload.float().cpu() * scale.cpu()
                # Existing pinned SG test's dequantized normalized-row-absmax
                # bound, not a new tolerance or the legacy RTP FP32 oracle.
                denom = reference.abs().amax(-1, keepdim=True).clamp_min(1e-6)
                self.assertLess(((dequant - reference).abs() / denom).max().item(), 8e-2)

    def test_ieee_empty_unsupported_layouts_input_canaries_and_no_alias(self):
        bank = torch.full((10, 4608), 3, dtype=torch.bfloat16, device="cuda:0")
        source = bank[1:-1]
        bits = torch.tensor([0, -32768, 1, -32767, 127, -32641, 128, -32640,
                             32639, -129, 32640, -128, 32705, -63, 32767, -1],
                            dtype=torch.int16, device=source.device).view(torch.bfloat16)
        source.copy_(bits.repeat(source.numel() // len(bits)).view_as(source))
        before = bank.clone()
        # NaN/min/max/infinity behavior is the real SG header's contract;
        # Torch clamp's NaN behavior is not substituted for BF16 intrinsics.
        self._compare_native(source)
        assert_bits(self, bank, before)
        with patch.object(ops, "_swiglu_module", side_effect=AssertionError("empty launch")):
            q, scale = ops.swiglu_quant(source[:0], 10)
        self.assertEqual((q.shape, scale.shape), ((0, 2304), (0, 1)))
        self.assertEqual(ops.hc_broadcast(torch.empty(0, 5120, device=source.device, dtype=source.dtype)).shape,
                         (0, 4, 5120))
        for value in (source.float(), source[:, ::2], torch._neg_view(source), source.to_sparse()):
            with patch.object(ops, "_swiglu_module", side_effect=AssertionError("unsupported launch")):
                with self.assertRaises(ValueError):
                    ops.swiglu_quant(value, 10.0)
        with patch.object(ops, "_expand_hc", side_effect=AssertionError("unsupported HC")):
            with self.assertRaises(ValueError):
                ops.hc_broadcast(source)

    def test_hc_full_odd_compact_exact_layout_storage_and_independent_lanes(self):
        for rows in (4096, 4097, 140):
            with self.subTest(rows=rows):
                bits = torch.tensor([0, -32768, 1, -32767, 32640, -128, 32705, -63],
                                    dtype=torch.int16, device="cuda:0").view(torch.bfloat16)
                source = bits.repeat(rows * 5120 // len(bits)).view(rows, 5120)
                before = source.clone()
                actual = ops.hc_broadcast(source)
                self.assertEqual(actual.shape, (rows, 4, 5120))
                self.assertEqual(actual.stride(), (20480, 5120, 1))
                self.assertNotEqual(actual.untyped_storage().data_ptr(), source.untyped_storage().data_ptr())
                assert_bits(self, actual.cpu(), source.cpu()[:, None].repeat(1, 4, 1))
                other = actual[:, 1].clone()
                actual[:, 0].fill_(123)
                assert_bits(self, actual[:, 1], other)
                assert_bits(self, source, before)

    def test_real_fused_and_hc_kernel_profiles_have_no_aten_auxiliary_chain(self):
        records = {}
        for rows in (4096, 140):
            source = torch.randn(rows, 4608, dtype=torch.bfloat16, device="cuda:0")
            ops.swiglu_quant(source, 10.0)
            torch.cuda.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as profile:
                outputs = ops.swiglu_quant(source, 10.0)
                torch.cuda.synchronize()
            events = _trace_events(profile, f"sg-swiglu-{rows}")
            kernels = [e["name"] for e in events if e.get("cat") == "kernel"]
            self.assertEqual(len(kernels), 1)
            self.assertIn("silu_and_mul_post_per_token_quant_fp8_kernel", kernels[0])
            self.assertFalse(any(e.get("name") in ("aten::silu", "aten::mul", "aten::clamp", "aten::_to_copy") for e in events))
            self.assertFalse(any(e.get("cat") == "gpu_memcpy" and "DtoH" in e.get("name", "") for e in events))
            records[str(rows)] = dict(kernels=kernels, output_shapes=[list(t.shape) for t in outputs])
        source = torch.randn(4096, 5120, dtype=torch.bfloat16, device="cuda:0")
        ops.hc_broadcast(source)
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as profile:
            output = ops.hc_broadcast(source)
            torch.cuda.synchronize()
        events = _trace_events(profile, "sg-hc-broadcast-4096")
        kernels = [e["name"] for e in events if e.get("cat") == "kernel"]
        self.assertEqual(kernels, ["_expand_hc_kernel"])
        self.assertFalse(any(e.get("name") in ("aten::repeat", "aten::copy_", "aten::clone") for e in events))
        records["hc"] = dict(kernels=kernels, shape=list(output.shape), stride=list(output.stride()))
        with _artifact_directory() as directory:
            (directory / "local-prefill-sg-ops.profile.json").write_text(json.dumps(dict(
                sg_commit=SG_COMMIT, original_header_sha256=SG_HEADER_SHA, records=records,
                hc_provider="existing RTP expand_hc; SG-equivalent bit transport with different launch tiling",
            ), indent=2))

    def test_interleaved_microtiming_uses_complete_primitive_cost(self):
        from rtp_llm.models_py.modules.dsv41.moe._silu_mul_bf16_triton import silu_mul_split_bf16_torch_exact
        from rtp_llm.models_py.modules.dsv41.ppu.ptpc_quant import quantize_ptpc_activation

        samples = []
        cases = [(f"swiglu-{rows}", torch.randn(rows, 4608, device="cuda:0", dtype=torch.bfloat16),
                  lambda x: quantize_ptpc_activation(silu_mul_split_bf16_torch_exact(x, 10.0)),
                  lambda x: ops.swiglu_quant(x, 10.0)) for rows in (4096, 140)]
        cases.append(("hc-4096", torch.randn(4096, 5120, device="cuda:0", dtype=torch.bfloat16),
                      lambda x: x[:, None].repeat(1, 4, 1), ops.hc_broadcast))
        for name, source, old, new in cases:
            functions = {"old": old, "sg": new}
            for function in functions.values():
                for _ in range(2):
                    function(source)
            torch.cuda.synchronize()
            for position, arm in enumerate(("old", "sg", "sg", "old") * 4):
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                begin.record()
                start = time.perf_counter_ns()
                result = functions[arm](source)
                end.record()
                end.synchronize()
                elapsed = (time.perf_counter_ns() - start) / 1e6
                gpu = begin.elapsed_time(end)
                self.assertTrue(math.isfinite(elapsed) and elapsed > 0)
                self.assertTrue(math.isfinite(gpu) and gpu > 0)
                samples.append(dict(case=name, arm=arm, position=position, gpu_ms=gpu, host_ms=elapsed))
                del result
        self.assertEqual(len(samples), 48)
        medians = {f"{name}/{arm}": {metric: statistics.median(s[metric] for s in samples
                    if s["case"] == name and s["arm"] == arm) for metric in ("gpu_ms", "host_ms")}
                   for name, *_ in cases for arm in ("old", "sg")}
        with _artifact_directory() as directory:
            (directory / "local-prefill-sg-ops.samples.json").write_text(json.dumps(dict(
                samples=samples, medians=medians, sg_commit=SG_COMMIT, original_header_sha256=SG_HEADER_SHA,
                scope="complete primitive allocations/activation/quant or broadcast; warmup/import/input excluded",
                semantics="old FP32 activation and SG BF16 two-round activation intentionally differ",
                decision="no speed assertion; not model wall time", seed=121112,
            ), indent=2))


if __name__ == "__main__":
    unittest.main()
