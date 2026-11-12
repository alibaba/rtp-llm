"""Test-only checked versus existing FP32 hybrid, with full wrapper costs.

No serving dispatch changes. Widen occurs inside the experimental selector;
the original BF16 sanitizer, bounds, finite finish and exact oracle remain.
"""

from contextlib import contextmanager, nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
import unittest
from unittest.mock import Mock, call, patch

import torch

from rtp_llm.models_py.modules.dsv41.fp8.attention_v41 import select_candidate_blocks
from rtp_llm.models_py.modules.dsv41.fp8.test.test_v41_prefill_topk import assert_equivalent, reference
from rtp_llm.models_py.modules.dsv41.ppu import _quant, prefill_token_topk as token
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_operator_flags import _trace_events
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_token_topk import (
    _adversarial, _assert_oracle, _BoundsDouble, _bytes_equal, _finish_double,
    _native_double, _raw_bytes,
)
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_token_topk_variants import (
    _actual_inputs, _artifact_directory, _source_hashes as _original_source_hashes,
)
from rtp_llm.ops.compute_ops import rtp_llm_ops


ARMS = ("checked", "fp32_hybrid")
CASES = ("causal_style", "wide_fallback")


def _hybrid_select(native, *, device_context=None):
    def select(logits, starts, ends, out):
        # The C++ op takes its current-device current stream. Both this cast
        # and that launch must use the score's device, including cuda:N.
        context = torch.cuda.device if device_context is None else device_context
        with context(logits.device):
            widened = logits.float()
            native(widened, starts, ends, out, widened.shape[0],
                   widened.stride(0), widened.stride(1), 512, False)
        return out
    return select


def _resolve(arm):
    if arm == "checked":
        return token._native_ops(False)
    if arm != "fp32_hybrid":
        raise ValueError(arm)
    native = getattr(rtp_llm_ops, "dsv4_top_k_per_row_prefill", None)
    if not callable(native):
        raise ImportError("existing dsv4_top_k_per_row_prefill is unavailable")
    from rtp_llm.models_py.kernels.ppu.attention.ppu_bf16_radix_topk import canonicalize_topk_indices

    return _hybrid_select(native), canonicalize_topk_indices


@contextmanager
def _variant(arm, *, ops=None):
    if arm not in ARMS:
        raise ValueError(arm)
    # Imports/provider resolution finish before timing. Widen is deliberately
    # inside select, never cached or prepared outside the measured wrapper.
    resolved = _resolve(arm) if ops is None else ops
    seen = []

    def load(prepared):
        if prepared is not False:
            raise AssertionError("experimental arms require bounds=None")
        seen.append(prepared)
        return resolved

    with patch.object(token, "_native_ops", new=load):
        yield seen


def _source_hashes():
    return dict(_original_source_hashes(), **{
        __name__: hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    })


class LocalPrefillNativeFP32TopKVariantsCPU(unittest.TestCase):
    def setUp(self):
        for item in (
            patch.dict(os.environ, {token.ENV: "1"}),
            patch.object(token, "_device_supported", return_value=True),
            patch.object(token._topk, "_prefill_topk_bounds_kernel", _BoundsDouble()),
            patch.object(token._topk, "finish_tokens", side_effect=_finish_double),
        ):
            item.start()
            self.addCleanup(item.stop)
        guard = torch.no_grad()
        guard.__enter__()
        self.addCleanup(guard.__exit__, None, None, None)

    def test_same_wrapper_widen_cpp_abi_device_context_and_scope_restore(self):
        base, visible = _adversarial("cpu", 1024)
        sanitized = base.nan_to_num(nan=torch.inf, posinf=torch.inf, neginf=-torch.inf)
        original = token._native_ops
        contexts, outputs = [], []

        def context(device):
            contexts.append(device)
            return nullcontext()

        def native(x32, starts, ends, out, rows, stride0, stride1, k, force):
            self.assertEqual(x32.dtype, torch.float32)
            self.assertEqual((rows, stride0, stride1, k, force), (24, 1024, 1, 512, False))
            _bytes_equal(self, x32, sanitized.float())
            return _native_double(x32, starts, ends, out)

        cpp = Mock(side_effect=native)
        canonicalize = Mock(side_effect=lambda out: out)
        for arm in ARMS:
            select = _native_double if arm == "checked" else _hybrid_select(cpp, device_context=context)
            finish_order = canonicalize if arm == "fp32_hybrid" else None
            scores = base.clone()
            with _variant(arm, ops=(select, finish_order)) as seen:
                got = token.try_select_tokens(scores, visible, disposable=True)
            self.assertEqual(seen, [False])
            self.assertIs(token._native_ops, original)
            _bytes_equal(self, scores, sanitized)
            assert_equivalent(self, got, reference(base, visible), base, visible)
            _assert_oracle(self, got, base, visible)
            outputs.append(got)
        cpp.assert_called_once()
        canonicalize.assert_called_once()
        self.assertEqual(contexts, [base.device])
        _, starts, ends, *_ = cpp.call_args.args
        self.assertTrue(torch.equal(starts, torch.zeros_like(ends)))
        self.assertTrue(torch.equal(ends, visible.clamp(0, 1024).int()))
        self.assertEqual(token._topk.finish_tokens.call_count, 2)
        assert_equivalent(self, outputs[1], outputs[0], base, visible)

    def test_missing_import_native_error_and_unsupported_do_not_retry_or_mutate(self):
        base = torch.full((2, 1024), torch.nan, dtype=torch.bfloat16)
        visible = torch.full((2,), 1024, dtype=torch.int64)
        original = token._native_ops
        with patch.object(rtp_llm_ops, "dsv4_top_k_per_row_prefill", None, create=True):
            with self.assertRaisesRegex(ImportError, "unavailable"):
                _resolve("fp32_hybrid")
        with patch.object(token, "_native_ops", side_effect=ImportError("unavailable")):
            self.assertIsNone(token.try_select_tokens(base, visible, disposable=True))
        _bytes_equal(self, base, torch.full_like(base, torch.nan))
        for arm in ARMS:
            failed = Mock(side_effect=RuntimeError("native launch failed"))
            select = failed if arm == "checked" else _hybrid_select(
                failed, device_context=lambda device: nullcontext()
            )
            scores = base.clone()
            with _variant(arm, ops=(select, None)):
                with self.assertRaisesRegex(RuntimeError, "native launch failed"):
                    token.try_select_tokens(scores, visible, disposable=True)
            failed.assert_called_once()
            self.assertTrue(scores.isposinf().all())
            self.assertIs(token._native_ops, original)
        for scores, options in (
            (base, {"disposable": False}), (base.float(), {"disposable": True}),
            (base[:, ::2], {"disposable": True}),
            (torch._neg_view(base), {"disposable": True}),
            (base, {"disposable": True, "out": torch.empty(2, 1024, dtype=torch.int32)[:, ::2]}),
        ):
            watched = (scores, visible) + ((options["out"],) if "out" in options else ())
            before = [_raw_bytes(t).clone() for t in watched]
            with patch.object(token, "_native_ops", side_effect=AssertionError("unsupported work")):
                self.assertIsNone(token.try_select_tokens(scores, visible, **options))
            for tensor, snapshot in zip(watched, before):
                self.assertTrue(torch.equal(_raw_bytes(tensor), snapshot))
        self.assertIs(token._native_ops, original)

    def test_default_dispatch_and_candidate_2048_publication_are_unchanged(self):
        base = torch.randn(2, 16400, dtype=torch.bfloat16)
        base[0, 0] = torch.nan
        visible = torch.full((2,), 16400, dtype=torch.int64)
        candidates = select_candidate_blocks(base, visible, 8, 2048)
        published = candidates.clone()
        default = Mock(return_value=(_native_double, None))
        with patch.object(token, "_native_ops", new=default):
            token.try_select_tokens(base.clone(), visible, disposable=True)
            with _variant("fp32_hybrid", ops=(_native_double, lambda out: out)):
                token.try_select_tokens(base.clone(), visible, disposable=True)
            token.try_select_tokens(base.clone(), visible, disposable=True)
            self.assertIs(token._native_ops, default)
            with patch.dict(os.environ, {token.ENV: "0"}):
                scores = base.clone()
                self.assertIsNone(token.try_select_tokens(scores, visible, disposable=True))
                _bytes_equal(self, scores, base)
        self.assertEqual(default.call_args_list, [call(False), call(False)])
        self.assertEqual(candidates.shape, (2, 2048))
        _bytes_equal(self, candidates, published)
        self.assertEqual(token.K, 512)


@unittest.skipUnless(os.environ.get("RTP_V41_TEST_LOCAL_WORKSPACE_DEVICE") == "1", "explicit PPU device test")
class LocalPrefillNativeFP32TopKVariantsDevice(unittest.TestCase):
    def setUp(self):
        self.assertTrue(_quant.is_ppu())
        self.assertEqual(torch.cuda.get_device_capability(), (8, 9))
        self.assertEqual(torch.cuda.current_device(), 0)
        self.assertEqual(torch.cuda.current_stream(), torch.cuda.default_stream())
        self.assertTrue(callable(getattr(rtp_llm_ops, "dsv4_top_k_per_row_prefill", None)))
        env = patch.dict(os.environ, {token.ENV: "1"})
        env.start()
        self.addCleanup(env.stop)
        guard = torch.no_grad()
        guard.__enter__()
        self.addCleanup(guard.__exit__, None, None, None)
        torch.manual_seed(1919)

    def _paired(self, base, visible, *, padded=False, complete=False, inspect_widen=False):
        outputs, widened = {}, []
        sanitized = base.nan_to_num(nan=torch.inf, posinf=torch.inf, neginf=-torch.inf)
        original_visible = visible.clone()
        rows, width = base.shape
        for arm in ARMS:
            stride = ((width + 128 + 15) // 16) * 16
            bank = torch.full((rows + 2, stride), 7, dtype=base.dtype, device=base.device) if padded else None
            scores = bank[1:rows + 1, :width] if padded else base.clone()
            if padded:
                scores.copy_(base)
            out_bank = torch.full((rows * 512 + 16,), -77, dtype=torch.int32, device=base.device)
            out = out_bank[8:-8].view(rows, 512)
            ops = None
            if inspect_widen and arm == "fp32_hybrid":
                native = rtp_llm_ops.dsv4_top_k_per_row_prefill

                def observed(x32, *args):
                    widened.append(x32)
                    return native(x32, *args)

                _, canonicalize = _resolve(arm)
                ops = (_hybrid_select(observed), canonicalize)
            with _variant(arm, ops=ops) as seen:
                got = token.try_select_tokens(scores, visible, out=out, disposable=True)
            self.assertEqual(seen, [False])
            self.assertIs(got, out)
            _bytes_equal(self, scores, sanitized)
            _bytes_equal(self, visible, original_visible)
            self.assertTrue((out_bank[:8] == -77).all() and (out_bank[-8:] == -77).all())
            if padded:
                self.assertTrue((bank[0] == 7).all() and (bank[-1] == 7).all())
                self.assertTrue((bank[1:-1, width:] == 7).all())
            outputs[arm] = got.clone()
        assert_equivalent(self, outputs["fp32_hybrid"], outputs["checked"], base, visible)
        assert_equivalent(self, outputs["checked"], outputs["fp32_hybrid"], base, visible)
        selected = None if complete else sorted({0, rows // 2, rows - 1})
        for got in outputs.values():
            _assert_oracle(self, got, base, visible, rows=selected)
        if complete:
            expected = reference(base, visible)
            for got in outputs.values():
                assert_equivalent(self, got, expected, base, visible)
        if inspect_widen:
            self.assertEqual(len(widened), 1)
            self.assertEqual(widened[0].dtype, torch.float32)
            self.assertEqual(widened[0].device, base.device)
            # A real native call consumed this exact Tensor. CPU conversion is
            # an independent byte oracle for the device widening, after launch.
            _bytes_equal(self, widened[0].cpu(), sanitized.cpu().float())
        return outputs

    def test_actual_primary_1024_and_secondary_140_full_wrapper_semantics(self):
        for kind in CASES:
            with self.subTest(kind=kind):
                base, visible = _actual_inputs(kind)
                self._paired(base, visible, padded=kind == "causal_style")
        base, visible = _actual_inputs("wide_fallback", rows=140, width=131072)
        self._paired(base, visible, padded=True)

    def test_complete_ieee_no_refill_short_empty_ties_and_widen_bytes(self):
        base, visible = _adversarial("cuda:0", 16392)
        extra = torch.randn(4, base.shape[1], dtype=base.dtype, device=base.device)
        extra[0, :512] = torch.inf
        extra[1, :513] = torch.inf
        extra[2].fill_(-torch.inf)
        extra[2, :511] = torch.arange(1, 512, device=base.device).to(base.dtype)
        bits = torch.tensor([1, -32767, 127, -32641, 128, -32640, 0, -32768, 32639, -129],
                            dtype=torch.int16, device=base.device).view(torch.bfloat16)
        extra[3] = bits.repeat((base.shape[1] + len(bits) - 1) // len(bits))[:base.shape[1]]
        base = torch.cat((base, extra))
        visible = torch.cat((visible, torch.full((4,), base.shape[1], dtype=visible.dtype, device=base.device)))
        self._paired(base, visible, complete=True, inspect_widen=True)

    def test_odd_width_padded_score_and_offset_output_canaries(self):
        base = torch.randn(7, 65537, dtype=torch.bfloat16, device="cuda:0")
        base[0, :513] = torch.inf
        base[3, 5] = torch.nan
        visible = torch.tensor([65537, 0, 1, 511, 512, 513, 2**40], dtype=torch.int64, device=base.device)
        self._paired(base, visible, padded=True, complete=True)

    def test_balanced_32_samples_include_widen_and_canonicalize(self):
        samples = []
        for kind in CASES:
            base, visible = _actual_inputs(kind)
            expected = self._paired(base, visible)["checked"]
            for arm in ARMS:
                with _variant(arm):
                    for _ in range(2):
                        token.try_select_tokens(base.clone(), visible, disposable=True)
            torch.cuda.synchronize()
            for position, arm in enumerate(("checked", "fp32_hybrid", "fp32_hybrid", "checked") * 4):
                scores = base.clone()
                output = torch.empty((1024, 512), dtype=torch.int32, device=base.device)
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                with _variant(arm) as seen:
                    torch.cuda.synchronize()
                    begin.record()
                    start_ns = time.perf_counter_ns()
                    got = token.try_select_tokens(scores, visible, out=output, disposable=True)
                    end.record()
                    end.synchronize()
                    stop_ns = time.perf_counter_ns()
                self.assertEqual(seen, [False])
                gpu_ms, host_ms = begin.elapsed_time(end), (stop_ns - start_ns) / 1e6
                self.assertTrue(math.isfinite(gpu_ms) and gpu_ms > 0)
                self.assertTrue(math.isfinite(host_ms) and host_ms > 0)
                samples.append(dict(case=kind, position=position, arm=arm, gpu_ms=gpu_ms, host_ms=host_ms))
                assert_equivalent(self, got, expected, base, visible)
                _bytes_equal(self, scores, base.nan_to_num(nan=torch.inf, posinf=torch.inf, neginf=-torch.inf))
        self.assertEqual(len(samples), 32)
        summaries = {}
        for kind in CASES:
            for arm in ARMS:
                selected = [s for s in samples if s["case"] == kind and s["arm"] == arm]
                self.assertEqual(len(selected), 8)
                summaries[f"{kind}/{arm}"] = {metric: dict(
                    median=statistics.median(s[metric] for s in selected),
                    quartiles=statistics.quantiles([s[metric] for s in selected], n=4, method="inclusive"),
                    minimum=min(s[metric] for s in selected), maximum=max(s[metric] for s in selected),
                ) for metric in ("gpu_ms", "host_ms")}
        report = dict(schema_version=1, arms=list(ARMS), shape=[1024, 65536], k=512,
                      bounds=None, starts="zero", force_radix_sort=False, seed=1919,
                      intervals=dict(causal_style="clamp((arange(1024)+126977)//2,0,65536)", wide_fallback="end65536"),
                      included="full BF16 sanitizer, bounds, FP32 widen/allocation, native, canonicalize, original finite finish",
                      excluded="clone/out/events allocation, import, JIT/warmup, numerical checks",
                      host_wall_scope="wrapper submission through end-event wait; not pure CPU compute",
                      provider_sources=_source_hashes(), samples=samples, summaries=summaries,
                      decision="no speed assertion; L0 evaluates exactness and complete cost")
        with _artifact_directory() as directory:
            (directory / "local-prefill-native-fp32-topk.samples.json").write_text(json.dumps(report, indent=2))

    def test_real_hybrid_cast_profiles_and_unpatched_late_register(self):
        original = token._native_ops
        records = {}
        for label, kind, rows, width in (
            ("primary", "causal_style", 1024, 65536),
            ("secondary", "wide_fallback", 140, 131072),
        ):
            base, visible = _actual_inputs(kind, rows=rows, width=width)
            outputs = {}
            for arm in ARMS:
                with _variant(arm):
                    token.try_select_tokens(base.clone(), visible, disposable=True)
            torch.cuda.synchronize()
            for arm in ARMS:
                scores = base.clone()
                torch.cuda.synchronize()
                with _variant(arm):
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                            torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as profile:
                        outputs[arm] = token.try_select_tokens(scores, visible, disposable=True)
                        torch.cuda.synchronize()
                events = _trace_events(profile, f"native-fp32-topk-{label}-{arm}")
                kernels = [e["name"] for e in events if e.get("cat") == "kernel"]
                hybrid = arm == "fp32_hybrid"
                self.assertEqual(sum("topk_prefill_bf16_checked_kernel" in name for name in kernels), int(not hybrid))
                self.assertEqual(sum("topKPerRowPrefill<512,false>" in name.replace(" ", "") for name in kernels), int(hybrid))
                self.assertFalse(any("topKPerRowPrefill<512,true>" in name.replace(" ", "") for name in kernels))
                self.assertTrue(any("_canonicalize_indices" in name for name in kernels))
                self.assertEqual(sum("_prefill_topk_finite_kernel" in name for name in kernels), 1)
                self.assertEqual(sum(e.get("name") == "aten::nan_to_num_" for e in events), 1)
                cast_ops = [e for e in events if e.get("cat") == "cpu_op" and e.get("name") == "aten::_to_copy"
                            and (e.get("args", {}).get("Input Dims") or [None])[0] == [rows, width]]
                self.assertEqual(len(cast_ops), int(hybrid))
                cast_kernels = []
                if hybrid:
                    cast = cast_ops[0]
                    correlations = {e.get("args", {}).get("correlation") for e in events
                                    if e.get("cat") in ("cuda_runtime", "cuda_driver")
                                    and e.get("args", {}).get("correlation") is not None
                                    and cast["ts"] <= e.get("ts", -1) < cast["ts"] + cast.get("dur", 0)}
                    cast_kernels = [e["name"] for e in events if e.get("cat") == "kernel"
                                    and e.get("args", {}).get("correlation") in correlations]
                    self.assertEqual(len(cast_kernels), 1)
                self.assertFalse(any(e.get("name") == "aten::topk" for e in events))
                self.assertFalse(any(e.get("cat") == "gpu_memcpy" and "DtoH" in e.get("name", "") for e in events))
                records[f"{label}/{arm}"] = dict(kernels=kernels, shape=[rows, width], score_stride=list(scores.stride()),
                                                  cast_kernels=cast_kernels, bounds=None, force_radix_sort=False if hybrid else None)
            assert_equivalent(self, outputs["fp32_hybrid"], outputs["checked"], base, visible)
        self.assertIs(token._native_ops, original)
        late = torch.randn(140, 16384, dtype=torch.bfloat16, device="cuda:0")
        visible = torch.full((140,), 16384, dtype=torch.int64, device=late.device)
        token.try_select_tokens(late.clone(), visible, disposable=True)
        scores = late.clone()
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as profile:
            output = token.try_select_tokens(scores, visible, disposable=True)
            torch.cuda.synchronize()
        events = _trace_events(profile, "native-fp32-topk-late-unpatched")
        self.assertTrue(any(e.get("cat") == "kernel" and "topk_prefill_bf16_register_kernel" in e.get("name", "") for e in events))
        self.assertFalse(any(e.get("cat") == "kernel" and "topKPerRowPrefill" in e.get("name", "") for e in events))
        assert_equivalent(self, output, reference(late, visible), late, visible)
        with _artifact_directory() as directory:
            (directory / "local-prefill-native-fp32-topk.profile.json").write_text(json.dumps(dict(
                provider_sources=_source_hashes(), arms=records,
                late=dict(shape=[140, 16384], dispatch="unpatched original register"),
                candidate_topk_2048="not intercepted or modified", native_api="dsv4_top_k_per_row_prefill",
            ), indent=2))


if __name__ == "__main__":
    unittest.main()
