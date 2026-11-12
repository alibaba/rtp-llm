"""Test-only, same-interval checked versus existing all-active radix K512.

Both arms call the unmodified public wrapper with bounds=None. No serving
dispatch, candidate publication, kernel, precision or acceptance rule changes.
"""

from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
import time
import unittest
from unittest.mock import Mock, call, patch

import torch

from rtp_llm.models_py.modules.dsv41.fp8.attention_v41 import select_candidate_blocks
from rtp_llm.models_py.modules.dsv41.fp8.test.test_v41_prefill_topk import assert_equivalent, reference
from rtp_llm.models_py.modules.dsv41.ppu import _quant, prefill_token_topk as token
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_operator_flags import _trace_events
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_token_topk import (
    _adversarial, _assert_oracle, _BoundsDouble, _bytes_equal, _finish_double, _native_double,
)


ARMS = ("checked", "radix")


@contextmanager
def _variant(arm, *, ops=None):
    """Resolve before patch/timing; change only the private provider result."""
    if arm not in ARMS:
        raise ValueError(arm)
    original = token._native_ops
    resolved = original(arm == "radix") if ops is None else ops
    seen = []

    def load(prepared):
        if prepared is not False:
            raise AssertionError("both experimental arms require bounds=None")
        seen.append(prepared)
        return resolved

    with patch.object(token, "_native_ops", new=load):
        yield seen


@contextmanager
def _artifact_directory():
    with tempfile.TemporaryDirectory(prefix="local-token-topk-variants-") as temporary:
        directory = Path(os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR") or temporary)
        directory.mkdir(parents=True, exist_ok=True)
        yield directory


def _source_hashes():
    from rtp_llm.models_py.kernels.ppu.attention import ppu_bf16_radix_topk
    from rtp_llm.models_py.kernels.ppu.jit import ppu_fp4_indexer

    return {module.__name__: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            for module in (token, ppu_bf16_radix_topk, ppu_fp4_indexer)}


def _actual_inputs(kind, *, rows=1024, width=65536):
    base = torch.randn(rows, width, device="cuda:0", dtype=torch.bfloat16)
    if kind == "causal_style":
        # First actual full Prefill tile: prefix126976, ratio2, fresh4096.
        visible = (torch.arange(rows, device=base.device, dtype=torch.int64) + 126977) // 2
        visible = visible.clamp(0, width)
        columns = torch.arange(width, device=base.device)
        base.masked_fill_(columns[None] >= visible[:, None], -torch.inf)
        base[0, :1] = torch.nan
        base[rows // 2, :511] = torch.inf
        base[-1, :513] = torch.nan
    elif kind == "wide_fallback":
        visible = torch.full((rows,), width, dtype=torch.int64, device=base.device)
        # Fewer than K values in the maximum-key 256-bin window. The
        # existing exact two-byte fallback must handle this broad range.
        base[:, :17] = 2.0**100
        base[:, 17:33] = -(2.0**100)
    else:
        raise ValueError(kind)
    return base, visible


class LocalPrefillTokenTopKVariantsCPU(unittest.TestCase):
    def setUp(self):
        patches = (
            patch.dict(os.environ, {token.ENV: "1"}),
            patch.object(token, "_device_supported", return_value=True),
            patch.object(token._topk, "_prefill_topk_bounds_kernel", _BoundsDouble()),
            patch.object(token._topk, "finish_tokens", side_effect=_finish_double),
        )
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        guard = torch.no_grad()
        guard.__enter__()
        self.addCleanup(guard.__exit__, None, None, None)

    def test_scoped_arms_keep_none_bounds_sanitizer_finish_and_restore(self):
        base, visible = _adversarial("cpu", 1024)
        original_loader = token._native_ops
        outputs = []
        for arm in ARMS:
            scores = base.clone()
            canonicalize = Mock(side_effect=lambda out: out) if arm == "radix" else None

            def select(logits, starts, ends, out):
                self.assertTrue(torch.equal(starts, torch.zeros_like(ends)))
                self.assertTrue(torch.equal(ends, visible.clamp(0, 1024).int()))
                _bytes_equal(self, logits, base.nan_to_num(nan=torch.inf, posinf=torch.inf, neginf=-torch.inf))
                return _native_double(logits, starts, ends, out)

            with _variant(arm, ops=(select, canonicalize)) as seen:
                got = token.try_select_tokens(scores, visible, disposable=True)
            self.assertEqual(seen, [False])
            self.assertIs(token._native_ops, original_loader)
            assert_equivalent(self, got, reference(base, visible), base, visible)
            _assert_oracle(self, got, base, visible)
            if canonicalize is not None:
                canonicalize.assert_called_once()
            outputs.append(got)
        assert_equivalent(self, outputs[1], outputs[0], base, visible)
        self.assertEqual(token._topk.finish_tokens.call_count, 2)

    def test_import_and_native_errors_keep_original_fallback_and_scope(self):
        base = torch.full((2, 1024), torch.nan, dtype=torch.bfloat16)
        visible = torch.full((2,), 1024, dtype=torch.int64)
        original_loader = token._native_ops
        with patch.object(token, "_native_ops", side_effect=ImportError("provider unavailable")):
            scores = base.clone()
            self.assertIsNone(token.try_select_tokens(scores, visible, disposable=True))
            _bytes_equal(self, scores, base)
        self.assertIs(token._native_ops, original_loader)
        for arm in ARMS:
            failed = Mock(side_effect=RuntimeError("qualified launch failed"))
            with _variant(arm, ops=(failed, None)):
                scores = base.clone()
                with self.assertRaisesRegex(RuntimeError, "qualified launch failed"):
                    token.try_select_tokens(scores, visible, disposable=True)
            failed.assert_called_once()
            self.assertTrue(scores.isposinf().all())
            self.assertIs(token._native_ops, original_loader)
        with patch.object(token, "_native_ops", side_effect=AssertionError("unsupported work")):
            self.assertIsNone(token.try_select_tokens(base, visible, disposable=False))
        _bytes_equal(self, base, torch.full_like(base, torch.nan))

    def test_patch_restores_default_dispatch_and_preserves_candidate_publication(self):
        base = torch.randn(2, 16400, dtype=torch.bfloat16)
        base[0, 0] = torch.nan
        visible = torch.full((2,), 16400, dtype=torch.int64)
        candidates = select_candidate_blocks(base, visible, 8, 2048)
        published = candidates.clone()
        default = Mock(return_value=(_native_double, None))
        with patch.object(token, "_native_ops", new=default):
            token.try_select_tokens(base.clone(), visible, disposable=True)
            with _variant("radix", ops=(_native_double, lambda out: out)):
                token.try_select_tokens(base.clone(), visible, disposable=True)
            token.try_select_tokens(base.clone(), visible, disposable=True)
            self.assertIs(token._native_ops, default)
        self.assertEqual(default.call_args_list, [call(False), call(False)])
        self.assertEqual(candidates.shape, (2, 2048))
        _bytes_equal(self, candidates, published)
        self.assertEqual(token.K, 512)


@unittest.skipUnless(os.environ.get("RTP_V41_TEST_LOCAL_WORKSPACE_DEVICE") == "1", "explicit PPU device test")
class LocalPrefillTokenTopKVariantsDevice(unittest.TestCase):
    def setUp(self):
        self.assertTrue(_quant.is_ppu())
        self.assertEqual(torch.cuda.get_device_capability(), (8, 9))
        self.assertEqual(torch.cuda.current_device(), 0)
        self.assertEqual(torch.cuda.current_stream(), torch.cuda.default_stream())
        env = patch.dict(os.environ, {token.ENV: "1"})
        env.start()
        self.addCleanup(env.stop)
        guard = torch.no_grad()
        guard.__enter__()
        self.addCleanup(guard.__exit__, None, None, None)
        torch.manual_seed(1515)

    def _paired(self, base, visible, *, padded=False, complete=False):
        outputs = {}
        sanitized = base.nan_to_num(nan=torch.inf, posinf=torch.inf, neginf=-torch.inf)
        rows, width = base.shape
        for arm in ARMS:
            stride = ((width + 128 + 15) // 16) * 16
            allocation = torch.full((rows + 2, stride), 7, dtype=base.dtype, device=base.device) if padded else None
            scores = allocation[1:rows + 1, :width] if padded else base.clone()
            if padded:
                scores.copy_(base)
            out_bank = torch.full((rows * 512 + 16,), -77, dtype=torch.int32, device=base.device)
            out = out_bank[8:-8].view(rows, 512)
            with _variant(arm) as seen:
                got = token.try_select_tokens(scores, visible, out=out, disposable=True)
            self.assertEqual(seen, [False])
            self.assertIs(got, out)
            _bytes_equal(self, scores, sanitized)
            self.assertTrue((out_bank[:8] == -77).all() and (out_bank[-8:] == -77).all())
            if padded:
                self.assertTrue((allocation[0] == 7).all() and (allocation[-1] == 7).all())
                self.assertTrue((allocation[1:-1, width:] == 7).all())
            outputs[arm] = got.clone()
        # Every actual-shape row is checked for exact score multiset,
        # valid count/uniqueness/range; full-width CPU sorting is sampled.
        assert_equivalent(self, outputs["radix"], outputs["checked"], base, visible)
        assert_equivalent(self, outputs["checked"], outputs["radix"], base, visible)
        selected_rows = None if complete else sorted({0, rows // 2, rows - 1})
        for got in outputs.values():
            _assert_oracle(self, got, base, visible, rows=selected_rows)
        if complete:
            expected = reference(base, visible)
            for got in outputs.values():
                assert_equivalent(self, got, expected, base, visible)
        return outputs

    def test_actual_1024_causal_and_wide_fallback_full_wrapper_semantics(self):
        for kind in ("causal_style", "wide_fallback"):
            with self.subTest(kind=kind):
                base, visible = _actual_inputs(kind)
                self._paired(base, visible, padded=kind == "causal_style")

    def test_complete_nonfinite_no_refill_empty_short_ties_and_bf16_specials(self):
        base, visible = _adversarial("cuda:0", 16392)
        bits = torch.tensor([1, -32767, 127, -32641, 128, -32640, 0, -32768, 32639, -129],
                            dtype=torch.int16, device=base.device).view(torch.bfloat16)
        base[16] = bits.repeat((base.shape[1] + len(bits) - 1) // len(bits))[:base.shape[1]]
        self._paired(base, visible, complete=True)

    def test_odd_width_padded_chunk_and_offset_output_keep_raw_canaries(self):
        base = torch.randn(7, 65537, dtype=torch.bfloat16, device="cuda:0")
        base[0, :513] = torch.inf
        base[3, 5] = torch.nan
        visible = torch.tensor([65537, 0, 1, 511, 512, 513, 2**40], dtype=torch.int64, device=base.device)
        self._paired(base, visible, padded=True, complete=True)

    def test_actual_same_interval_balanced_full_wrapper_latency_samples(self):
        samples = []
        for kind in ("causal_style", "wide_fallback"):
            base, visible = _actual_inputs(kind)
            expected = self._paired(base, visible)["checked"]
            for arm in ARMS:
                with _variant(arm):
                    for _ in range(2):
                        token.try_select_tokens(base.clone(), visible, disposable=True)
            torch.cuda.synchronize()
            # ABBA repeated balances position and predecessor for both arms.
            order = ("checked", "radix", "radix", "checked") * 4
            for position, arm in enumerate(order):
                scores = base.clone()
                output = torch.empty((1024, 512), dtype=torch.int32, device=base.device)
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                with _variant(arm) as seen:
                    # Clone/allocation/import/JIT and their GPU work finish
                    # before either timer. Numerical observations follow it.
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
        summaries = {}
        for kind in ("causal_style", "wide_fallback"):
            for arm in ARMS:
                selected = [s for s in samples if s["case"] == kind and s["arm"] == arm]
                self.assertEqual(len(selected), 8)
                summaries[f"{kind}/{arm}"] = {metric: dict(
                    median=statistics.median(s[metric] for s in selected),
                    quartiles=statistics.quantiles([s[metric] for s in selected], n=4, method="inclusive"),
                    minimum=min(s[metric] for s in selected), maximum=max(s[metric] for s in selected),
                ) for metric in ("gpu_ms", "host_ms")}
        report = dict(schema_version=1, purpose="test-only same-interval provider comparison",
                      shape=[1024, 65536], score_stride=[65536, 1], k=512, bounds=None,
                      starts="zero", stream="same default", seed=1515,
                      intervals=dict(causal_style="clamp((arange(1024)+126977)//2,0,65536)", wide_fallback="end65536"),
                      host_wall_scope="wrapper submission through recorded end-event wait; clone/JIT/checks excluded",
                      provider_sources=_source_hashes(), samples=samples, summaries=summaries,
                      decision="no timing threshold; L0 evaluates component and model evidence")
        with _artifact_directory() as directory:
            (directory / "local-prefill-token-topk-variants.samples.json").write_text(json.dumps(report, indent=2))

    def test_actual_same_input_profiles_and_unmodified_late_register(self):
        base, visible = _actual_inputs("causal_style")
        original_loader = token._native_ops
        # Both compilers/providers warmed outside profile; no changed starts.
        for arm in ARMS:
            with _variant(arm):
                token.try_select_tokens(base.clone(), visible, disposable=True)
        torch.cuda.synchronize()
        outputs = {}
        records = {}
        for arm in ARMS:
            scores = base.clone()
            torch.cuda.synchronize()
            with _variant(arm):
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                        torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as profile:
                    outputs[arm] = token.try_select_tokens(scores, visible, disposable=True)
                    torch.cuda.synchronize()
            events = _trace_events(profile, f"token-topk-variants-{arm}")
            kernels = [e["name"] for e in events if e.get("cat") == "kernel"]
            self.assertEqual(sum("topk_prefill_bf16_checked_kernel" in name for name in kernels), int(arm == "checked"))
            self.assertTrue(any("_selective_scatter" in name for name in kernels))
            self.assertTrue(any("_canonicalize_indices" in name for name in kernels))
            self.assertTrue(any("_prefill_topk_finite_kernel" in name for name in kernels))
            self.assertEqual(sum(e.get("name") == "aten::nan_to_num_" for e in events), 1)
            self.assertFalse(any(e.get("name") == "aten::topk" for e in events))
            self.assertFalse(any(e.get("cat") == "gpu_memcpy" and "DtoH" in e.get("name", "") for e in events))
            records[arm] = dict(kernels=kernels, shape=list(scores.shape), bounds=None,
                                score_stride=list(scores.stride()), rows_to_compute=None if arm == "radix" else "native fallback vector")
        self.assertIs(token._native_ops, original_loader)
        assert_equivalent(self, outputs["radix"], outputs["checked"], base, visible)
        late = torch.randn(140, 16384, dtype=torch.bfloat16, device=base.device)
        late_visible = torch.full((140,), 16384, dtype=torch.int64, device=base.device)
        token.try_select_tokens(late.clone(), late_visible, disposable=True)
        late_scores = late.clone()
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as profile:
            late_output = token.try_select_tokens(late_scores, late_visible, disposable=True)
            torch.cuda.synchronize()
        late_events = _trace_events(profile, "token-topk-variants-late-unpatched")
        self.assertTrue(any(e.get("cat") == "kernel" and "topk_prefill_bf16_register_kernel" in e.get("name", "")
                            for e in late_events))
        assert_equivalent(self, late_output, reference(late, late_visible), late, late_visible)
        with _artifact_directory() as directory:
            (directory / "local-prefill-token-topk-variants.profile.json").write_text(json.dumps(dict(
                provider_sources=_source_hashes(), arms=records, late=dict(shape=[140, 16384], dispatch="unpatched register"),
                candidate_topk_2048="not intercepted or modified by this experiment",
            ), indent=2))


if __name__ == "__main__":
    unittest.main()
