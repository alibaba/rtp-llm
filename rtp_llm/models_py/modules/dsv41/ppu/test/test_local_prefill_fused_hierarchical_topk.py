"""Exact fused metadata helpers around the two unchanged raw BF16 TopKs."""

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
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.kernels.ppu.attention import ppu_bf16_radix_topk as ordering
from rtp_llm.models_py.kernels.ppu.jit import ppu_fp4_indexer as native
from rtp_llm.models_py.modules.dsv41.fp8.test.test_v41_prefill_topk import assert_equivalent, reference
from rtp_llm.models_py.modules.dsv41.ppu import _quant, local_prefill_hierarchical_topk as fused
from rtp_llm.models_py.modules.dsv41.ppu import prefill_token_topk as token
from rtp_llm.models_py.modules.dsv41.ppu.test import test_local_prefill_hierarchical_topk_variants as frozen
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_operator_flags import _trace_events
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_token_topk import (
    _assert_oracle, _BoundsDouble, _bytes_equal, _finish_double, _native_double,
)


ENV = "RTP_V41_PPU_LOCAL_PREFILL_HIERARCHICAL_TOPK"
ARMS = ("checked", "fused")
HELPERS = ("_bf16_max8_keys", "_bf16_gather_block_candidates", "_remap_block_candidate_slots")


@contextmanager
def _scoped(name, scores):
    """Explicit provider experiment; root public rollout is off in BOTH arms."""
    original = token._native_ops
    narrow = (scores.dtype == torch.bfloat16 and scores.ndim == 2
              and 1 <= scores.shape[0] <= 4096
              and 32768 <= scores.shape[1] <= 131072 and scores.shape[1] % 8 == 0)
    ops = ((fused.select, ordering.canonicalize_topk_indices)
           if name == "fused" and narrow else original(False))
    if name not in ARMS:
        raise ValueError(name)
    seen = []

    def factory(prepared):
        if prepared is not False:
            raise AssertionError("experiment requires original bounds=None")
        seen.append(prepared)
        return ops

    with patch.dict(os.environ, {ENV: "0"}), patch.object(token, "_native_ops", new=factory):
        yield seen


@contextmanager
def _public(name):
    if name not in ARMS:
        raise ValueError(name)
    with patch.dict(os.environ, {ENV: "1" if name == "fused" else "0"}):
        yield


@contextmanager
def _artifacts():
    with tempfile.TemporaryDirectory(prefix="fused-hierarchical-topk-") as temporary:
        root = Path(os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR") or temporary)
        root.mkdir(parents=True, exist_ok=True)
        yield root


def _identity():
    return {module.__name__: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            for module in (token, fused, native)}


class _CPUKernel:
    """Launch-only CPU surrogate; the real PPU tests never install it."""
    def __init__(self, kind, calls):
        self.kind, self.calls = kind, calls

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append(self.kind)
            if self.kind == "keys":
                scores, ends, keys, block_ends, width, stride, key_stride, tile = args
                columns = torch.arange(width)[None]
                clamped = ends.clamp(0, width)
                masked = scores.masked_fill(columns >= clamped[:, None], -torch.inf)
                keys.fill_(-torch.inf)
                keys[:, :width // 8].copy_(masked.view(scores.shape[0], width // 8, 8).amax(-1))
                block_ends.copy_((clamped + 7) // 8)
            elif self.kind == "gather":
                scores, ends, blocks, candidates, candidate_ends, width, stride, tile = args
                positions = (blocks.long()[:, :, None] * 8 + torch.arange(8)).view(scores.shape[0], 4096)
                valid = (positions >= 0) & (positions < ends.clamp(0, width)[:, None])
                candidates.copy_(scores.gather(1, positions.clamp(0, width - 1)).masked_fill(~valid, -torch.inf))
                candidate_ends.fill_(4096)
            else:
                chosen, blocks, ends, output, width = args
                safe = chosen.long().clamp(0, 4095)
                block = blocks.gather(1, safe // 8)
                position = block * 8 + safe % 8
                valid = ((chosen >= 0) & (chosen < 4096) & (block >= 0)
                         & (position >= 0) & (position < ends.clamp(0, width)[:, None]))
                output.copy_(torch.where(valid, position, -1).int())
        return launch


class LocalPrefillFusedHierarchicalTopKCPU(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.raw = Mock(wraps=_native_double)
        for item in (
            patch.dict(os.environ, {token.ENV: "1", ENV: "0"}),
            patch.object(token, "_device_supported", return_value=True),
            patch.object(fused, "_device_supported", return_value=True),
            patch.object(native, "topk_bf16", new=self.raw),
            patch.object(ordering, "canonicalize_topk_indices", new=frozen._canonical_cpu),
            patch.object(token._topk, "_prefill_topk_bounds_kernel", _BoundsDouble()),
            patch.object(token._topk, "finish_tokens", side_effect=_finish_double),
            patch.object(fused, HELPERS[0], _CPUKernel("keys", self.calls)),
            patch.object(fused, HELPERS[1], _CPUKernel("gather", self.calls)),
            patch.object(fused, HELPERS[2], _CPUKernel("remap", self.calls)),
        ):
            item.start()
            self.addCleanup(item.stop)
        guard = torch.no_grad()
        guard.__enter__()
        self.addCleanup(guard.__exit__, None, None, None)

    def test_metadata_rejection_has_no_launch_or_mutation(self):
        score = torch.full((2, 65536), torch.nan, dtype=torch.bfloat16)
        ends = torch.tensor([65536, 511], dtype=torch.int32)
        out = torch.full((2, 512), -77, dtype=torch.int32)
        self.assertTrue(fused.is_supported(score, ends, out))
        aliases = score.view(torch.int32).reshape(-1)
        cases = (
            (score[:, :16384], ends, out), (score[:, :32769], ends, out),
            (score.float(), ends, out), (torch._neg_view(score), ends, out),
            (score.to_sparse(), ends, out), (score, ends.long(), out),
            (score, torch._neg_view(ends), out), (score, ends, torch._neg_view(out)),
            (score, aliases[:2], out), (score, ends, aliases[:1024].view(2, 512)),
            (score, ends, out.long()), (score, ends, out[:, :511]),
        )
        before_score, before_out = score.clone(), out.clone()
        for values, lengths, output in cases:
            with self.subTest(shape=values.shape, dtype=values.dtype):
                self.assertFalse(fused.is_supported(values, lengths, output))
                self.assertIsNone(fused.select(values, torch.zeros(2, dtype=torch.int32), lengths, output))
        with torch.enable_grad():
            self.assertFalse(fused.is_supported(score, ends, out))
        with patch.object(fused, "_device_supported", return_value=False):
            self.assertFalse(fused.is_supported(score, ends, out))
        self.assertIsNone(fused.select(score, ends, ends, out))
        self.assertIsNone(fused.select(score, torch._neg_view(ends), ends, out))
        self.assertFalse(self.calls)
        _bytes_equal(self, score, before_score)
        _bytes_equal(self, out, before_out)

    def test_composition_keeps_frozen_partial_nonfinite_and_invalid_contract(self):
        base, visible = frozen._partial_cases("cpu")
        base = torch.cat((base, torch.randn(4, 32768, dtype=torch.bfloat16)))
        visible = torch.cat((visible, torch.full((4,), 32768)))
        for row, count in zip(range(16, 20), (1, 511, 512, 513)):
            base[row, torch.arange(count) * 8] = torch.nan if row % 2 else torch.inf
        with _scoped("fused", base) as seen:
            scores = base.clone()
            got = token.try_select_tokens(scores, visible, disposable=True)
        self.assertEqual(seen, [False])
        self.assertEqual(self.calls, ["keys", "gather", "remap"])
        self.assertEqual(self.raw.call_count, 2)
        self.assertEqual(self.raw.call_args_list[0].args[0].dtype, torch.bfloat16)
        self.assertEqual(self.raw.call_args_list[1].args[0].shape, (20, 4096))
        assert_equivalent(self, got, reference(base, visible), base, visible)
        _assert_oracle(self, got, base, visible)
        _bytes_equal(self, scores, base.nan_to_num(nan=torch.inf, posinf=torch.inf, neginf=-torch.inf))

    def test_real_public_flag_route_admission_change_and_native_errors(self):
        base = torch.randn(2, 32768, dtype=torch.bfloat16)
        visible = torch.tensor([32768, 513])
        with _public("fused"):
            got = token.try_select_tokens(base.clone(), visible, disposable=True)
        self.assertEqual(self.calls, ["keys", "gather", "remap"])
        assert_equivalent(self, got, reference(base, visible), base, visible)
        with _public("fused"), patch.object(fused, "select", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "admission changed"):
                token.try_select_tokens(base.clone(), visible, disposable=True)
        with _public("fused"), patch.object(native, "topk_bf16", side_effect=RuntimeError("raw launch failed")):
            with self.assertRaisesRegex(RuntimeError, "raw launch failed"):
                token.try_select_tokens(base.clone(), visible, disposable=True)
        with _public("fused"), patch.object(fused, "select", side_effect=AssertionError("old fallback")) as select:
            with patch.object(fused, "is_supported", return_value=False):
                self.assertIsNotNone(token.try_select_tokens(base.clone(), visible, disposable=True))
            select.assert_not_called()
        bounds = (torch.zeros(2, dtype=torch.int32), visible.int())
        with _public("fused"), patch.object(fused, "select", side_effect=AssertionError("prepared bounds")) as select:
            with patch.object(token, "_native_ops", return_value=(_native_double, None)) as factory:
                got = token.try_select_tokens(base.clone(), visible, bounds=bounds, disposable=True)
            factory.assert_called_once_with(True)
            select.assert_not_called()
        assert_equivalent(self, got, reference(base, visible), base, visible)


@unittest.skipUnless(os.environ.get("RTP_V41_TEST_LOCAL_WORKSPACE_DEVICE") == "1", "explicit PPU device test")
class LocalPrefillFusedHierarchicalTopKDevice(unittest.TestCase):
    def setUp(self):
        self.assertTrue(_quant.is_ppu())
        self.assertEqual(torch.cuda.get_device_capability(), (8, 9))
        self.assertEqual(torch.cuda.current_device(), 0)
        self.assertEqual(torch.cuda.current_stream(), torch.cuda.default_stream())
        env = patch.dict(os.environ, {token.ENV: "1", ENV: "0"})
        env.start()
        self.addCleanup(env.stop)
        guard = torch.no_grad()
        guard.__enter__()
        self.addCleanup(guard.__exit__, None, None, None)
        torch.manual_seed(2525)

    def _paired(self, base, visible, *, padded=False, complete=False):
        rows, width = base.shape
        results = {}
        sanitized = base.nan_to_num(nan=torch.inf, posinf=torch.inf, neginf=-torch.inf)
        for name in ARMS:
            stride = (width + 128 + 15) // 16 * 16
            bank = torch.full((rows + 2, stride), 7, dtype=base.dtype, device=base.device) if padded else None
            scores = bank[1:-1, :width] if padded else base.clone()
            if padded:
                scores.copy_(base)
            out_bank = torch.full((rows * 512 + 16,), -77, dtype=torch.int32, device=base.device)
            out = out_bank[8:-8].view(rows, 512)
            records = []
            original = native.topk_bf16

            def record_raw(values, starts, ends, output):
                before = values.clone()
                result = original(values, starts, ends, output)
                records.append((values, before, ends.clone()))
                return result

            with _scoped(name, scores) as seen:
                if name == "fused" and frozen._narrow(scores):
                    with patch.object(native, "topk_bf16", new=record_raw):
                        got = token.try_select_tokens(scores, visible, out=out, disposable=True)
                else:
                    got = token.try_select_tokens(scores, visible, out=out, disposable=True)
            self.assertEqual(seen, [False])
            self.assertIs(got, out)
            _bytes_equal(self, scores, sanitized)
            self.assertTrue((out_bank[:8] == -77).all() and (out_bank[-8:] == -77).all())
            if padded:
                self.assertTrue((bank[0] == 7).all() and (bank[-1] == 7).all())
                self.assertTrue((bank[1:-1, width:] == 7).all())
            if records:
                self.assertEqual(len(records), 2)
                keys, before, block_ends = records[0]
                self.assertEqual(keys.dtype, torch.bfloat16)
                self.assertEqual(keys.shape, (rows, width // 8))
                self.assertEqual(keys.stride(0) % 8, 0)
                self.assertEqual(keys.data_ptr() % 16, 0)
                _bytes_equal(self, keys, before)
                columns = torch.arange(width, device=base.device)
                expected = sanitized.masked_fill(columns[None] >= visible.clamp(0, width)[:, None], -torch.inf)
                expected = expected.view(rows, width // 8, 8).amax(-1)
                torch.testing.assert_close(keys, expected, rtol=0, atol=0)
                torch.testing.assert_close(block_ends, ((visible.clamp(0, width).int() + 7) // 8), rtol=0, atol=0)
                if keys.stride(0) != keys.shape[1]:
                    padding = keys.as_strided((rows, keys.stride(0) - keys.shape[1]), (keys.stride(0), 1),
                                              storage_offset=keys.storage_offset() + keys.shape[1])
                    self.assertTrue(padding.isneginf().all())
                candidates, before, candidate_ends = records[1]
                self.assertEqual(candidates.shape, (rows, 4096))
                self.assertEqual(candidates.dtype, torch.bfloat16)
                _bytes_equal(self, candidates, before)
                self.assertTrue((candidate_ends == 4096).all())
            results[name] = got.clone()
        for name, other in (("checked", "fused"), ("fused", "checked")):
            assert_equivalent(self, results[name], results[other], base, visible)
            _assert_oracle(self, results[name], base, visible,
                           rows=None if complete else sorted({0, rows // 2, rows - 1}))
        if complete:
            expected = reference(base, visible)
            for result in results.values():
                assert_equivalent(self, result, expected, base, visible)
        return results

    # Reuse qualified fixture constructors and their exact assertions, never
    # the older Torch hierarchy/variant factory or its benchmark/profile.
    def test_actual_full_compact_and_padded_exact(self):
        frozen.LocalPrefillHierarchicalTopKVariantsDevice.test_actual_full_and_compact_shapes_preserve_exact_score_contract(self)

    def test_all_bf16_bits_threshold_overflow_and_nonfinite_slots(self):
        frozen.LocalPrefillHierarchicalTopKVariantsDevice.test_all_bf16_bits_and_coarse_overflow_511_512_513_exact(self)

    def test_partial_all_remainders_empty_short_and_no_newest_pin(self):
        frozen.LocalPrefillHierarchicalTopKVariantsDevice.test_partial_poison_all_remainders_no_pin_short_empty_and_real_negative_inf(self)

    def test_unaligned_logical_keys_guards_odd_and_late_old_routes(self):
        frozen.LocalPrefillHierarchicalTopKVariantsDevice.test_unaligned_logical_keys_padded_views_out_guards_and_old_routes(self)

    def test_balanced_real_public_flags_full_wrapper_latency(self):
        samples = []
        for kind in ("causal_style", "wide_overflow"):
            base, visible = frozen._primary(kind)
            expected = self._paired(base, visible)["checked"]
            for name in ARMS:
                with _public(name):
                    for _ in range(2):
                        token.try_select_tokens(base.clone(), visible, disposable=True)
            torch.cuda.synchronize()
            for position, name in enumerate(("checked", "fused", "fused", "checked") * 4):
                scores = base.clone()
                output = torch.empty((1024, 512), dtype=torch.int32, device=base.device)
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                with _public(name):
                    torch.cuda.synchronize()
                    start.record()
                    before = time.perf_counter_ns()
                    got = token.try_select_tokens(scores, visible, out=output, disposable=True)
                    end.record()
                    end.synchronize()
                    after = time.perf_counter_ns()
                gpu_ms, host_ms = start.elapsed_time(end), (after - before) / 1e6
                self.assertTrue(math.isfinite(gpu_ms) and gpu_ms > 0)
                self.assertTrue(math.isfinite(host_ms) and host_ms > 0)
                samples.append(dict(case=kind, position=position, arm=name, public_flag=int(name == "fused"),
                                    gpu_ms=gpu_ms, host_ms=host_ms))
                assert_equivalent(self, got, expected, base, visible)
                _bytes_equal(self, scores, base.nan_to_num(nan=torch.inf, posinf=torch.inf, neginf=-torch.inf))
        summaries = {}
        for kind in ("causal_style", "wide_overflow"):
            for name in ARMS:
                chosen = [s for s in samples if s["case"] == kind and s["arm"] == name]
                self.assertEqual(len(chosen), 8)
                summaries[f"{kind}/{name}"] = {metric: dict(
                    median=statistics.median(s[metric] for s in chosen),
                    quartiles=statistics.quantiles([s[metric] for s in chosen], n=4, method="inclusive"),
                    minimum=min(s[metric] for s in chosen), maximum=max(s[metric] for s in chosen),
                ) for metric in ("gpu_ms", "host_ms")}
        with _artifacts() as root:
            (root / "local-prefill-fused-hierarchical-topk.samples.json").write_text(json.dumps(dict(
                provider_sources=_identity(), shape=[1024, 65536], bounds=None, seed=2525,
                public_flags={"checked": 0, "fused": 1}, factory_patch=False, stream="same default",
                host_scope="public wrapper submission through end-event wait; clone/JIT/checks excluded",
                samples=samples, summaries=summaries, decision="no timing threshold; root decides",
            ), indent=2))

    def _assert_profile(self, events, fused_route):
        kernels = [e["name"] for e in events if e.get("cat") == "kernel"]
        for helper in HELPERS:
            self.assertEqual(sum(helper in name for name in kernels), int(fused_route))
        self.assertEqual(sum("topk_prefill_bf16_register_kernel" in name for name in kernels), 2 if fused_route else 0)
        self.assertEqual(sum("topk_prefill_bf16_checked_kernel" in name for name in kernels), 0 if fused_route else 1)
        self.assertEqual(sum("_canonicalize_indices" in name for name in kernels), 3 if fused_route else 1)
        self.assertEqual(sum("_prefill_topk_finite_kernel" in name for name in kernels), 1)
        self.assertEqual(sum(e.get("name") == "aten::nan_to_num_" for e in events), 1)
        self.assertFalse(any(e.get("name") == "aten::topk" for e in events))
        self.assertFalse(any(e.get("cat") == "gpu_memcpy" and "DtoH" in e.get("name", "") for e in events))
        if fused_route:
            self.assertFalse(any("_selective_" in name or "_fallback_" in name or "_prepare_rows" in name for name in kernels))
            self.assertFalse(any(e.get("cat") == "cpu_op" and e.get("name") in (
                "aten::amax", "aten::gather", "aten::scatter_", "aten::masked_fill_", "aten::where",
            ) for e in events))
        return kernels

    def test_profile_three_helpers_public_flag_and_old_fallbacks(self):
        base, visible = frozen._primary("causal_style")
        original = token._native_ops
        contexts = (
            ("checked", lambda scores: _scoped("checked", scores), False),
            ("scoped-fused", lambda scores: _scoped("fused", scores), True),
            ("public-fused", lambda scores: _public("fused"), True),
        )
        for _, context, _ in contexts:
            with context(base):
                token.try_select_tokens(base.clone(), visible, disposable=True)
        torch.cuda.synchronize()
        records, outputs = {}, {}
        for label, context, route in contexts:
            scores = base.clone()
            torch.cuda.synchronize()
            with context(scores):
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                        torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as profile:
                    outputs[label] = token.try_select_tokens(scores, visible, disposable=True)
                    torch.cuda.synchronize()
            events = _trace_events(profile, f"fused-hierarchical-topk-{label}")
            records[label] = dict(shape=list(scores.shape), bounds=None, kernels=self._assert_profile(events, route))
        self.assertIs(token._native_ops, original)
        for got in outputs.values():
            assert_equivalent(self, got, outputs["checked"], base, visible)
        for width in (16384, 32769):
            rows = 140 if width == 16384 else 1
            bank = torch.randn(rows + 2, (width + 15) // 16 * 16, dtype=torch.bfloat16, device=base.device)
            scores = bank[1:-1, :width]
            source = scores.clone()
            lengths = torch.full((rows,), width, dtype=torch.int64, device=base.device)
            with _public("fused"), patch.object(fused, "select", side_effect=AssertionError("old route")) as select:
                if width == 16384:
                    token.try_select_tokens(source.clone(), lengths, disposable=True)
                    torch.cuda.synchronize()
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                            torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as profile:
                        got = token.try_select_tokens(scores, lengths, disposable=True)
                        torch.cuda.synchronize()
                    events = _trace_events(profile, "fused-hierarchical-topk-late-old-route")
                    late_kernels = [e["name"] for e in events if e.get("cat") == "kernel"]
                    self.assertEqual(sum("topk_prefill_bf16_register_kernel" in name for name in late_kernels), 1)
                    self.assertFalse(any(helper in name for helper in HELPERS for name in late_kernels))
                    records["public-late-old-route"] = dict(shape=[140, 16384], kernels=late_kernels)
                else:
                    got = token.try_select_tokens(scores, lengths, disposable=True)
            select.assert_not_called()
            assert_equivalent(self, got, reference(source, lengths), source, lengths)
        scores = base[:2].clone()
        source = scores.clone()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), _public("fused"), patch.object(fused, "select", side_effect=AssertionError("side stream")) as select:
            got = token.try_select_tokens(scores, visible[:2], disposable=True)
        torch.cuda.current_stream().wait_stream(side)
        select.assert_not_called()
        assert_equivalent(self, got, reference(source, visible[:2]), source, visible[:2])
        with _artifacts() as root:
            (root / "local-prefill-fused-hierarchical-topk.profile.json").write_text(json.dumps(dict(
                provider_sources=_identity(), arms=records, public_fallbacks=["late", "padded odd", "nondefault stream"],
            ), indent=2))


if __name__ == "__main__":
    unittest.main()
