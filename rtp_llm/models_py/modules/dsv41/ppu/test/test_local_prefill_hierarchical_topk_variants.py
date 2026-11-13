"""Test-only fresh BF16 hierarchy versus the unchanged checked K512 wrapper.

No production flag/dispatch, published candidates, existing test, or kernel is
modified. Both arms retain full sanitization and the original finite finish.
"""

from contextlib import contextmanager
from functools import partial
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

from rtp_llm.models_py.kernels.ppu.attention.ppu_bf16_radix_topk import canonicalize_topk_indices
from rtp_llm.models_py.kernels.ppu.jit.ppu_fp4_indexer import topk_bf16
from rtp_llm.models_py.modules.dsv41.fp8.test.test_v41_prefill_topk import assert_equivalent, reference
from rtp_llm.models_py.modules.dsv41.ppu import _quant, prefill_token_topk as token
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_operator_flags import _trace_events
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_token_topk import (
    _assert_oracle, _BoundsDouble, _bytes_equal, _finish_double, _native_double,
)


ARMS = ("checked", "hierarchical")


def _narrow(scores):
    return (scores.ndim == 2 and scores.dtype == torch.bfloat16
            and 32768 <= scores.shape[1] <= 131072 and scores.shape[1] % 8 == 0)


def _hierarchical_raw(scores, starts, ends, output, *, raw, observe=None):
    """Select only raw slots: no finite filter at either intermediate stage."""
    rows, width = scores.shape
    blocks = width // 8
    block_view = scores.view(rows, blocks, 8)  # Split columns; retain source stride/storage.
    keys = block_view.amax(-1)  # BF16 values, no FP16/FP32 block-key conversion.
    # Complete invisible blocks are excluded by block_ends. Only this one
    # partial key needs repair, even when future score values are poisoned.
    offsets = torch.arange(8, dtype=torch.int64, device=scores.device)
    last = (ends.long() - 1).clamp_min(0) // 8
    tail_ids = last[:, None] * 8 + offsets[None]
    tail = scores.gather(1, tail_ids)
    tail.masked_fill_(tail_ids >= ends[:, None], -torch.inf)
    keys.scatter_(1, last[:, None], tail.amax(-1, keepdim=True))
    key_stride = (blocks + 7) // 8 * 8
    bank = None
    if key_stride != blocks:
        # Poison physical padding: the raw selector must honor logical ends.
        bank = torch.full((rows, key_stride), torch.inf, dtype=torch.bfloat16, device=scores.device)
        padded = bank[:, :blocks]
        padded.copy_(keys)
        keys = padded
    keys_before = keys.clone() if observe is not None else None
    block_ends = (ends + 7) // 8
    chosen_blocks = torch.empty((rows, 512), dtype=torch.int32, device=scores.device)
    raw(keys, starts, block_ends, chosen_blocks)
    ids = (chosen_blocks.long()[:, :, None] * 8 + offsets).view(rows, 4096)
    valid = ((chosen_blocks >= 0)[:, :, None].expand(rows, 512, 8).reshape(rows, 4096)
             & (ids >= 0) & (ids < ends[:, None]) & (ids < width))
    candidates = scores.gather(1, ids.clamp(0, width - 1))
    candidates.masked_fill_(~valid, -torch.inf)
    chosen = torch.empty_like(output)
    raw(candidates, starts, torch.full_like(ends, 4096), chosen)
    safe = chosen.long().clamp(0, 4095)
    logical = ids.gather(1, safe)
    keep = ((chosen >= 0) & (chosen < 4096) & valid.gather(1, safe)
            & (logical >= 0) & (logical < ends[:, None]))
    output.copy_(torch.where(keep, logical, -1).int())
    if observe is not None:
        observe.append(dict(
            source_view_alias=block_view.untyped_storage().data_ptr() == scores.untyped_storage().data_ptr(),
            source_view_stride=list(block_view.stride()), key_dtype=str(keys.dtype),
            key_shape=list(keys.shape), key_stride=list(keys.stride()),
            key_alignment=keys.data_ptr() % 16, candidate_dtype=str(candidates.dtype),
            candidate_shape=list(candidates.shape), candidate_stride=list(candidates.stride()),
            keys=keys, keys_before=keys_before,
            key_padding=None if bank is None else bank[:, blocks:],
        ))
    return output


@contextmanager
def _arm(name, scores, *, raw=None, canonicalize=None, observe=None):
    """Own scoped factory: do not import another experiment's variant mock."""
    if name not in ARMS:
        raise ValueError(name)
    original = token._native_ops
    use_hierarchy = name == "hierarchical" and _narrow(scores)
    if use_hierarchy:
        select = partial(_hierarchical_raw, raw=topk_bf16 if raw is None else raw, observe=observe)
        ops = select, canonicalize_topk_indices if canonicalize is None else canonicalize
    else:
        ops = original(False)
    seen = []

    def factory(prepared):
        if prepared is not False:
            raise AssertionError("both arms require bounds=None")
        seen.append(prepared)
        return ops

    with patch.object(token, "_hierarchical_ops", return_value=None), patch.object(token, "_native_ops", new=factory):
        yield seen


def _canonical_cpu(output):
    ordered = torch.where(output < 0, 2**31 - 1, output).sort(-1).values
    output.copy_(torch.where(ordered == 2**31 - 1, -1, ordered))
    return output


@contextmanager
def _artifacts():
    with tempfile.TemporaryDirectory(prefix="hierarchical-bf16-topk-") as temporary:
        root = Path(os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR") or temporary)
        root.mkdir(parents=True, exist_ok=True)
        yield root


def _identity():
    import rtp_llm.models_py.kernels.ppu.jit.ppu_fp4_indexer as native

    return {module.__name__: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            for module in (token, native)}


def _primary(kind, *, compact=False):
    rows, width, ratio, prefix = (140, 131072, 1, 130932) if compact else (1024, 65536, 2, 126976)
    values = torch.randn(rows, width, dtype=torch.bfloat16, device="cuda:0")
    if kind == "causal_style":
        ends = (torch.arange(rows, dtype=torch.int64, device=values.device) + prefix + 1) // ratio
        values.masked_fill_(torch.arange(width, device=values.device)[None] >= ends[:, None], -torch.inf)
        values[0, :1], values[rows // 2, :511], values[-1, :513] = torch.nan, torch.inf, torch.nan
    elif kind == "wide_overflow":
        ends = torch.full((rows,), width, dtype=torch.int64, device=values.device)
        values[:, :17], values[:, 17:33] = 2.0**100, -(2.0**100)
    else:
        raise ValueError(kind)
    return values, ends


def _partial_cases(device):
    ends = torch.tensor([0, 1, 7, 8, 9, 511, 512, 513] + list(range(4096, 4104)),
                        dtype=torch.int64, device=device)
    values = torch.full((len(ends), 32768), -3, dtype=torch.bfloat16, device=device)
    values[:, torch.arange(512, device=device) * 8] = torch.arange(1, 513, device=device).bfloat16()
    # Invisible +Inf/NaN in a partial newest block must not steal a key.
    for row, end in enumerate([0, 1, 7, 8, 9, 511, 512, 513] + list(range(4096, 4104))):
        if end % 8:
            values[row, end:(end + 7) // 8 * 8] = torch.inf
            values[row, end] = torch.nan
    return values, ends


class LocalPrefillHierarchicalTopKVariantsCPU(unittest.TestCase):
    def setUp(self):
        self.default = Mock(return_value=(_native_double, None))
        for item in (
            patch.dict(os.environ, {token.ENV: "1"}),
            patch.object(token, "_native_ops", new=self.default),
            patch.object(token, "_device_supported", return_value=True),
            patch.object(token._topk, "_prefill_topk_bounds_kernel", _BoundsDouble()),
            patch.object(token._topk, "finish_tokens", side_effect=_finish_double),
        ):
            item.start()
            self.addCleanup(item.stop)
        guard = torch.no_grad()
        guard.__enter__()
        self.addCleanup(guard.__exit__, None, None, None)

    def _check(self, base, visible):
        before = token._native_ops
        scores = base.clone()
        raw = Mock(wraps=_native_double)
        observed = []
        with _arm("hierarchical", scores, raw=raw, canonicalize=_canonical_cpu, observe=observed) as seen:
            result = token.try_select_tokens(scores, visible, disposable=True)
        self.assertIs(token._native_ops, before)
        self.assertEqual(seen, [False])
        self.assertEqual(raw.call_count, 2)
        self.assertEqual(raw.call_args_list[0].args[0].dtype, torch.bfloat16)
        self.assertEqual(raw.call_args_list[1].args[0].shape[1], 4096)
        _bytes_equal(self, scores, base.nan_to_num(nan=torch.inf, posinf=torch.inf, neginf=-torch.inf))
        assert_equivalent(self, result, reference(base, visible), base, visible)
        _assert_oracle(self, result, base, visible)
        self.assertTrue(observed[0]["source_view_alias"])
        _bytes_equal(self, observed[0]["keys"], observed[0]["keys_before"])
        if observed[0]["key_padding"] is not None:
            self.assertTrue(observed[0]["key_padding"].isposinf().all())
        return result, raw, observed

    def test_raw_two_stage_composition_keeps_nonfinite_slots_without_refill(self):
        base = torch.randn(4, 32768, dtype=torch.bfloat16)
        visible = torch.tensor([32768] * 4)
        for row, count in enumerate((1, 511, 512, 513)):
            base[row, torch.arange(count) * 8] = torch.nan if row % 2 else torch.inf
        result, _, _ = self._check(base, visible)
        self.assertTrue(torch.equal((result >= 0).sum(1), torch.tensor([511, 1, 0, 0])))

    def test_partial_poison_empty_invalid_padding_and_no_newest_pin(self):
        self._check(*_partial_cases("cpu"))
        base = torch.full((2, 32776), -torch.inf, dtype=torch.bfloat16)
        base[0, 0] = 7
        result, _, observed = self._check(base, torch.tensor([0, 1]))
        self.assertTrue((result[0] == -1).all())
        self.assertEqual(int((result[1] >= 0).sum()), 0)
        self.assertEqual(observed[0]["key_shape"][1], 4097)
        self.assertEqual(observed[0]["key_stride"][0] % 8, 0)
        self.assertEqual(observed[0]["key_alignment"], 0)

    def test_odd_late_old_route_and_admitted_error_restore_the_factory(self):
        raw = Mock(side_effect=AssertionError("hierarchy must not run"))
        for width in (16384, 32769):
            stride = (width + 128 + 15) // 16 * 16
            bank = torch.full((3, stride), 7, dtype=torch.bfloat16)
            scores = bank[1:2, :width]
            scores.normal_()
            scores[0, 0] = torch.nan
            original = scores.clone()
            out_bank = torch.full((512 + 16,), -77, dtype=torch.int32)
            out = out_bank[8:-8].view(1, 512)
            visible = torch.tensor([width])
            self.assertEqual(scores.stride(0) % 8, 0)
            with _arm("hierarchical", scores, raw=raw, canonicalize=_canonical_cpu) as seen:
                got = token.try_select_tokens(scores, visible, out=out, disposable=True)
            self.assertIsNotNone(got)
            self.assertIs(got, out)
            self.assertEqual(seen, [False])
            assert_equivalent(self, got, reference(original, visible), original, visible)
            _bytes_equal(self, scores, original.nan_to_num(nan=torch.inf, posinf=torch.inf, neginf=-torch.inf))
            self.assertTrue((bank[0] == 7).all() and (bank[-1] == 7).all())
            self.assertTrue((bank[1:2, width:] == 7).all())
            self.assertTrue((out_bank[:8] == -77).all() and (out_bank[-8:] == -77).all())
        raw.assert_not_called()
        self.assertIs(token._native_ops, self.default)
        tight = torch.full((2, 32769), torch.nan, dtype=torch.bfloat16)
        tight_before = tight.clone()
        output = torch.full((2, 512), -77, dtype=torch.int32)
        output_before = output.clone()
        with patch.object(token, "_native_ops", side_effect=AssertionError("tight odd must decline before factory")) as factory:
            self.assertIsNone(token.try_select_tokens(tight, torch.tensor([32769, 32769]), out=output, disposable=True))
        factory.assert_not_called()
        _bytes_equal(self, tight, tight_before)
        _bytes_equal(self, output, output_before)
        base = torch.full((1, 32768), torch.nan, dtype=torch.bfloat16)
        failed = Mock(side_effect=RuntimeError("raw selection failed"))
        with _arm("hierarchical", base, raw=failed, canonicalize=_canonical_cpu):
            with self.assertRaisesRegex(RuntimeError, "raw selection failed"):
                token.try_select_tokens(base, torch.tensor([32768]), disposable=True)
        failed.assert_called_once()
        self.assertTrue(base.isposinf().all())
        self.assertIs(token._native_ops, self.default)


@unittest.skipUnless(os.environ.get("RTP_V41_TEST_LOCAL_WORKSPACE_DEVICE") == "1", "explicit PPU device test")
class LocalPrefillHierarchicalTopKVariantsDevice(unittest.TestCase):
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
        torch.manual_seed(2121)

    def _paired(self, base, visible, *, padded=False, complete=False):
        rows, width = base.shape
        outputs = {}
        sanitized = base.nan_to_num(nan=torch.inf, posinf=torch.inf, neginf=-torch.inf)
        for name in ARMS:
            stride = (width + 128 + 15) // 16 * 16
            bank = torch.full((rows + 2, stride), 7, dtype=base.dtype, device=base.device) if padded else None
            scores = bank[1:-1, :width] if padded else base.clone()
            if padded:
                scores.copy_(base)
            guard = torch.full((rows * 512 + 16,), -77, dtype=torch.int32, device=base.device)
            out = guard[8:-8].view(rows, 512)
            observed = []
            with _arm(name, scores, observe=observed) as seen:
                got = token.try_select_tokens(scores, visible, out=out, disposable=True)
            self.assertEqual(seen, [False])
            self.assertIs(got, out)
            _bytes_equal(self, scores, sanitized)
            self.assertTrue((guard[:8] == -77).all() and (guard[-8:] == -77).all())
            if padded:
                self.assertTrue((bank[0] == 7).all() and (bank[-1] == 7).all())
                self.assertTrue((bank[1:-1, width:] == 7).all())
            if name == "hierarchical" and _narrow(scores):
                self.assertTrue(observed[0]["source_view_alias"])
                self.assertEqual(observed[0]["key_stride"][0] % 8, 0)
                self.assertEqual(observed[0]["key_alignment"], 0)
                self.assertEqual(observed[0]["key_dtype"], "torch.bfloat16")
                self.assertEqual(observed[0]["candidate_dtype"], "torch.bfloat16")
                _bytes_equal(self, observed[0]["keys"], observed[0]["keys_before"])
                if observed[0]["key_padding"] is not None:
                    self.assertTrue(observed[0]["key_padding"].isposinf().all())
            else:
                self.assertFalse(observed)
            outputs[name] = got.clone()
        for name, other in (("checked", "hierarchical"), ("hierarchical", "checked")):
            assert_equivalent(self, outputs[name], outputs[other], base, visible)
            _assert_oracle(self, outputs[name], base, visible,
                           rows=None if complete else sorted({0, rows // 2, rows - 1}))
        if complete:
            expected = reference(base, visible)
            for got in outputs.values():
                assert_equivalent(self, got, expected, base, visible)
        return outputs

    def test_actual_full_and_compact_shapes_preserve_exact_score_contract(self):
        for kind, compact in (("causal_style", False), ("wide_overflow", False), ("causal_style", True)):
            with self.subTest(kind=kind, compact=compact):
                self._paired(*_primary(kind, compact=compact), padded=not compact)

    def test_all_bf16_bits_and_coarse_overflow_511_512_513_exact(self):
        bits = torch.arange(65536, device="cuda:0", dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
        base = torch.stack((bits, bits.roll(3), bits.flip(0), bits.roll(7)))
        _bytes_equal(self, base[0], bits)  # Preserve every raw BF16 pattern in the fixture.
        self._paired(base, torch.full((4,), 65536, dtype=torch.int64, device=base.device), complete=True)
        edges = []
        for spread in (False, True):
            for count in (511, 512, 513):
                row = torch.full((32768,), -torch.inf, dtype=torch.bfloat16, device=base.device)
                values = torch.tensor([2.0**(80 + i % 31) for i in range(count)], dtype=row.dtype, device=row.device)
                row[torch.arange(count, device=row.device) * (8 if spread else 1)] = values
                edges.append(row)
        for negative in (False, True):
            for count in (511, 512, 513):
                row = torch.full((32768,), -torch.inf, dtype=torch.bfloat16, device=base.device)
                codes = torch.arange(1, count + 1, dtype=torch.int32, device=row.device)
                if negative:
                    codes = codes | 32768
                row[torch.arange(count, device=row.device) * 8] = codes.to(torch.int16).view(torch.bfloat16)
                edges.append(row)
        zero = torch.zeros(32768, dtype=torch.bfloat16, device=base.device)
        zero[::2] = -0.0
        edges.extend((zero, torch.full_like(zero, -torch.inf)))
        for spread in (False, True):
            for count in (1, 511, 512, 513):
                row = torch.randn(32768, dtype=torch.bfloat16, device=base.device)
                locations = torch.arange(count, device=row.device) * (8 if spread else 1)
                if spread:
                    codes = torch.tensor([32705, -63], dtype=torch.int16, device=row.device).view(torch.bfloat16)
                    row[locations] = codes.repeat((count + 1) // 2)[:count]
                else:
                    row[locations] = torch.inf
                edges.append(row)
        base = torch.stack(edges)
        self.assertEqual(base.shape[0], 22)
        self._paired(base, torch.full((22,), 32768, dtype=torch.int64, device=base.device), complete=True)

    def test_partial_poison_all_remainders_no_pin_short_empty_and_real_negative_inf(self):
        self._paired(*_partial_cases("cuda:0"), complete=True)

    def test_unaligned_logical_keys_padded_views_out_guards_and_old_routes(self):
        for width in (32776, 131072, 32769, 16384):
            base = torch.randn(7, width, dtype=torch.bfloat16, device="cuda:0")
            base[0, :513] = torch.nan
            visible = torch.tensor([width, 0, 1, 511, 512, 513, 2**40], dtype=torch.int64, device=base.device)
            self._paired(base, visible, padded=True, complete=True)
        tight = torch.full((2, 32769), torch.nan, dtype=torch.bfloat16, device="cuda:0")
        before = tight.clone()
        output = torch.full((2, 512), -77, dtype=torch.int32, device=tight.device)
        output_before = output.clone()
        visible = torch.full((2,), 32769, dtype=torch.int64, device=tight.device)
        with patch.object(token, "_native_ops", side_effect=AssertionError("tight odd must decline before factory")) as factory:
            self.assertIsNone(token.try_select_tokens(tight, visible, out=output, disposable=True))
        factory.assert_not_called()
        _bytes_equal(self, tight, before)
        _bytes_equal(self, output, output_before)

    def test_balanced_actual_full_wrapper_latency_counts_all_hierarchy_work(self):
        samples = []
        for kind in ("causal_style", "wide_overflow"):
            base, visible = _primary(kind)
            expected = self._paired(base, visible)["checked"]
            for name in ARMS:
                with _arm(name, base):
                    for _ in range(2):
                        token.try_select_tokens(base.clone(), visible, disposable=True)
            torch.cuda.synchronize()
            for position, name in enumerate(("checked", "hierarchical", "hierarchical", "checked") * 4):
                scores = base.clone()
                output = torch.empty((1024, 512), dtype=torch.int32, device=base.device)
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                with _arm(name, scores) as seen:
                    torch.cuda.synchronize()  # Clone/allocation/JIT/import outside both timers.
                    start.record()
                    before = time.perf_counter_ns()
                    got = token.try_select_tokens(scores, visible, out=output, disposable=True)
                    end.record()
                    end.synchronize()
                    after = time.perf_counter_ns()
                self.assertEqual(seen, [False])
                gpu_ms, host_ms = start.elapsed_time(end), (after - before) / 1e6
                self.assertTrue(math.isfinite(gpu_ms) and gpu_ms > 0)
                self.assertTrue(math.isfinite(host_ms) and host_ms > 0)
                samples.append(dict(case=kind, position=position, arm=name, gpu_ms=gpu_ms, host_ms=host_ms))
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
            (root / "local-prefill-hierarchical-topk.samples.json").write_text(json.dumps(dict(
                shape=[1024, 65536], bounds=None, starts="zero", seed=2121, stream="same default",
                host_scope="wrapper submission through end-event wait; clones/JIT/checks excluded",
                hierarchy="BF16 amax/partial repair/padding + raw512 + gather4096 + raw512 + remap + final canonicalize",
                sanitizer="original full tile", finite_finish="original once, no refill",
                provider_sources=_identity(), samples=samples, summaries=summaries,
                decision="no speed threshold; L0 decides from exact and full cost",
            ), indent=2))

    def test_profile_two_register_three_canonicalize_original_finish_and_late_old_route(self):
        base, visible = _primary("causal_style")
        original = token._native_ops
        for name in ARMS:
            with _arm(name, base):
                token.try_select_tokens(base.clone(), visible, disposable=True)
        torch.cuda.synchronize()
        records, outputs = {}, {}
        for name in ARMS:
            scores = base.clone()
            torch.cuda.synchronize()
            with _arm(name, scores):
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                        torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as profile:
                    outputs[name] = token.try_select_tokens(scores, visible, disposable=True)
                    torch.cuda.synchronize()
            events = _trace_events(profile, f"hierarchical-topk-{name}")
            kernels = [e["name"] for e in events if e.get("cat") == "kernel"]
            hierarchical = name == "hierarchical"
            self.assertEqual(sum("topk_prefill_bf16_register_kernel" in k for k in kernels), 2 if hierarchical else 0)
            self.assertEqual(sum("topk_prefill_bf16_checked_kernel" in k for k in kernels), 0 if hierarchical else 1)
            self.assertEqual(sum("_canonicalize_indices" in k for k in kernels), 3 if hierarchical else 1)
            self.assertEqual(sum("_prefill_topk_finite_kernel" in k for k in kernels), 1)
            self.assertEqual(sum(e.get("name") == "aten::nan_to_num_" for e in events), 1)
            self.assertFalse(any(e.get("name") == "aten::topk" for e in events))
            self.assertFalse(any(e.get("cat") == "gpu_memcpy" and "DtoH" in e.get("name", "") for e in events))
            if hierarchical:
                self.assertFalse(any("_selective_" in k or "_fallback_" in k or "_prepare_rows" in k for k in kernels))
            records[name] = dict(shape=list(scores.shape), score_stride=list(scores.stride()), kernels=kernels, bounds=None)
        self.assertIs(token._native_ops, original)
        assert_equivalent(self, outputs["hierarchical"], outputs["checked"], base, visible)
        late = torch.randn(140, 16384, dtype=torch.bfloat16, device=base.device)
        late_visible = torch.full((140,), 16384, dtype=torch.int64, device=base.device)
        token.try_select_tokens(late.clone(), late_visible, disposable=True)
        scores = late.clone()
        torch.cuda.synchronize()
        with _arm("hierarchical", scores):
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as profile:
                got = token.try_select_tokens(scores, late_visible, disposable=True)
                torch.cuda.synchronize()
        events = _trace_events(profile, "hierarchical-topk-late-old-route")
        self.assertEqual(sum(e.get("cat") == "kernel" and "topk_prefill_bf16_register_kernel" in e.get("name", "")
                             for e in events), 1)
        assert_equivalent(self, got, reference(late, late_visible), late, late_visible)
        with _artifacts() as root:
            (root / "local-prefill-hierarchical-topk.profile.json").write_text(json.dumps(dict(
                provider_sources=_identity(), arms=records, late="unmodified register path",
            ), indent=2))


if __name__ == "__main__":
    unittest.main()
