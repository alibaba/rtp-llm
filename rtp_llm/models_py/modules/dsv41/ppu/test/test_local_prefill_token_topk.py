"""The existing unordered Prefill token boundary on PPU89, without tolerances.

Registration must include this module, fp8/test/test_v41_prefill_topk.py
(unchanged assert_equivalent/reference), and existing v41 PPU/native sources.
CPU tests use an explicit test-only native double; device tests require ZW89.
"""

import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.modules.dsv41.fp8 import _v41_sparse_prefill_indexer as sparse
from rtp_llm.models_py.modules.dsv41.fp8.attention_v41 import mask_candidate_logits, select_candidate_blocks
from rtp_llm.models_py.modules.dsv41.fp8.test.test_v41_prefill_topk import assert_equivalent, reference
from rtp_llm.models_py.modules.dsv41.ppu import _quant, prefill_token_topk as token


def _bytes_equal(case, actual, expected):
    case.assertTrue(torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)))


def _raw_bytes(tensor):
    if tensor.layout != torch.strided:
        return tensor.to_dense().contiguous().view(torch.uint8)
    storage = tensor.untyped_storage()
    # Read the allocation directly, without resolving a negative/conjugate
    # view's logical values. This also covers guards/padding and input aliases.
    return torch.empty(0, dtype=torch.uint8, device=tensor.device).set_(
        storage, 0, (storage.nbytes(),), (1,),
    )


def _intervals(logits, visible, bounds=None):
    ends = visible.clamp(0, logits.shape[1]).int()
    starts = torch.zeros_like(ends)
    if bounds is not None:
        ends = torch.minimum(ends, bounds[1].clamp(0, logits.shape[1]))
        starts = torch.minimum(bounds[0].clamp(0, logits.shape[1]), ends)
    return starts, ends


def _reference(logits, visible, bounds=None):
    starts, ends = _intervals(logits, visible, bounds)
    columns = torch.arange(logits.shape[1], device=logits.device)
    masked = logits.masked_fill((columns[None] < starts[:, None]) | (columns[None] >= ends[:, None]), -torch.inf)
    scores, selected = masked.topk(512, dim=-1)
    return torch.where(scores.isfinite(), selected, -1).int()


def _assert_oracle(case, got, logits, visible, bounds=None, rows=None):
    """Independent h/f/q/tau rule, using original BF16 values and Python sets."""
    starts, ends = _intervals(logits, visible, bounds)
    rows = range(logits.shape[0]) if rows is None else rows
    for row in rows:
        start, end = int(starts[row]), int(ends[row])
        values = logits[row].float().cpu().tolist()
        selected = [int(index) for index in got[row].cpu().tolist() if index >= 0]
        finite = [(index, values[index]) for index in range(start, end) if math.isfinite(values[index])]
        h = sum(math.isnan(values[index]) or values[index] == math.inf for index in range(start, end))
        q = min(len(finite), max(512 - h, 0))
        case.assertEqual(len(selected), q, (row, h, len(finite), q))
        case.assertEqual(len(set(selected)), q)
        case.assertTrue(all(start <= index < end and math.isfinite(values[index]) for index in selected))
        if q:
            tau = sorted((value for _, value in finite), reverse=True)[q - 1]
            required = {index for index, value in finite if value > tau}
            allowed = {index for index, value in finite if value >= tau}
            case.assertTrue(required <= set(selected) <= allowed)
            if len(allowed) == q:
                case.assertEqual(set(selected), allowed, "non-tied cutoff membership must be exact")


def _assert_semantics(case, got, logits, visible, bounds=None, sample=None):
    expected = reference(logits, visible) if bounds is None else _reference(logits, visible, bounds)
    # Reuse the established exact multiset/uniqueness/count contract unchanged.
    assert_equivalent(case, got, expected, logits, visible)
    _assert_oracle(case, got, logits, visible, bounds, sample)


class _BoundsDouble:
    def __getitem__(self, grid):
        def launch(visible, ends, rows, width, stride, tile):
            ends.copy_(visible.clamp(0, width).int())
        return launch


def _native_double(logits, starts, ends, out):
    # Test-only native admission/composition double, never serving dispatch.
    out.fill_(-1)
    for row in range(logits.shape[0]):
        start, end = int(starts[row]), int(ends[row])
        count = min(512, end - start)
        if count:
            chosen = logits[row, start:end].topk(count).indices + start
            out[row, :count] = chosen.sort().values.int()
    return out


def _finish_double(logits, ends, output):
    values = logits.gather(1, output.long().clamp_min(0))
    keep = (output >= 0) & (output < ends[:, None]) & values.isfinite()
    return output.masked_fill_(~keep, -1)


def _adversarial(device, width=8192):
    values = torch.randn(24, width, device=device, dtype=torch.bfloat16)
    values[0].fill_(1)                       # thousands of cutoff ties
    values[1].fill_(-3)
    values[1, :511] = 3
    values[1, 511:1024] = 2                 # 513 equal cutoff candidates
    values[2].fill_(0)
    values[2, ::2] = -0.0
    values[3].fill_(-torch.inf)
    values[4].fill_(torch.inf)
    values[5].fill_(torch.nan)
    bits = torch.tensor([32705, -63, 32767, -1], dtype=torch.int16, device=device).view(torch.bfloat16)
    for row, high in ((6, 1), (7, 511), (8, 512), (9, 513), (10, 1024)):
        values[row, :high] = bits.repeat((high + 3) // 4)[:high]
    values[11, :511] = torch.inf
    values[11, 511] = -torch.inf
    values[12, :17] = bits.repeat(5)[:17]
    values[12, 17:33] = torch.inf
    maximum = torch.finfo(torch.bfloat16).max
    values[13, :4] = torch.tensor([maximum, -maximum, 3e38, -3e38], device=device, dtype=torch.bfloat16)
    values[14].fill_(1)
    values[14, :512] = 1.0078125             # exactly 512 finite winners
    values[15].fill_(1)
    values[15, :513] = 1.0078125
    values[16].fill_(-1)
    values[16, :512] = torch.arange(512, device=device).to(torch.bfloat16)
    ends = torch.full((24,), width, dtype=torch.int64, device=device)
    ends[17:] = torch.tensor([-7, 0, 1, 511, 512, 513, 2**40], device=device)
    # Invisible values must not enter native selection, even after sanitizing.
    values[18:] = torch.inf
    values[19, 0] = -1
    return values, ends


class LocalPrefillTokenTopKCPU(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {token.ENV: "1"})
        env.start()
        self.addCleanup(env.stop)
        guard = torch.no_grad()
        guard.__enter__()
        self.addCleanup(guard.__exit__, None, None, None)

    def test_unsupported_metadata_rolls_back_before_launch_or_mutation(self):
        base = torch.full((3, 1152), torch.nan, dtype=torch.bfloat16)
        good = base[:, :1024]
        contiguous = good.contiguous()
        shared_output = torch.full((3, 512), 19, dtype=torch.int32)
        visible = torch.tensor([0, 513, 1024])
        cases = [
            (good, visible, {}, "actual_cpu"),
            (good, visible, {"disposable": False}, "ownership"),
            (good.float(), visible, {}, "fp32"),
            (good[:, :511], visible, {}, "short_width"),
            (torch.empty(0, 1024, dtype=torch.bfloat16), visible[:0], {}, "empty_rows"),
            (torch.empty(1, 262152, dtype=torch.bfloat16), visible[:1], {}, "wide"),
            (base[:, ::2], visible, {}, "columns"),
            (base[:, 1:1025], visible, {}, "alignment"),
            (good, visible.float(), {}, "visible_dtype"),
            (good, visible[:2], {}, "visible_shape"),
            (good, visible, {"bounds": (visible.int(),)}, "bounds_count"),
            (good, visible, {"bounds": (visible, visible)}, "bounds_dtype"),
            (good, visible, {"out": torch.full((3, 1024), 87, dtype=torch.int32)[:, ::2]}, "out_stride"),
            (good, visible, {"out": torch.empty(3, 512, dtype=torch.int64)}, "out_dtype"),
            (contiguous, visible, {"out": contiguous.view(torch.int32)}, "out_alias"),
            (good, good.view(torch.int32)[0, :3], {}, "metadata_alias"),
            (good, shared_output.view(-1)[:3], {"out": shared_output}, "out_metadata_alias"),
            (good, visible[:1].expand(3), {}, "visible_broadcast"),
            (torch._neg_view(good), visible, {}, "negative_scores"),
            (good, torch._neg_view(visible), {}, "negative_visible"),
            (good, visible, {"bounds": (torch._neg_view(visible.int()), visible.int())}, "negative_bounds"),
            (good, visible, {"out": torch._neg_view(shared_output)}, "negative_output"),
            (good.to_sparse(), visible, {}, "sparse_scores"),
            (good, visible.to_sparse(), {}, "sparse_visible"),
            (good, visible, {"bounds": (visible.int().to_sparse(), visible.int())}, "sparse_bounds"),
            (good, visible, {"out": shared_output.to_sparse()}, "sparse_output"),
        ]
        for logits, lengths, arguments, reason in cases:
            with self.subTest(reason=reason):
                destination = arguments.get("out")
                inputs = (logits, lengths, *arguments.get("bounds", ()))
                if destination is not None:
                    inputs += (destination,)
                before = tuple(_raw_bytes(t).clone() for t in inputs)
                supplied = dict(disposable=True, **{k: v for k, v in arguments.items() if k != "disposable"})
                supplied["disposable"] = arguments.get("disposable", True)
                device = patch.object(token, "_device_supported", return_value=reason != "actual_cpu")
                with device, patch.object(token, "_native_ops", side_effect=AssertionError("unsupported launch")):
                    self.assertIsNone(token.try_select_tokens(logits, lengths, **supplied))
                for tensor, snapshot in zip(inputs, before):
                    self.assertTrue(torch.equal(_raw_bytes(tensor), snapshot))
        for reason in ("disabled", "grad", "requires_grad"):
            scores = good.clone().requires_grad_(reason == "requires_grad")
            with self.subTest(reason=reason), patch.object(token, "_device_supported", return_value=True):
                with patch.dict(os.environ, {token.ENV: "0" if reason == "disabled" else "1"}), torch.set_grad_enabled(reason == "grad"):
                    self.assertIsNone(token.try_select_tokens(scores, visible, disposable=True))

    def test_composition_reuses_bounds_and_finite_filter_without_refilling(self):
        logits, visible = _adversarial("cpu")
        original = logits.clone()
        native = Mock(wraps=_native_double)
        finish = Mock(wraps=_finish_double)
        with patch.object(token, "_device_supported", return_value=True), patch.object(
            token, "_native_ops", return_value=(native, None)
        ), patch.object(token._topk, "_prefill_topk_bounds_kernel", _BoundsDouble()), patch.object(
            token._topk, "finish_tokens", finish
        ):
            result = token.try_select_tokens(logits, visible, disposable=True)
        native.assert_called_once()
        finish.assert_called_once()
        self.assertFalse(logits.isnan().any())
        self.assertTrue(torch.equal(logits.isposinf(), original.isnan() | original.isposinf()))
        self.assertTrue(torch.equal(logits.isneginf(), original.isneginf()))
        _assert_semantics(self, result, original, visible)

    def test_prepared_bounds_share_storage_and_use_radix_then_finish(self):
        logits, visible = _adversarial("cpu")
        original = logits.clone()
        # Same allocation, disjoint starts/ends views is a legal native ABI.
        allocation = torch.empty(48, dtype=torch.int32)
        starts, ends = allocation[:24], allocation[24:]
        starts.copy_(torch.arange(24) * 3 - 9)
        ends.copy_(torch.arange(24) * 400 + 17)
        ends[0], starts[1], ends[2] = -7, 2**30, 2**30
        native, canonicalize = Mock(wraps=_native_double), Mock(side_effect=lambda out: out)
        with patch.object(token, "_device_supported", return_value=True), patch.object(
            token, "_native_ops", return_value=(native, canonicalize)
        ) as load, patch.object(token._topk, "_prefill_topk_bounds_kernel", _BoundsDouble()), patch.object(
            token._topk, "finish_tokens", side_effect=_finish_double
        ):
            result = token.try_select_tokens(logits, visible, bounds=(starts, ends), disposable=True)
        load.assert_called_once_with(True)
        canonicalize.assert_called_once()
        _assert_semantics(self, result, original, visible, (starts, ends))

    def test_missing_dependency_returns_before_mutation_and_execution_errors_propagate(self):
        logits = torch.full((2, 1024), torch.nan, dtype=torch.bfloat16)
        visible = torch.full((2,), 1024, dtype=torch.int32)
        before = logits.clone()
        with patch.object(token, "_device_supported", return_value=True), patch.object(
            token, "_native_ops", side_effect=ImportError("native missing")
        ):
            self.assertIsNone(token.try_select_tokens(logits, visible, disposable=True))
        _bytes_equal(self, logits, before)
        failed = Mock(side_effect=RuntimeError("native launch failure"))
        with patch.object(token, "_device_supported", return_value=True), patch.object(
            token, "_native_ops", return_value=(failed, None)
        ), patch.object(token._topk, "_prefill_topk_bounds_kernel", _BoundsDouble()):
            with self.assertRaisesRegex(RuntimeError, "native launch failure"):
                token.try_select_tokens(logits, visible, disposable=True)
        failed.assert_called_once()
        self.assertTrue(logits.isposinf().all(), "post-admission failure must not be presented as rollback")

    def test_independent_oracle_accepts_ties_but_rejects_refill_or_wrong_values(self):
        scores = torch.zeros(1, 1024, dtype=torch.bfloat16)
        visible = torch.tensor([1024])
        first = torch.arange(512, dtype=torch.int32)[None]
        second = torch.arange(512, 1024, dtype=torch.int32)[None]
        for selected in (first, second.flip(1)):
            _assert_oracle(self, selected, scores, visible)
        scores[0, :511] = torch.nan
        correct = torch.full_like(first, -1)
        correct[0, 0] = 511
        _assert_oracle(self, correct, scores, visible)
        with self.assertRaises(AssertionError):
            _assert_oracle(self, second, scores, visible)
        scores.zero_()
        scores[0, 0] = 1
        with self.assertRaises(AssertionError):
            _assert_oracle(self, second, scores, visible)


@unittest.skipUnless(os.environ.get("RTP_V41_TEST_LOCAL_WORKSPACE_DEVICE") == "1", "explicit PPU device test")
class LocalPrefillTokenTopKDevice(unittest.TestCase):
    def setUp(self):
        self.assertTrue(_quant.is_ppu(), "requires an actual ZW PPU")
        self.assertEqual(torch.cuda.get_device_capability(), (8, 9))
        env = patch.dict(os.environ, {token.ENV: "1"})
        env.start()
        self.addCleanup(env.stop)
        guard = torch.no_grad()
        guard.__enter__()
        self.addCleanup(guard.__exit__, None, None, None)

    def test_unresolved_views_and_sparse_layouts_roll_back_before_native_work(self):
        logits = torch.full((3, 1024), torch.nan, device="cuda", dtype=torch.bfloat16)
        visible = torch.full((3,), 1024, device="cuda", dtype=torch.int32)
        output = torch.full((3, 512), 19, device="cuda", dtype=torch.int32)
        cases = (
            (torch._neg_view(logits), visible, {}),
            (logits, torch._neg_view(visible), {}),
            (logits, visible, {"bounds": (torch._neg_view(visible), visible.clone())}),
            (logits, visible, {"out": torch._neg_view(output)}),
            (logits.to_sparse(), visible, {}),
            (logits, visible.to_sparse(), {}),
            (logits, visible, {"out": output.to_sparse()}),
        )
        for scores, lengths, arguments in cases:
            inputs = (scores, lengths, *arguments.get("bounds", ()))
            if "out" in arguments:
                inputs += (arguments["out"],)
            snapshots = tuple(_raw_bytes(t).clone() for t in inputs)
            with patch.object(token, "_native_ops", side_effect=AssertionError("unsupported native launch")):
                self.assertIsNone(token.try_select_tokens(scores, lengths, disposable=True, **arguments))
            for tensor, snapshot in zip(inputs, snapshots):
                self.assertTrue(torch.equal(_raw_bytes(tensor), snapshot))

    def test_native_nonfinite_ties_signed_zero_short_empty_and_poisoned_tail(self):
        for width in (8192, 65536):
            with self.subTest(width=width):
                logits, visible = _adversarial("cuda", width)
                original = logits.clone()
                got = token.try_select_tokens(logits, visible, disposable=True)
                self.assertIsNotNone(got)
                _assert_semantics(self, got, original, visible)
                # Native canonicalization is repeatable, although Torch tie
                # membership/order is intentionally not prescribed.
                again = token.try_select_tokens(original.clone(), visible, disposable=True)
                self.assertTrue(torch.equal(got, again))

    def test_actual_full_compact_wide_and_chunk_stride_preserve_canaries(self):
        for rows, width in ((4096, 65536), (4097, 65536), (140, 131072),
                            (3, 262144), (3, 65537), (3, 512)):
            with self.subTest(rows=rows, width=width):
                # A real chunk view, with preceding/following rows and padded
                # native row stride. Only the disposable score view may change.
                row_stride = ((width + 128 + 15) // 16) * 16
                allocation = torch.full((rows + 2, row_stride), 7, device="cuda", dtype=torch.bfloat16)
                logits = allocation[1:rows + 1, :width]
                self.assertEqual(logits.stride(0) % 8, 0)
                self.assertEqual(logits.data_ptr() % 16, 0)
                logits.normal_()
                logits[0, 0] = torch.nan
                visible = torch.full((rows,), width, device="cuda", dtype=torch.int64)
                visible[-1] = 513 if width > 512 else 0
                original = logits.clone()
                outputs = torch.full((rows * 512 + 16,), -77, device="cuda", dtype=torch.int32)
                out = outputs[8:-8].view(rows, 512)
                got = token.try_select_tokens(logits, visible, out=out, disposable=True)
                self.assertIs(got, out)
                _assert_semantics(self, got, original, visible, sample=sorted({0, rows // 2, rows - 1}))
                self.assertTrue((outputs[:8] == -77).all() and (outputs[-8:] == -77).all())
                self.assertTrue((allocation[0] == 7).all() and (allocation[-1] == 7).all())
                self.assertTrue((allocation[1:-1, width:] == 7).all())

    def test_native_nonzero_starts_clamped_bounds_and_strided_destination_rollback(self):
        logits, visible = _adversarial("cuda", 65536)
        original = logits.clone()
        allocation = torch.empty(48, device="cuda", dtype=torch.int32)
        starts, ends = allocation[:24], allocation[24:]
        starts.copy_(torch.arange(24, device="cuda") * 13 - 17)
        ends.fill_(65536)
        ends[0], starts[1], ends[2] = -7, 2**30, 2**30
        result = token.try_select_tokens(logits, visible, bounds=(starts, ends), disposable=True)
        self.assertIsNotNone(result)
        _assert_semantics(self, result, original, visible, (starts, ends))
        out = torch.full((24, 1024), 71, device="cuda", dtype=torch.int32)[:, ::2]
        before = original.clone()
        self.assertIsNone(token.try_select_tokens(original, visible, out=out, disposable=True))
        _bytes_equal(self, original, before)
        self.assertTrue((out == 71).all())

    def test_candidate_publication_mask_and_sparse_remap_keep_request_domains(self):
        logits = torch.randn(4, 32768, device="cuda", dtype=torch.bfloat16)
        visible = torch.tensor([32768, 29001, 17001, 0], device="cuda", dtype=torch.int64)
        columns = torch.arange(32768, device="cuda")
        logits.masked_fill_(columns[None] >= visible[:, None], -torch.inf)
        published = select_candidate_blocks(logits, visible, 8, 2048)
        frozen = published.clone()
        mask_candidate_logits(logits, published, 8)
        original = logits.clone()
        selected = token.try_select_tokens(logits, visible, disposable=True)
        self.assertIsNotNone(selected)
        _assert_semantics(self, selected, original, visible)
        self.assertTrue(torch.equal(published, frozen), "token selection must not republish K2048 blocks")
        for row in range(4):
            ids = selected[row][selected[row] >= 0]
            self.assertTrue(torch.isin(ids // 8, published[row]).all())
        # A sorted/deduplicated existing provider plan, including duplicate and
        # invalid published candidates. Remap itself is the existing kernel.
        candidates = torch.tensor([[7, 1, 7, 3, -1, 9, 1, 11]] * 4, device="cuda", dtype=torch.int32)
        logical_visible = torch.tensor([128, 79, 31, 0], device="cuda", dtype=torch.int32)
        plan = sparse.prepare_plan(candidates, logical_visible, 256, 8, ppu=True)
        self.assertIsNotNone(plan)
        # Native token K512 requires a >=512 physical score width. Use a real
        # 128-block plan so the sparse score domain is 1024, not an invented ABI.
        candidates = torch.arange(128, device="cuda", dtype=torch.int32)[None].repeat(4, 1)
        candidates[:, -4:] = candidates[:, :4]
        plan = sparse.prepare_plan(candidates, logical_visible, 2048, 8, ppu=True)
        self.assertIsNotNone(plan)
        sparse_logits = torch.randn(4, plan.sparse_columns, device="cuda", dtype=torch.bfloat16)
        sparse_logits[:, :2] = torch.nan
        sparse_original = sparse_logits.clone()
        selected = token.try_select_tokens(sparse_logits, plan.end, disposable=True)
        self.assertIsNotNone(selected)
        _assert_semantics(self, selected, sparse_original, plan.end)
        mapped = sparse.remap(selected, plan, logits=sparse_logits)
        self.assertIsNotNone(mapped)
        blocks = plan.sparse_indices.gather(1, selected.long().clamp_min(0) // 8)
        expected = blocks * 8 + selected.clamp_min(0) % 8
        keep = (selected >= 0) & (selected < plan.end[:, None]) & (expected < logical_visible[:, None])
        expected = expected.masked_fill(~keep, -1)
        self.assertTrue(torch.equal(mapped, expected))

    def test_actual_profile_native_wide_sanitizer_finish_and_no_torch_topk(self):
        base = torch.randn(140, 65536, device="cuda", dtype=torch.bfloat16)
        base[0, :3] = torch.nan
        visible = torch.full((140,), 65536, device="cuda", dtype=torch.int32)
        bounds = (torch.full_like(visible, 7), visible.clone())
        token.try_select_tokens(base.clone(), visible, disposable=True)  # JIT outside capture.
        token.try_select_tokens(base.clone(), visible, bounds=bounds, disposable=True)
        reference(base, visible)
        traces = []
        for name, native, prepared in (("torch", False, None), ("checked", True, None), ("radix", True, bounds)):
            logits = base.clone()
            torch.cuda.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as profile:
                got = token.try_select_tokens(logits, visible, bounds=prepared, disposable=True) if native else reference(logits, visible)
                torch.cuda.synchronize()
            self.assertIsNotNone(got)
            _assert_semantics(self, got, base, visible, bounds=prepared, sample=(0, 70, 139))
            root = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
            with tempfile.TemporaryDirectory(prefix="local-token-topk-") as temporary:
                path = Path(root or temporary) / f"local-prefill-token-topk-{name}.trace.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                profile.export_chrome_trace(str(path))
                traces.append(json.loads(path.read_text())["traceEvents"])
        old, checked, radix = traces
        self.assertTrue(any(e.get("name") == "aten::topk" for e in old))
        self.assertTrue(any(e.get("cat") == "kernel" and "gatherTopK" in e["name"] for e in old))
        for current, backend in ((checked, "topk_prefill_bf16_checked_kernel"), (radix, "_selective_scatter")):
            self.assertFalse(any(e.get("name") == "aten::topk" for e in current))
            self.assertFalse(any(e.get("cat") == "kernel" and "mbtopk::" in e["name"] for e in current))
            self.assertTrue(any(e.get("name") == "aten::nan_to_num_" for e in current))
            kernels = [e["name"] for e in current if e.get("cat") == "kernel"]
            for required in (backend, "_prefill_topk_bounds_kernel", "_prefill_topk_finite_kernel", "_canonicalize_indices"):
                self.assertTrue(any(required in name for name in kernels), required)


if __name__ == "__main__":
    unittest.main()
