"""Pinned SG native selection: adapter contracts, device semantics and timings.

Nonfinite cases compare the original native entries, not the old RTP finite
filter oracle. Finite cases additionally use an independent score threshold.
No test asserts a speed threshold or requires an order among equal scores.
"""

import hashlib
import json
import os
from pathlib import Path
import statistics
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.modules.dsv41.ppu import _quant
from rtp_llm.models_py.modules.dsv41.ppu import local_prefill_sg_topk as sg


def _bundle():
    from rtp_llm.models_py.kernels.ppu.jit import sglang_jit

    return Path(sglang_jit.__file__).parent / "dsv4_ppu" / "prefill_topk"


def _bytes(test, left, right):
    test.assertTrue(torch.equal(left.detach().cpu().reshape(-1).view(torch.uint8),
                                right.detach().cpu().reshape(-1).view(torch.uint8)))


def _multiset(values):
    # NaN payloads and arbitrary tied IDs are native details, not a new order
    # requirement. Keep counts of NaN/Inf and the finite score multiset.
    values = values.float()
    return (int(values.isnan().sum()), int(values.isposinf().sum()),
            int(values.isneginf().sum()), sorted(values[values.isfinite()].tolist()))


def _assert_finite(test, indices, scores, lengths, blocks=None):
    indices, scores, lengths = indices.cpu().long(), scores.cpu(), lengths.cpu()
    blocks = blocks.cpu().long() if blocks is not None else None
    for row, end in enumerate(lengths.tolist()):
        selected = indices[row][indices[row] >= 0]
        test.assertEqual(selected.numel(), min(512, end))
        test.assertEqual(selected.unique().numel(), selected.numel())
        if blocks is None:
            test.assertTrue(bool((selected < end).all()))
            slots = selected
        else:
            # Fixtures use unique page IDs, so invert the native transform.
            inverse = {int(page) * 8 + lane: col * 8 + lane
                       for col, page in enumerate(blocks[row].tolist())
                       for lane in range(8)}
            slots = torch.tensor([inverse[int(i)] for i in selected], dtype=torch.long)
            test.assertTrue(bool((slots < end).all()))
        wanted = torch.topk(scores[row, :end].float(), min(512, end)).values
        got = scores[row, slots].float()
        test.assertEqual(_multiset(got), _multiset(wanted))
        test.assertTrue(bool((indices[row][indices[row] < 0] == -1).all()))


def _direct_dense(scores, ends, block_k, *, out=None, blocks_out=None,
                  check_blocks=None, use_blocks=None):
    """Original SG three-call pipeline, independently of adapter dispatch."""
    from rtp_llm.models_py.kernels.ppu.jit.sglang_jit import load_sglang_kernel

    # Export the two untouched host methods individually. This is deliberately
    # independent of the adapter's optional-block host export for ABI parity.
    ragged = load_sglang_kernel(
        "prefill_topk_reference_ragged", "csrc/deepseek_v4/topk_v2.cuh",
        "sglang::TopKKernel<false>::transform_ragged", (8, 9), bundle="prefill_topk")
    candidate = load_sglang_kernel(
        "prefill_topk_reference_candidates", "csrc/deepseek_v4/topk_v2.cuh",
        "sglang::TopKKernel<false>::transform_candidates", (8, 9), bundle="prefill_topk")
    rows, width = scores.shape
    keys = torch.empty((rows, ((width // 8 + 3) // 4) * 4), dtype=torch.float32, device=scores.device)
    blocks = (torch.empty((rows, block_k), dtype=torch.int32, device=scores.device)
              if blocks_out is None else blocks_out)
    out = (torch.empty((rows, 512), dtype=torch.int32, device=scores.device)
           if out is None else out)
    zero = torch.zeros_like(ends)
    if use_blocks is None:
        sg._module("amax").forward(scores, ends, keys, 0)
        ragged.forward(keys, (ends + 7) // 8, None, zero, blocks)
    else:
        blocks = use_blocks
    if check_blocks is not None:
        check_blocks(blocks)
    candidate.forward(scores, ends, blocks, zero, out)
    return out, blocks


class LocalPrefillSGTopKCPU(unittest.TestCase):
    def setUp(self):
        guard = torch.no_grad()
        guard.__enter__()
        self.addCleanup(guard.__exit__, None, None, None)
        mocked = patch.object(sg, "_device_supported", return_value=True)
        mocked.start()
        self.addCleanup(mocked.stop)

    def test_frozen_manifest_and_native_header_identity(self):
        root = _bundle()
        manifest = json.loads((root / "source-manifest.json").read_text())
        self.assertEqual(manifest["upstream_commit"], "3bc4f87d5f9bb6f04b724540fb12a3acdc62d2fa")
        for name, expected in manifest["sources"].items():
            self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), expected, name)
        expected = {
            "block_amax.cuh": "57edc969d1ad4edaece2a1c4e1b7182c8c134ae94d059404ba8a6d0e6edba794",
            "topk_v2.cuh": "158292a6c3cd43002b6bf4254df9abf691347e7ad49e55a4662a3a11336e0f80",
            "topk_bf16_small.cuh": "7c53f8c13e6cbf0ae1ba96cab0566def897b681abfddb2b951bb9df1503b7408",
        }
        for name, digest in expected.items():
            self.assertEqual(manifest["sources"]["csrc/deepseek_v4/" + name], digest)
        self.assertIn("Apache License", (root / "LICENSE").read_text())

    def test_dense_publication_identity_one_amax_and_no_cleanup(self):
        scores = torch.full((2, 8192), torch.nan)
        scores[0, 0], scores[0, 1] = torch.inf, -torch.inf
        before = scores.clone()
        ends = torch.tensor([8192, 513], dtype=torch.int32)
        out = torch.empty((2, 512), dtype=torch.int32)
        blocks = torch.empty((2, 2048), dtype=torch.int32)
        calls = []

        def load(kind):
            def forward(*args):
                calls.append((kind, args))
                if kind == "amax":
                    args[2].fill_(torch.nan)
                else:
                    args[-1].fill_(17 if args[2] is None else 31)
            return Mock(forward=Mock(side_effect=forward))

        with patch.object(sg, "_module", side_effect=load):
            got = sg.select_dense(scores, ends, out, block_topk=2048, blocks_out=blocks)
        self.assertIs(got[0], out)
        self.assertIs(got[1], blocks)
        self.assertEqual([kind for kind, _ in calls], ["amax", "select", "select"])
        self.assertIs(calls[0][1][0], scores)
        self.assertIs(calls[0][1][1], ends)
        self.assertIsNone(calls[1][1][2])
        self.assertIs(calls[1][1][-1], blocks)
        self.assertIs(calls[2][1][2], blocks)
        self.assertIs(calls[2][1][0], scores)
        self.assertTrue(torch.equal(calls[1][1][1], (ends + 7) // 8))
        self.assertTrue(torch.equal(out, torch.full_like(out, 31)))
        _bytes(self, scores, before)

    def test_sparse_native_mapping_identity_and_errors_propagate(self):
        scores = torch.full((2, 16384), torch.nan, dtype=torch.bfloat16)
        ends = torch.tensor([0, 16381], dtype=torch.int32)
        blocks = torch.arange(4096, dtype=torch.int32).reshape(2, 2048)
        out = torch.empty((2, 512), dtype=torch.int32)
        native = Mock()
        with patch.object(sg, "_module", return_value=native):
            self.assertIs(sg.select_sparse(scores, ends, blocks, out), out)
        args = native.forward.call_args.args
        self.assertTrue(all(a is b for a, b in zip(args[:4], (scores, ends, blocks, out))))
        self.assertEqual(args[4], 8)
        with patch.object(sg, "_module", side_effect=ImportError("native unavailable")):
            with self.assertRaisesRegex(ImportError, "native unavailable"):
                sg.select_dense(scores.float(), ends)
        native.forward.side_effect = RuntimeError("native failure")
        with patch.object(sg, "_module", return_value=native):
            with self.assertRaisesRegex(RuntimeError, "native failure"):
                sg.select_sparse(scores, ends, blocks)

    def test_admission_is_metadata_only_fail_closed_without_mutation(self):
        scores = torch.ones((2, 8192))
        ends = torch.tensor([8192, 7], dtype=torch.int32)
        out = torch.full((2, 512), -71, dtype=torch.int32)
        self.assertTrue(sg.is_supported_dense(scores, ends, out))
        bad = (scores[:, :-1], scores[:, ::2], scores.double(), scores._neg_view())
        with patch.object(sg, "_module", side_effect=AssertionError("unsupported launch")):
            for value in bad:
                self.assertIsNone(sg.select_dense(value, ends, out))
            self.assertIsNone(sg.select_dense(scores, ends.long(), out))
            self.assertIsNone(sg.select_dense(scores, ends, out, block_topk=1024))
            alias = scores.view(torch.int32)[:, :512]
            self.assertIsNone(sg.select_dense(scores, ends, alias))
            with torch.enable_grad():
                self.assertIsNone(sg.select_dense(scores, ends, out))
        self.assertTrue(torch.equal(out, torch.full_like(out, -71)))
        self.assertTrue(torch.equal(scores, torch.ones_like(scores)))


@unittest.skipUnless(os.environ.get("RTP_V41_TEST_LOCAL_WORKSPACE_DEVICE") == "1", "explicit PPU test")
class LocalPrefillSGTopKDevice(unittest.TestCase):
    def setUp(self):
        self.assertTrue(_quant.is_ppu())
        self.assertEqual(torch.cuda.get_device_capability(), (8, 9))
        self.assertEqual(torch.cuda.current_stream(), torch.cuda.default_stream())
        guard = torch.no_grad()
        guard.__enter__()
        self.addCleanup(guard.__exit__, None, None, None)
        torch.manual_seed(11212)

    def test_dense_finite_short_partial_padding_and_both_publication_widths(self):
        rows, width = 9, 32768
        bank = torch.full((rows + 2, width + 16), -77., device="cuda")
        scores = bank[1:-1, :width]
        scores.copy_(torch.randn_like(scores))
        ends = torch.tensor([0, 1, 7, 8, 511, 512, 513, 4097, width - 3], dtype=torch.int32, device="cuda")
        original = scores.clone()
        for block_k in (512, 2048):
            output_bank = torch.full((rows * 512 + 16,), -77, dtype=torch.int32, device="cuda")
            out = output_bank[8:-8].view(rows, 512)
            blocks = torch.full((rows, block_k), -81, dtype=torch.int32, device="cuda")
            got, publication = sg.select_dense(scores, ends, out, block_topk=block_k, blocks_out=blocks)
            self.assertIs(got, out)
            self.assertIs(publication, blocks)
            _assert_finite(self, got, scores, ends)
            self.assertTrue(bool((output_bank[:8] == -77).all()))
            self.assertTrue(bool((output_bank[-8:] == -77).all()))
            host_blocks = blocks.cpu()
            for row, end in enumerate(ends.cpu().tolist()):
                count = min(block_k, (end + 7) // 8)
                selected = host_blocks[row, :count]
                self.assertEqual(selected.unique().numel(), count)
                self.assertTrue(bool((selected >= 0).all()))
                self.assertTrue(bool((selected < (end + 7) // 8).all()))
                if count:
                    self.assertIn((end - 1) // 8, selected.tolist())
                self.assertTrue(bool((host_blocks[row, count:] == -1).all()))
        _bytes(self, scores, original)
        self.assertTrue(bool((bank[[0, -1]] == -77).all()))
        self.assertTrue(bool((bank[1:-1, width:] == -77).all()))

    def test_dense_nonfinite_matches_original_sg_bindings_without_old_filter(self):
        scores = torch.randn((8, 8192), device="cuda")
        scores[0].fill_(torch.nan)
        scores[1].fill_(torch.inf)
        scores[2].fill_(-torch.inf)
        scores[3, :513] = torch.nan
        scores[4, :511] = torch.inf
        scores[5, ::8] = torch.nan
        scores[6, :512] = -0.
        ends = torch.tensor([8192, 8192, 8192, 8191, 4097, 8192, 513, 7], dtype=torch.int32, device="cuda")
        before = scores.clone()
        host_scores = scores.cpu()
        host_ends = ends.cpu().tolist()
        cases = ("all_nan", "all_posinf", "all_neginf", "nan_prefix513_partial8191",
                 "posinf_prefix511_partial4097", "nan_each_block", "negative_zero_prefix512", "short7")
        record = []
        original_loader = sg._module
        for block_k in (512, 2048):
            previous_masks = {}
            # -1 is the actual RTP caller initialization. The other values are
            # test-only write canaries; never turn a native result into -1 or
            # scan NaNs in production. SG itself allocates blocks with empty.
            for sentinel in (-1, -1073741767, -1073741711):
                stage_records = {}

                def checked_blocks(label, blocks):
                    host = blocks.cpu().long()
                    safe = True
                    details = []
                    for row, end in enumerate(host_ends):
                        count = min(block_k, (end + 7) // 8)
                        prefix = host[row, :count]
                        valid = (prefix >= 0) & (prefix < (end + 7) // 8)
                        safe = safe and bool(valid.all())
                        details.append({"row": row, "case": cases[row], "expected_prefix": count,
                                        "prefix_valid": int(valid.sum()),
                                        "untouched_prefix": int((prefix == sentinel).sum()),
                                        "invalid_slots": torch.where(~valid)[0][:32].tolist(),
                                        "prefix_first16": prefix[:16].tolist()})
                    stage_records[label] = details
                    self._write("sg-topk-nonfinite-progress.json",
                                {"completed": record, "block_k": block_k, "sentinel": sentinel,
                                 "stage": stage_records})
                    self.assertTrue(safe, (block_k, sentinel, label, details))

                class CheckedNative:
                    def __init__(self, kind):
                        self.kind, self.module = kind, original_loader(kind)

                    def forward(proxy, *args):
                        proxy.module.forward(*args)
                        if proxy.kind == "select" and args[2] is None:
                            # Test-only fence prevents invalid block canaries
                            # reaching the unchanged raw-pointer candidate load.
                            checked_blocks("adapter", args[-1])

                actual = torch.full((8, 512), sentinel, dtype=torch.int32, device="cuda")
                blocks = torch.full((8, block_k), sentinel, dtype=torch.int32, device="cuda")
                reference, ref_blocks = torch.full_like(actual, sentinel), torch.full_like(blocks, sentinel)
                with patch.object(sg, "_module", side_effect=CheckedNative):
                    sg.select_dense(scores, ends, actual, block_topk=block_k, blocks_out=blocks)
                _direct_dense(scores, ends, block_k, out=reference, blocks_out=ref_blocks,
                              check_blocks=lambda value: checked_blocks("reference", value))
                # Candidate-stage isolation: original symbol consumes the exact
                # adapter block tensor. Repeat the original on its own blocks
                # too, separating coarse tie choice from token-stage stability.
                shared_blocks = torch.full_like(actual, sentinel)
                repeated = torch.full_like(actual, sentinel)
                _direct_dense(scores, ends, block_k, out=shared_blocks, use_blocks=blocks)
                _direct_dense(scores, ends, block_k, out=repeated, use_blocks=ref_blocks)
                results = {"adapter": actual.cpu().long(), "reference": reference.cpu().long(),
                           "reference_adapter_blocks": shared_blocks.cpu().long(),
                           "reference_repeat": repeated.cpu().long()}
                for row, end in enumerate(host_ends):
                    a, b = results["adapter"][row], results["reference"][row]
                    detail = {"block_k": block_k, "row": row, "case": cases[row], "sentinel": sentinel,
                              "end": end, "stage": {name: values[row] for name, values in stage_records.items()},
                              "outputs": {}}
                    for name, values in results.items():
                        value = values[row]
                        safe_ids = value[(value >= 0) & (value < end)]
                        detail["outputs"][name] = {
                            "valid_count": int((value >= 0).sum()),
                            "untouched_count": int((value == sentinel).sum()),
                            "untouched_slots": torch.where(value == sentinel)[0].tolist(),
                            "first32": value[:32].tolist(),
                            "invalid_not_sentinel": int(((value < -1) & (value != sentinel)).sum()),
                            "selected_score_counts": _multiset(host_scores[row, safe_ids])[:3],
                            "selected_first16_bits": [hex(int(v) & 0xffffffff) for v in
                                host_scores[row, safe_ids[:16]].contiguous().view(torch.int32).tolist()],
                        }
                    mismatches = torch.where(a != b)[0][:32]
                    detail["first_mismatch_slots"] = mismatches.tolist()
                    detail["actual_mismatch_values"] = a[mismatches].tolist()
                    detail["reference_mismatch_values"] = b[mismatches].tolist()
                    record.append(detail)
                    # Persist all raw write evidence before any assertion.
                    self._write("sg-topk-nonfinite-progress.json", record)
                    with self.subTest(block_k=block_k, row=row, case=cases[row], sentinel=sentinel):
                        if sentinel == -1:
                            # Original range/count/score-multiset assertions,
                            # now with the actual caller's initialized outputs.
                            self.assertTrue(bool(((a == -1) | ((a >= 0) & (a < end))).all()), detail)
                            self.assertEqual((a >= 0).sum().item(), (b >= 0).sum().item(), detail)
                            av = host_scores[row, a[a >= 0]]
                            bv = host_scores[row, b[b >= 0]]
                            self.assertEqual(_multiset(av), _multiset(bv), detail)
                        else:
                            self.assertTrue(torch.equal(a == sentinel, b == sentinel), detail)
                            for name, values in results.items():
                                key = name, row
                                mask = values[row] != sentinel
                                if key in previous_masks:
                                    self.assertTrue(torch.equal(mask, previous_masks[key]), detail)
                                previous_masks[key] = mask
        _bytes(self, scores, before)
        self._write("sg-topk-nonfinite.json", record)

    def test_sparse_native_semantics_mapping_partial_and_all_bf16_bits(self):
        rows, width = 8, 16384
        bank = torch.full((rows + 2, width + 16), 9, dtype=torch.bfloat16, device="cuda")
        scores = bank[1:-1, :width]
        scores.copy_(torch.randn_like(scores))
        ends = torch.tensor([0, 7, 512, 513, 16381, width, width, width], dtype=torch.int32, device="cuda")
        blocks = torch.arange(rows * (width // 8), dtype=torch.int32, device="cuda").reshape(rows, width // 8) + 101
        out = sg.select_sparse(scores, ends, blocks)
        _assert_finite(self, out, scores, ends, blocks)
        for chunk in range(4):
            # Every BF16 bit pattern appears once; the last two rows exercise
            # positive/negative NaN occupancy and Inf without RTP's sanitizer.
            patterns = torch.arange(chunk * width, (chunk + 1) * width, dtype=torch.int32)
            bits = patterns.to(torch.int16).view(torch.bfloat16)
            special = bits.repeat(4, 1).to("cuda")
            special[1, :513] = torch.nan
            special[2, :511] = torch.inf
            special[3, :512] = -torch.inf
            valid = torch.tensor([width, width, 513, 7], dtype=torch.int32, device="cuda")
            pages = blocks[:4].contiguous()
            actual = sg.select_sparse(special, valid, pages)
            direct = torch.empty_like(actual)
            sg._module("small").forward(special, valid, pages, direct, 8)
            # Compare mapped score multisets, allowing arbitrary tied IDs and
            # staging order. Short rows separately retain native exact IDs.
            for row in range(4):
                a, b = actual[row].cpu().long(), direct[row].cpu().long()
                self.assertEqual(int((a >= 0).sum()), int((b >= 0).sum()))
                page_row = pages[row].cpu().long()
                inverse = {int(page) * 8 + lane: col * 8 + lane
                           for col, page in enumerate(page_row.tolist()) for lane in range(8)}
                av = torch.tensor([inverse[int(i)] for i in a[a >= 0]], dtype=torch.long)
                bv = torch.tensor([inverse[int(i)] for i in b[b >= 0]], dtype=torch.long)
                source = special[row].cpu()
                self.assertEqual(_multiset(source[av]), _multiset(source[bv]))
                if int(valid[row]) <= 512:
                    self.assertTrue(torch.equal(a, b))
            if chunk == 0:
                self.assertEqual(int((actual[1] >= 0).sum()), 0)
                self.assertEqual(int((actual[2] >= 0).sum()), 512)
                self.assertTrue(bool((actual[3, 7:] == -1).all()))
        self.assertTrue(bool((bank[[0, -1]] == 9).all()))
        self.assertTrue(bool((bank[1:-1, width:] == 9).all()))

    def test_nondefault_stream_raw_views_alias_and_native_failure_boundaries(self):
        scores = torch.randn((2, 8192), device="cuda")
        ends = torch.full((2,), 8192, device="cuda", dtype=torch.int32)
        with torch.cuda.stream(torch.cuda.Stream()):
            with patch.object(sg, "_module", side_effect=AssertionError("nondefault launch")):
                self.assertIsNone(sg.select_dense(scores, ends))
        with patch.object(sg, "_module", side_effect=AssertionError("view launch")):
            self.assertIsNone(sg.select_dense(scores._neg_view(), ends))
            self.assertIsNone(sg.select_dense(scores, ends, scores.view(torch.int32)[:, :512]))
        small = torch.randn((2, 16384), device="cuda", dtype=torch.bfloat16)
        pages = torch.arange(4096, dtype=torch.int32, device="cuda").view(2, 2048)
        native = Mock(forward=Mock(side_effect=RuntimeError("original native error")))
        with patch.object(sg, "_module", return_value=native):
            with self.assertRaisesRegex(RuntimeError, "original native error"):
                sg.select_sparse(small, ends, pages)

    def _write(self, name, value):
        # Runtime test evidence, never project source.
        directory = Path(os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR") or tempfile.gettempdir())
        directory.mkdir(parents=True, exist_ok=True)
        (directory / name).write_text(json.dumps(value, indent=2))
        return directory

    def _profile(self, operation, path, *, sparse=False):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as profile:
            operation()
            torch.cuda.synchronize()
        profile.export_chrome_trace(str(path))
        events = json.loads(path.read_text())["traceEvents"]
        kernels = [event.get("name", "") for event in events if event.get("cat") == "kernel"]
        wanted = (("topk_bf16_small_kernel",) if sparse else
                  ("amax8_varlen_kernel", "topk_ragged_kernel", "topk_candidate_kernel"))
        for name in wanted:
            self.assertEqual(sum(name in kernel for kernel in kernels), 1, (name, kernels))
        for forbidden in ("canonicalize", "finish_tokens", "sanitize", "checked", "radix_retry"):
            self.assertFalse(any(forbidden in kernel.lower() for kernel in kernels), kernels)
        self.assertFalse(any("DtoH" in event.get("name", "") for event in events))
        self.assertFalse(any(event.get("name") == "aten::topk" for event in events))

    def test_actual_shapes_balanced_complete_selection_timing_and_profiles(self):
        from rtp_llm.models_py.modules.dsv41.ppu import prefill_token_topk as old

        reports = []
        for rows, width, block_k in ((4096, 65536, 512), (140, 131072, 2048)):
            # Same BF16-landed finite values, widened exactly for SG. Production
            # SG FP32 scoring is a separate caller precision/semantics contract.
            bf16 = torch.randn((rows, width), dtype=torch.bfloat16, device="cuda")
            fp32 = bf16.float()
            ends = (torch.arange(rows, device="cuda", dtype=torch.int32) + width - rows).clamp(0, width)
            visible = ends.long()
            out = torch.empty((rows, 512), dtype=torch.int32, device="cuda")
            blocks = torch.empty((rows, block_k), dtype=torch.int32, device="cuda")
            call_new = lambda: sg.select_dense(fp32, ends, out, block_topk=block_k, blocks_out=blocks)
            call_old = lambda: old.try_select_tokens(bf16, visible, disposable=True)
            env = {old.ENV: "1", "RTP_V41_PPU_LOCAL_PREFILL_HIERARCHICAL_TOPK": "0"}
            with patch.dict(os.environ, env):
                for _ in range(3):
                    self.assertIsNotNone(call_old())
                    self.assertIsNotNone(call_new())
                torch.cuda.synchronize()
                samples = {"legacy_bf16": [], "sg_fp32": []}
                for repeat in range(16):
                    arms = (("legacy_bf16", call_old), ("sg_fp32", call_new))
                    for name, operation in (arms if repeat % 2 == 0 else arms[::-1]):
                        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        wall = time.perf_counter()
                        begin.record()
                        result = operation()
                        end.record()
                        submit_ms = (time.perf_counter() - wall) * 1000
                        end.synchronize()
                        samples[name].append({"gpu_ms": begin.elapsed_time(end),
                                              "host_submit_ms": submit_ms,
                                              "host_complete_ms": (time.perf_counter() - wall) * 1000})
                        self.assertIsNotNone(result)
                # Representative rows use an independent finite score oracle;
                # copying/sorting the entire one-GiB score tile is unnecessary.
                ids = torch.tensor([0, rows // 2, rows - 1], device="cuda")
                _assert_finite(self, out[ids], fp32[ids], ends[ids])
                directory = self._write("sg-topk-timing-progress.json", reports)
                for name, operation in (("legacy_bf16", call_old), ("sg_fp32", call_new)):
                    path = directory / f"sg-topk-{rows}-{width}-{name}.json"
                    if name == "sg_fp32":
                        self._profile(operation, path)
                    else:
                        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                               torch.profiler.ProfilerActivity.CUDA]) as profile:
                            operation()
                            torch.cuda.synchronize()
                        profile.export_chrome_trace(str(path))
                reports.append({"rows": rows, "width": width, "block_k": block_k,
                                "samples": samples,
                                "median_gpu_ms": {k: statistics.median(x["gpu_ms"] for x in v)
                                                  for k, v in samples.items()}})
            del fp32, bf16
        # Actual late ABI, native BF16 small with fused page remap.
        sparse = torch.randn((140, 16384), device="cuda", dtype=torch.bfloat16)
        lens = torch.full((140,), 16381, device="cuda", dtype=torch.int32)
        pages = torch.arange(140 * 2048, device="cuda", dtype=torch.int32).view(140, 2048)
        got = sg.select_sparse(sparse, lens, pages)
        _assert_finite(self, got[[0, 70, 139]], sparse[[0, 70, 139]], lens[[0, 70, 139]], pages[[0, 70, 139]])
        directory = self._write("sg-topk-timings.json", reports)
        self._profile(lambda: sg.select_sparse(sparse, lens, pages),
                      directory / "sg-topk-140-16384-sparse.json", sparse=True)


if __name__ == "__main__":
    unittest.main()
