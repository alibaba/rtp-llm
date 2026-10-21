"""Exact first, timing second: isolated two-query reuse vs selected producer.

Existing explicit-launch helper and independent double-dot oracle are reused;
the old selected kernel is never redirected to the candidate. No tolerance
relaxation, no silent candidate rejection classified as PASS.
"""

import json
import os
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

import torch
import triton

from rtp_llm.models_py.modules.dsv41.fp8 import _v41_prefill_topk as prefill_topk
from rtp_llm.models_py.modules.dsv41.fp8 import _v41_sm90_indexer as production_score
from rtp_llm.models_py.modules.dsv41.fp8 import (
    _v41_sm90_indexer_multiquery as multiquery_score,
)
from rtp_llm.models_py.modules.dsv41.fp8 import (
    _v41_sparse_prefill_indexer as sparse_score,
)
from rtp_llm.models_py.modules.dsv41.fp8._v41_prefill_indexer import _fp4_rows_torch
from rtp_llm.models_py.modules.dsv41.fp8._v41_sm90_indexer_multiquery import (
    _score_stage_probe,
    _score_two_queries_concat_rejected_kernel,
    dense_score_two_queries,
)
from rtp_llm.models_py.modules.dsv41.fp8.test.test_v41_sm90_indexer_tuning import (
    independent_score,
    launch_existing,
)


def _inputs(rows, width, seed=4601):
    torch.manual_seed(seed + rows)
    q0 = torch.randn(rows, 32, 128, device="cuda", dtype=torch.bfloat16) * 0.2
    k0 = torch.randn(width, 128, device="cuda", dtype=torch.bfloat16) * 0.2
    q, qs, qr = _fp4_rows_torch(q0)
    keys, ks, kr = _fp4_rows_torch(k0)
    q, qs = q.view(rows, 32, 64), qs.view(rows, 32)
    weights = (torch.randn(rows, 32, device="cuda") / 64).bfloat16()
    weights[rows // 2].zero_()  # Exact score ties; selection ORDER is compared.
    return q, qs, keys, ks, weights, qr.view(rows, 32, 128), kr


def _bounds(rows, width, phase):
    offsets = torch.arange(rows, dtype=torch.int32, device="cuda")
    if phase == "early":
        ends = offsets + 1
    elif phase == "late":
        ends = width - rows + 1 + offsets
    elif phase == "cp_zigzag_ratio2":
        # Adjacent queries deliberately alternate opposite CP halves. Positions
        # belong to the uncompressed stream; ratio2 conversion precedes scoring.
        positions = torch.where(offsets % 2 == 0, offsets, 2 * width - 1 - offsets)
        ends = ((positions + 1) // 2).clamp(0, width)
    elif phase == "all_masked":
        ends = torch.zeros_like(offsets)
    else:
        raise ValueError(phase)
    starts = torch.zeros_like(ends)
    starts[1::7] = 17
    starts[2::7] = 127
    if rows > 3:
        ends[3::13] = 0
        starts[4::17] = ends[4::17]
    return starts, ends


def _old(q, qs, keys, ks, weights, starts, ends, out):
    # Explicit CURRENT SELECTED launch, not the older BLOCK64 baseline for
    # large prefill; no credit for the BLOCK128 win already shipped in round2.
    block = 128 if q.shape[0] >= 512 and keys.shape[0] >= 1024 else 64
    return launch_existing(q, qs, keys, ks, weights, starts, ends, block, 4, out)


def _profile_one(fn, path, expected_kernel):
    """One arm, one independently captured invocation, outside event timing."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
    ) as profile:
        with torch.profiler.record_function(expected_kernel):
            fn()
        torch.cuda.synchronize()
    profile.export_chrome_trace(str(path))
    # Torch profile.events() also exposes a GPU user-annotation with the same
    # name as this producer and reported two events for one actual launch in
    # bench05. Chrome categories distinguish real kernel work unambiguously.
    trace_events = json.loads(path.read_text())["traceEvents"]
    kernels = [event for event in trace_events if event.get("cat") == "kernel"]
    copies = [
        event
        for event in trace_events
        if event.get("cat") in ("gpu_memcpy", "gpu_memset")
    ]
    # One producer launch per arm: ensure the intended kernel actually ran,
    # and reject extra producer/copy work hidden inside event span.
    assert len(kernels) == 1, [(event["name"], event["dur"]) for event in kernels]
    assert expected_kernel in kernels[0]["name"], kernels[0]["name"]
    return dict(
        iterations=1,
        kernel_count=1,
        kernel_sum_us=sum(event["dur"] for event in kernels),
        gpu_events=[
            dict(
                name=event["name"],
                count=1,
                us=event["dur"],
                category=event["cat"],
                stream=event.get("args", {}).get("stream"),
            )
            for event in kernels
        ],
        gpu_copy_count=len(copies),
        gpu_copy_us=sum(event["dur"] for event in copies),
        gpu_copies=[
            dict(name=event["name"], us=event["dur"], category=event["cat"])
            for event in copies
        ],
        trace=str(path),
    )


class SM90IndexerMultiqueryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (
            9,
            0,
        ):
            raise RuntimeError(
                "Dedicated multiquery candidate target requires SM90; skip is not PASS"
            )

    def _check(self, inputs, starts, ends, block):
        q, qs, keys, ks, weights, qr, kr = inputs
        rows, width = q.shape[0], keys.shape[0]
        baseline = torch.empty((rows, width), device="cuda", dtype=torch.float32)
        out = torch.empty_like(baseline)
        _old(q, qs, keys, ks, weights, starts, ends, baseline)
        actual = dense_score_two_queries(
            q, qs, keys, ks, weights, starts, ends, block=block, out=out
        )
        torch.testing.assert_close(actual, baseline, rtol=0, atol=0)
        # FP32 outputs contain rounded BF16 finite values/-inf: compare bits as
        # well to retain signed-zero/tie behavior, not just approximate scores.
        torch.testing.assert_close(
            actual.view(torch.int32), baseline.view(torch.int32), rtol=0, atol=0
        )
        # The scorer accepts int64 bounds; native TopK's explicitly prepared
        # bounds contract is int32. Adapt BOTH arms only at the selector API.
        # Fixture bounds are within int32, so this does not clamp/change values
        # or remove the above int64 scorer/bitwise checks.
        topk_bounds = (starts.to(torch.int32), ends.to(torch.int32))
        bt = prefill_topk.try_select_tokens(baseline, ends, bounds=topk_bounds)
        at = prefill_topk.try_select_tokens(actual, ends, bounds=topk_bounds)
        self.assertIsNotNone(bt)
        self.assertIsNotNone(at)
        torch.testing.assert_close(at, bt, rtol=0, atol=0)
        columns = torch.linspace(0, width - 1, min(137, width), device="cuda").long()
        for row in sorted({0, rows // 2, rows - 1}):
            oracle = independent_score(
                qr[row].bfloat16(), kr[columns].bfloat16(), weights[row]
            )
            oracle.masked_fill_(
                (columns < starts[row]) | (columns >= ends[row]), -torch.inf
            )
            # Same controlled dyadic range as the previous launch-tuning oracle.
            torch.testing.assert_close(actual[row, columns], oracle, rtol=0, atol=0)
        return baseline, out

    def test_small_fixture_exact(self):
        for rows in (1, 2, 3, 17, 33):
            inputs = _inputs(rows, 1031)
            for phase in ("early", "late", "cp_zigzag_ratio2", "all_masked"):
                starts, ends = _bounds(rows, 1031, phase)
                for block in (64, 128):
                    with self.subTest(rows=rows, phase=phase, block=block):
                        self._check(inputs, starts, ends, block)

    def test_actual_prefill_exact(self):
        for rows, width in ((512, 32768), (1024, 32768), (513, 32769), (512, 131072)):
            inputs = _inputs(rows, width)
            for phase in ("early", "late", "cp_zigzag_ratio2", "all_masked"):
                starts, ends = _bounds(rows, width, phase)
                for block in (64, 128):
                    with self.subTest(rows=rows, width=width, phase=phase, block=block):
                        self._check(inputs, starts, ends, block)

    def test_production_payload_bytes_and_bound_dtypes(self):
        inputs = _inputs(3, 1031)
        self.assertEqual(inputs[0].dtype, torch.int8)
        self.assertEqual(inputs[2].dtype, torch.int8)
        starts, ends = _bounds(3, 1031, "late")
        for payload_dtype in (torch.int8, torch.uint8):
            # Bit reinterpretation only. Never numerical cast or requantize
            # signed bytes to make a synthetic fixture satisfy the wrapper.
            variant = (
                inputs[0].view(payload_dtype),
                inputs[1],
                inputs[2].view(payload_dtype),
                inputs[3],
                inputs[4].float(),
                inputs[5],
                inputs[6],
            )
            for bounds_dtype in (torch.int32, torch.int64):
                self._check(
                    variant, starts.to(bounds_dtype), ends.to(bounds_dtype), 128
                )
        with self.assertRaisesRegex(ValueError, "scales"):
            dense_score_two_queries(
                inputs[0],
                inputs[1].long(),
                inputs[2],
                inputs[3],
                inputs[4],
                starts,
                ends,
            )

    def test_current_stream_graph_replay(self):
        inputs = _inputs(3, 1031)
        q, qs, keys, ks, weights, _, _ = inputs
        starts, ends = _bounds(3, 1031, "late")
        out = torch.empty((3, 1031), device="cuda", dtype=torch.float32)
        baseline = torch.empty_like(out)
        eager = torch.empty_like(out)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        fn = lambda: dense_score_two_queries(
            q, qs, keys, ks, weights, starts, ends, out=out
        )
        with torch.cuda.stream(stream):
            for _ in range(3):
                fn()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = fn()
        pointer = captured.data_ptr()
        for iteration, phase in enumerate(
            ("early", "all_masked", "cp_zigzag_ratio2", "late")
        ):
            # Same allocations, changing ALL live inputs. Roll payload/scales
            # together by legal row dimensions to retain real FP4 bytes. A
            # capture must not freeze causal bounds, Q/K, scales or weights.
            with torch.cuda.stream(stream):
                new_starts, new_ends = _bounds(3, 1031, phase)
                starts.copy_(new_starts)
                ends.copy_(new_ends)
                q.copy_(q.roll(1, dims=0))
                qs.copy_(qs.roll(1, dims=0))
                keys.copy_(keys.roll(17, dims=0))
                ks.copy_(ks.roll(17, dims=0))
                if iteration == 1:
                    weights.zero_()
                elif iteration == 2:
                    weights.fill_(1 / 32)
                _old(q, qs, keys, ks, weights, starts, ends, baseline)
                dense_score_two_queries(
                    q, qs, keys, ks, weights, starts, ends, out=eager
                )
                graph.replay()
            stream.synchronize()
            self.assertEqual(captured.data_ptr(), pointer)
            torch.testing.assert_close(
                eager.view(torch.int32), baseline.view(torch.int32), rtol=0, atol=0
            )
            torch.testing.assert_close(
                captured.view(torch.int32), baseline.view(torch.int32), rtol=0, atol=0
            )

    def test_malformed_layout_guards_and_empty(self):
        q, qs, keys, ks, weights, _, _ = _inputs(3, 1031)
        starts, ends = _bounds(3, 1031, "late")
        args = [q, qs, keys, ks, weights, starts, ends]
        malformed = [
            (0, q.float()),
            (1, qs.long()),
            (3, ks.long()),
            (4, weights.long()),
            (5, starts.float()),
            (6, ends.float()),
            (0, q.repeat_interleave(2, dim=-1)[..., ::2]),
            (0, q[:, :16, :].contiguous()),
            (0, q.cpu()),
        ]
        for position, bad in malformed:
            changed = args.copy()
            changed[position] = bad
            with self.subTest(
                position=position, dtype=bad.dtype, shape=tuple(bad.shape)
            ):
                with self.assertRaises(ValueError):
                    dense_score_two_queries(*changed)
        with self.assertRaises(ValueError):
            dense_score_two_queries(*args, block=256)
        with self.assertRaises(ValueError):
            dense_score_two_queries(
                *args, out=torch.empty((3, 1031), device="cuda", dtype=torch.bfloat16)
            )
        # Zero-stride metadata view needs no giant allocation; the indexing
        # bound must reject before contiguity or shape checks/any launch.
        huge_q = q.as_strided((2**21, 32, 64), (0, 64, 1))
        with self.assertRaisesRegex(ValueError, "rows \\* capacity"):
            dense_score_two_queries(huge_q, qs, keys, ks, weights, starts, ends)
        empty_rows = dense_score_two_queries(
            q[:0], qs[:0], keys, ks, weights[:0], starts[:0], ends[:0]
        )
        self.assertEqual(empty_rows.shape, (0, 1031))
        empty_keys = dense_score_two_queries(
            q, qs, keys[:0], ks[:0], weights, starts, ends
        )
        self.assertEqual(empty_keys.shape, (3, 0))

    def test_poisoned_output_bounds_and_masked_scales(self):
        q, qs, keys, ks, weights, _, _ = _inputs(3, 1031)
        # These bounds remain int64 all the way into BOTH score kernels. No
        # TopK conversion here: intentionally extreme values are not its API.
        cases = [
            ([-7, 18, 1000], [-1, 3, 2**63 - 1]),
            ([-(2**63), 2**63 - 1, 0], [2**63 - 1, 2**63 - 1, -(2**63)]),
            ([0, 513, 1031], [1, 1031, 1031]),
        ]
        for lo, hi in cases:
            starts = torch.tensor(lo, device="cuda", dtype=torch.int64)
            ends = torch.tensor(hi, device="cuda", dtype=torch.int64)
            old = torch.full((3, 1031), torch.nan, device="cuda")
            _old(q, qs, keys, ks, weights, starts, ends, old)
            for block in (64, 128):
                out = torch.full_like(old, torch.nan)
                dense_score_two_queries(
                    q, qs, keys, ks, weights, starts, ends, block=block, out=out
                )
                self.assertFalse(torch.isnan(out).any().item())
                torch.testing.assert_close(
                    out.view(torch.int32), old.view(torch.int32), rtol=0, atol=0
                )

        self._check_poison_scales()

    def test_production_dispatch_large_exact(self):
        # The SAME previously failing 128K fixture, through the actual root-
        # owned dispatcher. Expected scores still use explicit old launch.
        q, qs, keys, ks, weights, _, _ = _inputs(512, 131072)
        starts, ends = _bounds(512, 131072, "late")
        reference = torch.empty((512, 131072), device="cuda", dtype=torch.float32)
        _old(q, qs, keys, ks, weights, starts, ends, reference)
        reference_topk = prefill_topk.try_select_tokens(
            reference, ends, bounds=(starts, ends)
        )
        self.assertIsNotNone(reference_topk)
        for enabled in ("0", "1", None):
            # None exercises the actual promoted default, not an explicit "1".
            # patch.dict restores the caller's full environment after deletion.
            updates = (
                {} if enabled is None else {"DSV41_SM90_MULTIQUERY_INDEXER": enabled}
            )
            with mock.patch.dict(os.environ, updates):
                if enabled is None:
                    os.environ.pop("DSV41_SM90_MULTIQUERY_INDEXER", None)
                with mock.patch.object(
                    multiquery_score,
                    "dense_score_two_queries",
                    wraps=dense_score_two_queries,
                ) as spy:
                    actual = production_score.dense_score(
                        q, qs, keys, ks, weights, starts, ends
                    )
                    self.assertEqual(spy.call_count, 0 if enabled == "0" else 1)
                    if enabled != "0":
                        self.assertEqual(spy.call_args.kwargs["block"], 128)
                        self.assertEqual(
                            spy.call_args.kwargs["out"].data_ptr(), actual.data_ptr()
                        )
                    torch.testing.assert_close(
                        actual.view(torch.int32),
                        reference.view(torch.int32),
                        rtol=0,
                        atol=0,
                    )
                    actual_topk = prefill_topk.try_select_tokens(
                        actual, ends, bounds=(starts, ends)
                    )
                    self.assertIsNotNone(actual_topk)
                    torch.testing.assert_close(
                        actual_topk, reference_topk, rtol=0, atol=0
                    )
                    del actual, actual_topk

    def test_production_dispatch_fallbacks(self):
        q, qs, keys, ks, weights, _, _ = _inputs(512, 32768)
        starts = torch.zeros(512, device="cuda", dtype=torch.int32)
        ends = torch.full_like(starts, 17)
        # Each excluded route otherwise meets the tested large shape threshold
        # where practical, so a spy detects accidental backend redirection.
        q64, qs64, weights64 = q.repeat(1, 2, 1), qs.repeat(1, 2), weights.repeat(1, 2)
        pool = torch.zeros((2, 64, 68), device="cuda", dtype=torch.uint8)
        table = torch.zeros((1, 512), device="cuda", dtype=torch.int32)
        candidates = torch.tensor(
            [0, 1, -1, 4097], device="cuda", dtype=torch.int32
        ).repeat(512, 1)
        plan = sparse_score.prepare_plan(candidates, ends, 32768, sm90=True)
        self.assertIsNotNone(plan)
        routes = {
            "small_rows": lambda: production_score.dense_score(
                q[:17], qs[:17], keys, ks, weights[:17], starts[:17], ends[:17]
            ),
            "small_capacity": lambda: production_score.dense_score(
                q, qs, keys[:1031], ks[:1031], weights, starts, ends
            ),
            "h64": lambda: production_score.dense_score(
                q64, qs64, keys, ks, weights64, starts, ends
            ),
            "paged": lambda: production_score.paged_score(
                q.reshape(1, 512, 32, 64), qs, pool, weights, table, ends, 32768, 64
            ),
            "sparse": lambda: production_score.candidate_score(
                q, qs, keys, ks, weights, plan
            ),
        }
        for name, run in routes.items():
            with self.subTest(route=name):
                with mock.patch.dict(
                    os.environ, {"DSV41_SM90_MULTIQUERY_INDEXER": "0"}
                ):
                    reference = run()
                with mock.patch.dict(
                    os.environ, {"DSV41_SM90_MULTIQUERY_INDEXER": "1"}
                ):
                    with mock.patch.object(
                        multiquery_score,
                        "dense_score_two_queries",
                        side_effect=AssertionError(
                            "unsupported route reached candidate"
                        ),
                    ) as spy:
                        actual = run()
                        spy.assert_not_called()
                torch.testing.assert_close(
                    actual.view(torch.int32),
                    reference.view(torch.int32),
                    rtol=0,
                    atol=0,
                )

    def _check_poison_scales(self):
        q, qs, keys, ks, weights, _, _ = _inputs(3, 1031)
        # Fully masked Q with NaN scale must not contaminate its live partner.
        starts = torch.tensor([0, 17, 1031], device="cuda", dtype=torch.int32)
        ends = torch.tensor([0, 1031, 1031], device="cuda", dtype=torch.int32)
        poison_qs = qs.clone()
        poison_qs[0].fill_(-1)  # Packed 0xFFFFFFFF = four UE8M0 NaN codes.
        # Also poison a K gap loaded by the union but invisible to BOTH rows.
        gap_starts = torch.tensor([0, 29, 1031], device="cuda", dtype=torch.int32)
        gap_ends = torch.tensor([7, 1031, 1031], device="cuda", dtype=torch.int32)
        poison_ks = ks.clone()
        poison_ks[7:29].fill_(-1)
        for test_qs, test_ks, lo, hi in (
            (poison_qs, ks, starts, ends),
            (qs, poison_ks, gap_starts, gap_ends),
        ):
            old = torch.full((3, 1031), torch.nan, device="cuda")
            _old(q, test_qs, keys, test_ks, weights, lo, hi, old)
            for block in (64, 128):
                out = torch.full_like(old, torch.nan)
                dense_score_two_queries(
                    q, test_qs, keys, test_ks, weights, lo, hi, block=block, out=out
                )
                self.assertFalse(torch.isnan(out).any().item())
                torch.testing.assert_close(
                    out.view(torch.int32), old.view(torch.int32), rtol=0, atol=0
                )

    @unittest.skipUnless(
        os.environ.get("DSV41_SM90_MULTIQUERY_DIAG") == "1",
        "explicit same-fixture numerical diagnosis",
    )
    def test_same_fixture_stage_diagnosis(self):
        # SAME seed, generation order, complete shape, bounds and explicit
        # selected baseline launch as correctness02. Never shrink Q/K before
        # generating inputs or accidentally switch old BLOCK128 to BLOCK64.
        rows, width = 512, 131072
        inputs = _inputs(rows, width)
        q, qs, keys, ks, weights, qr, kr = inputs
        starts, ends = _bounds(rows, width, "late")
        baseline = torch.empty((rows, width), device="cuda", dtype=torch.float32)
        candidate = torch.empty_like(baseline)
        _old(q, qs, keys, ks, weights, starts, ends, baseline)
        root = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
        records = []
        for block in (64, 128):
            # Preserve the REJECTED concat failure independently of the fixed
            # public candidate. This is historical diagnosis, not acceptance.
            _score_two_queries_concat_rejected_kernel[
                (triton.cdiv(rows, 2), triton.cdiv(width, block))
            ](
                q,
                qs,
                keys,
                ks,
                weights,
                starts,
                ends,
                candidate,
                ROWS=rows,
                CAPACITY=width,
                BLOCK=block,
                num_warps=4,
            )
            differences = (
                (candidate.view(torch.int32) != baseline.view(torch.int32))
                .nonzero()
                .cpu()
                .tolist()
            )
            print(
                "MULTIQUERY_FULL_SHAPE_DIFF "
                + json.dumps(
                    dict(block=block, count=len(differences), indices=differences[:32])
                ),
                flush=True,
            )
            for row, col in differences[:8]:
                stage = {}
                for name, query_rows, row_base, tile in (
                    ("old", 1, row, 128),
                    ("candidate", 2, row // 2 * 2, block),
                ):
                    col_base = col // tile * tile
                    raw = torch.empty(
                        (query_rows * 32, tile), device="cuda", dtype=torch.float32
                    )
                    dot, weighted = torch.empty_like(raw), torch.empty_like(raw)
                    sums = torch.empty(
                        (query_rows, tile), device="cuda", dtype=torch.float32
                    )
                    score = torch.empty_like(sums)
                    compiled = _score_stage_probe[(1,)](
                        q,
                        qs,
                        keys,
                        ks,
                        weights,
                        raw,
                        dot,
                        weighted,
                        sums,
                        score,
                        ROW_BASE=row_base,
                        COL_BASE=col_base,
                        QUERY_ROWS=query_rows,
                        BLOCK=tile,
                        num_warps=4,
                    )
                    # Persist compiled evidence: debug stores can affect layout.
                    for extension in ("ttgir", "ptx"):
                        (
                            root / f"stage_{block}_{row}_{col}_{name}.{extension}"
                        ).write_text(compiled.asm[extension])
                    head_slice = slice((row - row_base) * 32, (row - row_base + 1) * 32)
                    offset = col - col_base
                    stage[name] = dict(
                        raw_dot=raw[head_slice, offset].cpu().tolist(),
                        dot_bf16=dot[head_slice, offset].cpu().tolist(),
                        weighted_bf16=weighted[head_slice, offset].cpu().tolist(),
                        sum_fp32=sums[row - row_base, offset].item(),
                        score=score[row - row_base, offset].item(),
                    )
                oracle_raw = (qr[row].double() * kr[col].double()[None]).sum(-1)
                oracle_dot = oracle_raw.bfloat16().float()
                oracle_weighted = (
                    (oracle_dot.relu() * weights[row].float()).bfloat16().float()
                )
                record = dict(
                    row=row,
                    col=col,
                    block=block,
                    full_baseline=baseline[row, col].item(),
                    full_candidate=candidate[row, col].item(),
                    stage=stage,
                    oracle=dict(
                        raw_dot_fp64=oracle_raw.cpu().tolist(),
                        dot_bf16=oracle_dot.cpu().tolist(),
                        weighted_bf16=oracle_weighted.cpu().tolist(),
                        sum_fp64=oracle_weighted.double().sum().item(),
                        final_fp64_sum_bf16=oracle_weighted.double()
                        .sum()
                        .bfloat16()
                        .float()
                        .item(),
                    ),
                    scope="diagnostic only; not acceptance or performance",
                )
                records.append(record)
                print("MULTIQUERY_STAGE_DIAGNOSTIC " + json.dumps(record), flush=True)
                (root / "round6_indexer_stage_diagnosis.json").write_text(
                    json.dumps(records, indent=2)
                )
                # Instrumented stages are valid evidence only if they reproduce
                # each arm's actual full-fixture failing score independently.
                self.assertEqual(stage["old"]["score"], record["full_baseline"])
                self.assertEqual(stage["candidate"]["score"], record["full_candidate"])

    @unittest.skipUnless(
        os.environ.get("DSV41_SM90_MULTIQUERY_BENCH") == "1",
        "explicit component benchmark opt-in",
    )
    def test_benchmark_exact_before_timing(self):
        output_dir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
        if not output_dir:
            self.fail("Benchmark requires persistent TEST_UNDECLARED_OUTPUTS_DIR")
        root = Path(output_dir)
        results = []
        # First bounded producer campaign, not a 36-point sweep. M1/small
        # behavior remains strictly tested separately, without claiming a gain.
        cases = (
            (512, 131072, "late"),
            (1024, 32768, "late"),
            (1024, 32768, "cp_zigzag_ratio2"),
        )
        for rows, width, phase in cases:
            inputs = _inputs(rows, width)
            q, qs, keys, ks, weights, _, _ = inputs
            for phase in (phase,):
                starts, ends = _bounds(rows, width, phase)
                for block in (64, 128):
                    # Every shape/launch has strict score + TopK + independent
                    # checks before any timing. Numerical failures fail target.
                    baseline, candidate = self._check(inputs, starts, ends, block)
                    old = lambda: _old(q, qs, keys, ks, weights, starts, ends, baseline)
                    new = lambda: dense_score_two_queries(
                        q,
                        qs,
                        keys,
                        ks,
                        weights,
                        starts,
                        ends,
                        block=block,
                        out=candidate,
                    )
                    for _ in range(3):
                        old()
                        new()
                    torch.cuda.synchronize()
                    timing = {"baseline_ms": [], "candidate_ms": []}
                    profiles = {"baseline": [], "candidate": []}
                    for round_id in range(3):
                        order = (("baseline_ms", old), ("candidate_ms", new))
                        if round_id % 2:
                            order = tuple(reversed(order))
                        for name, fn in order:
                            timing[name].append(
                                triton.testing.do_bench(fn, warmup=25, rep=100)
                            )
                        # Independent profiles, never wrapped around clean
                        # timing events; verify one expected launch per arm.
                        for name, fn, expected in (
                            ("baseline", old, "_score_kernel"),
                            ("candidate", new, "_score_two_queries_kernel"),
                        ):
                            path = (
                                root
                                / f"round6_score_{rows}_{width}_{phase}_bk{block}_{name}_{round_id}.trace.json"
                            )
                            profiles[name].append(_profile_one(fn, path, expected))
                    record = dict(
                        rows=rows,
                        keys=width,
                        heads=32,
                        dim=128,
                        phase=phase,
                        query_tile=2,
                        block=block,
                        warps=4,
                        status="FALLBACK_M1" if rows == 1 else "EXACT_PASS",
                        scope="score producer only; no full model inference",
                        timing_note="event spans and separately sampled profiler kernel sums must not be subtracted to infer CPU time",
                        profiles=profiles,
                        **timing,
                    )
                    results.append(record)
                    print("SM90_MULTIQUERY_BENCH " + json.dumps(record), flush=True)
                    # Preserve each completed point if a later strict case fails.
                    (root / "round6_indexer_multiquery.json").write_text(
                        json.dumps(results, indent=2)
                    )


if __name__ == "__main__":
    unittest.main()
