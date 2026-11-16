"""Test-only launch tuning of the existing SM90 FP4 indexer score kernel.

No new attention/GEMM implementation, quantization or rounding changes.
Default tests strictly compare the selected production launch to an explicit
old BLOCK64/warps4 launch, including native TopK output order. Full exploration
is opt-in diagnostics, not an acceptance test: bad candidates are REJECTed.
"""

import json
import os
import unittest

import torch
import triton

from rtp_llm.models_py.modules.dsv41.fp8 import _v41_prefill_topk as prefill_topk
from rtp_llm.models_py.modules.dsv41.fp8._v41_prefill_indexer import _fp4_rows_torch
from rtp_llm.models_py.modules.dsv41.fp8._v41_sm90_indexer import (
    _score_kernel,
    dense_score,
)


def launch_existing(q, qs, keys, ks, weights_bf16, starts, ends, block, warps, out):
    """The production kernel, only its tile and launch geometry vary."""
    _score_kernel[(q.shape[0], triton.cdiv(keys.shape[0], block))](
        q,
        qs,
        keys,
        ks,
        weights_bf16,
        starts,
        starts,
        ends,
        out,
        CAPACITY=keys.shape[0],
        HEADS=32,
        QUERY_LEN=1,
        PAGED=False,
        PHYSICAL=1,
        LOGICAL=1,
        PAGE_STRIDE=1,
        TABLE_STRIDE=1,
        TABLE_COL_STRIDE=1,
        BLOCK=block,
        num_warps=warps,
    )
    return out


def independent_score(q_bf16, keys_bf16, weights_bf16):
    """Independent double dot, explicit original BF16 rounding boundaries."""
    dot = (q_bf16.double() @ keys_bf16.double().T).bfloat16().float()
    weighted = (dot.relu() * weights_bf16.float()[:, None]).bfloat16()
    return weighted.float().sum(0).bfloat16().float()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class SM90IndexerLaunchTuningTest(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() != (9, 0):
            self.skipTest("requires SM90")

    def test_selected_dense_producer_strict_regression(self):
        # Includes an unaligned final K tile and M boundary, nonzero starts,
        # empty rows, causal early/late rows, and exact score ties. This test
        # must FAIL on any mismatch; it never classifies failures as diagnostics.
        for rows, width in ((512, 32768), (1024, 32768), (513, 32769), (512, 1024)):
            torch.manual_seed(4410 + rows)
            q_input = (
                torch.randn(rows, 32, 128, device="cuda", dtype=torch.bfloat16) * 0.2
            )
            k_input = torch.randn(width, 128, device="cuda", dtype=torch.bfloat16) * 0.2
            q, qs, _ = _fp4_rows_torch(q_input)
            keys, ks, _ = _fp4_rows_torch(k_input)
            q, qs = q.view(rows, 32, 64), qs.view(rows, 32)
            weights = (torch.randn(rows, 32, device="cuda") / 64).bfloat16()
            weights[rows // 2].zero_()
            for phase in ("early_causal", "late_causal", "all_masked"):
                with self.subTest(rows=rows, width=width, phase=phase):
                    first = 1 if phase == "early_causal" else width - rows + 1
                    ends = torch.arange(
                        first, first + rows, device="cuda", dtype=torch.int32
                    )
                    if phase == "all_masked":
                        ends.zero_()
                    starts = torch.zeros_like(ends)
                    starts[1::7] = 17
                    starts[2::7] = 127
                    ends[::13] = 0
                    starts[3::17] = ends[3::17]  # Explicit fully masked rows.
                    baseline = torch.empty(
                        (rows, width), device="cuda", dtype=torch.float32
                    )
                    launch_existing(
                        q, qs, keys, ks, weights, starts, ends, 64, 4, baseline
                    )
                    actual = dense_score(q, qs, keys, ks, weights, starts, ends)
                    torch.testing.assert_close(actual, baseline, rtol=0, atol=0)
                    baseline_topk = prefill_topk.try_select_tokens(
                        baseline, ends, bounds=(starts, ends)
                    )
                    actual_topk = prefill_topk.try_select_tokens(
                        actual, ends, bounds=(starts, ends)
                    )
                    self.assertIsNotNone(baseline_topk)
                    self.assertIsNotNone(actual_topk)
                    torch.testing.assert_close(
                        actual_topk, baseline_topk, rtol=0, atol=0
                    )

    @unittest.skipUnless(
        os.environ.get("DSV41_SM90_INDEXER_TUNING_EXPLORE") == "1",
        "full launch exploration is opt-in diagnostics, not selected-config acceptance",
    )
    def test_dense_producer_launch_tuning(self):
        bench = os.environ.get("DSV41_SM90_INDEXER_TUNING_BENCH") == "1"
        results = []
        for rows in (512, 1024):
            torch.manual_seed(4410 + rows)
            width = 32768
            q_input = (
                torch.randn(rows, 32, 128, device="cuda", dtype=torch.bfloat16) * 0.2
            )
            k_input = torch.randn(width, 128, device="cuda", dtype=torch.bfloat16) * 0.2
            q, qs, q_values = _fp4_rows_torch(q_input)
            keys, ks, k_values = _fp4_rows_torch(k_input)
            q, qs = q.view(rows, 32, 64), qs.view(rows, 32)
            weights = (torch.randn(rows, 32, device="cuda") / 64).bfloat16()
            # Exact boundary ties must retain the same native selection order.
            weights[rows // 2].zero_()
            starts = torch.zeros(rows, device="cuda", dtype=torch.int32)
            for phase in ("early_causal", "late_causal"):
                # A contiguous chunk from either half of CP's zigzag token
                # ownership: same full logical K slab, true causal row ends.
                first = 1 if phase == "early_causal" else width - rows + 1
                ends = torch.arange(
                    first, first + rows, device="cuda", dtype=torch.int32
                )
                baseline = torch.empty(
                    (rows, width), device="cuda", dtype=torch.float32
                )
                launch_existing(q, qs, keys, ks, weights, starts, ends, 64, 4, baseline)
                baseline_topk = prefill_topk.try_select_tokens(
                    baseline, ends, bounds=(starts, ends)
                )
                self.assertIsNotNone(baseline_topk)
                # Independent reference is sampled in both query and K, but
                # baseline-vs-candidate comparisons below cover every element.
                columns = torch.linspace(0, width - 1, 137, device="cuda").long()
                for row in (0, rows // 2, rows - 1):
                    oracle = independent_score(
                        q_values.view(rows, 32, 128)[row].bfloat16(),
                        k_values[columns].bfloat16(),
                        weights[row],
                    )
                    oracle.masked_fill_(
                        (columns < starts[row]) | (columns >= ends[row]), -torch.inf
                    )
                    # FP4 dyadic products in this controlled finite range are
                    # exactly representable in FP32; no tolerance is relaxed.
                    torch.testing.assert_close(
                        baseline[row, columns], oracle, rtol=0, atol=0
                    )
                for block in (64, 128, 256):
                    for warps in (4, 8):
                        with self.subTest(
                            rows=rows, phase=phase, block=block, warps=warps
                        ):
                            out = torch.empty_like(baseline)

                            def run():
                                return launch_existing(
                                    q,
                                    qs,
                                    keys,
                                    ks,
                                    weights,
                                    starts,
                                    ends,
                                    block,
                                    warps,
                                    out,
                                )

                            try:
                                actual = run()
                            except triton.runtime.errors.OutOfResources as error:
                                results.append(
                                    {
                                        "rows": rows,
                                        "keys": width,
                                        "phase": phase,
                                        "block": block,
                                        "warps": warps,
                                        "status": "UNSUPPORTED_LAUNCH",
                                        "reason": str(error),
                                    }
                                )
                                continue
                            record = {
                                "rows": rows,
                                "keys": width,
                                "heads": 32,
                                "dim": 128,
                                "phase": phase,
                                "block": block,
                                "warps": warps,
                                "status": "EXACT_PASS",
                            }
                            try:
                                torch.testing.assert_close(
                                    actual, baseline, rtol=0, atol=0
                                )
                                actual_topk = prefill_topk.try_select_tokens(
                                    actual, ends, bounds=(starts, ends)
                                )
                                self.assertIsNotNone(actual_topk)
                                torch.testing.assert_close(
                                    actual_topk, baseline_topk, rtol=0, atol=0
                                )
                            except AssertionError as error:
                                record.update(status="REJECT", reason=str(error))
                                results.append(record)
                                print(
                                    "SM90_INDEXER_LAUNCH_TUNING " + json.dumps(record),
                                    flush=True,
                                )
                                continue
                            if (block, warps) == (64, 8):
                                # Already failed M512 late-causal by 3 elements
                                # in round2_deepgemm_indexer_02. Passing another
                                # case must not rehabilitate this launch.
                                record.update(
                                    status="REJECT_PRIOR_NUMERICAL_MISMATCH",
                                    reason="BLOCK64/warps8 failed prior M512 late-causal exact comparison",
                                )
                            if bench:
                                run()
                                run()
                                torch.cuda.synchronize()
                                # Separate runs (not profiler timings); output
                                # allocation and all checks are outside timing.
                                record["kernel_ms_3_runs"] = [
                                    triton.testing.do_bench(run, warmup=25, rep=100)
                                    for _ in range(3)
                                ]
                            results.append(record)
                            print(
                                "SM90_INDEXER_LAUNCH_TUNING " + json.dumps(record),
                                flush=True,
                            )
        output_dir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            # Runtime test artifact, never a production selector/config file.
            with open(
                os.path.join(output_dir, "sm90_indexer_launch_tuning.json"), "w"
            ) as output:
                json.dump(results, output, indent=2)


if __name__ == "__main__":
    unittest.main()
