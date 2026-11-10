"""One-launch dual-table verdict for the QSA side-cache block tables.

The warm light-layer census showed every prefix-reuse layer window pays six
device launches on the two required-blocks verdict groups (two one-program
Triton reductions, two ``result == 0`` elementwise conversions and two
``_assert_async`` traps).  Both groups validate step-frozen inputs (the same
table tensors and required-column vectors for every layer of a step) against
this layer's pool block counts, so the two reductions fold into ONE Triton
launch that stores the combined boolean verdict directly (no elementwise
conversion) and ONE async assert.

This test locks the replacement contract:

* the dual-table verdict matches the two single-table verdicts' bitwise AND
  across valid and adversarial geometries (both tables, B > 1, boundary
  blocks, empty required windows and the 4096-lane bound);
* the non-strict geometry path enqueues exactly two device kernels per call
  without a forward scope, or per equivalent table/capacity/stream inside
  one explicit forward scope, and still traps invalid tables with one
  assert whose message keeps both cache names;
* unsupported geometries fall back to the combined single-table verdicts with
  identical results;
* new table shapes do not add compiled variants (fixed lanes,
  ``do_not_specialize`` shape arguments).
"""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

import torch

_DEV = "cuda"


def _table(rows, width):
    table = torch.zeros((len(rows), width), dtype=torch.int32, device=_DEV)
    for index, row in enumerate(rows):
        if row:
            table[index, : len(row)] = torch.tensor(row, dtype=torch.int32, device=_DEV)
    return table


class _DualCase:
    def __init__(
        self,
        tag,
        kv_rows,
        kv_width,
        kv_required,
        kv_pool,
        state_rows,
        state_width,
        state_required,
        state_pool,
        valid,
    ):
        self.tag = tag
        self.kv_table = _table(kv_rows, kv_width)
        self.kv_required = torch.tensor(kv_required, dtype=torch.int32, device=_DEV)
        self.kv_pool = kv_pool
        self.state_table = _table(state_rows, state_width)
        self.state_required = torch.tensor(
            state_required, dtype=torch.int32, device=_DEV
        )
        self.state_pool = state_pool
        self.valid = valid


_CASES = [
    _DualCase(
        "both_valid",
        [[1, 2, 3, 0], [5, 6, 7, 9]],
        4,
        [3, 3],
        16,
        [[1, 2], [4, 5]],
        3,
        [2, 2],
        16,
        True,
    ),
    _DualCase(
        "kv_required_column_unallocated",
        [[1, 0, 0, 0]],
        4,
        [2],
        16,
        [[1, 2]],
        3,
        [2],
        16,
        False,
    ),
    _DualCase(
        "state_required_column_unallocated",
        [[1, 2, 3, 0]],
        4,
        [3],
        16,
        [[1, 0]],
        3,
        [2],
        16,
        False,
    ),
    _DualCase(
        "kv_required_uncovered",
        [[1, 2, 3, 0]],
        4,
        [5],
        16,
        [[1, 2]],
        3,
        [2],
        16,
        False,
    ),
    _DualCase(
        "state_required_uncovered",
        [[1, 2, 3, 0]],
        4,
        [3],
        16,
        [[1, 2]],
        3,
        [4],
        16,
        False,
    ),
    _DualCase(
        "kv_negative_required",
        [[1, 2, 3, 0]],
        4,
        [-1],
        16,
        [[1, 2]],
        3,
        [2],
        16,
        False,
    ),
    _DualCase(
        "state_negative_required",
        [[1, 2, 3, 0]],
        4,
        [3],
        16,
        [[1, 2]],
        3,
        [-2],
        16,
        False,
    ),
    _DualCase(
        "kv_stale_column_out_of_pool",
        [[1, 2, 0, 0], [3, 0, 0, 17]],
        4,
        [2, 1],
        16,
        [[1, 2]],
        3,
        [2],
        16,
        False,
    ),
    _DualCase(
        "state_stale_column_out_of_pool",
        [[1, 2, 3, 0]],
        4,
        [3],
        16,
        [[1, 2, 18]],
        3,
        [2],
        16,
        False,
    ),
    _DualCase(
        "kv_pool_boundary",
        [[1, 15, 0, 0]],
        4,
        [2],
        16,
        [[1, 2]],
        3,
        [2],
        16,
        True,
    ),
    _DualCase(
        "state_pool_boundary_exceeded",
        [[1, 2, 3, 0]],
        4,
        [3],
        16,
        [[1, 16]],
        3,
        [2],
        16,
        False,
    ),
    _DualCase(
        "kv_required_equals_width",
        [[1, 2, 3, 4]],
        4,
        [4],
        16,
        [[1, 2, 3]],
        3,
        [3],
        16,
        True,
    ),
    _DualCase(
        "empty_required_windows",
        [[1, 2, 3, 0]],
        4,
        [0],
        16,
        [[1, 2]],
        3,
        [0],
        16,
        True,
    ),
    _DualCase(
        "batch_mixed_one_bad_state_row",
        [[1, 2], [3, 4], [5, 6]],
        2,
        [2, 2, 2],
        16,
        [[1, 2], [0, 0], [4, 5]],
        3,
        [2, 2, 2],
        16,
        False,
    ),
    _DualCase(
        "production_asymmetric_widths",
        [[((i % 4000) + 1) for i in range(55)]],
        55,
        [55],
        4096,
        [[((i % 4000) + 1) for i in range(110)]],
        110,
        [110],
        4096,
        True,
    ),
    _DualCase(
        "lane_bound_4096",
        [[(i % 63) + 1 for i in range(64)]] * 64,
        64,
        [64] * 64,
        64,
        [[(i % 63) + 1 for i in range(64)]] * 64,
        64,
        [64] * 64,
        64,
        True,
    ),
]


class DualTableVerdictTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_dual_verdict_matches_single_verdicts(self):
        from rtp_llm.models_py.modules.qwen4_exp.qsa.qsa_validation_triton import (
            dual_required_blocks_supported,
            dual_required_blocks_verdict,
            required_blocks_verdict,
        )

        for case in _CASES:
            with self.subTest(case.tag):
                self.assertTrue(
                    dual_required_blocks_supported(
                        case.kv_table,
                        case.kv_required,
                        case.state_table,
                        case.state_required,
                    ),
                    case.tag,
                )
                merged = dual_required_blocks_verdict(
                    case.kv_table,
                    case.kv_required,
                    case.state_table,
                    case.state_required,
                    kv_pool_blocks=case.kv_pool,
                    state_pool_blocks=case.state_pool,
                )
                singles = required_blocks_verdict(
                    case.kv_table, case.kv_required, case.kv_pool
                ) & required_blocks_verdict(
                    case.state_table, case.state_required, case.state_pool
                )
                self.assertEqual(bool(singles.item()), case.valid, case.tag)
                self.assertEqual(bool(merged.item()), bool(singles.item()), case.tag)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_strided_tables_match_contiguous_verdicts(self):
        from rtp_llm.models_py.modules.qwen4_exp.qsa.qsa_validation_triton import (
            dual_required_blocks_supported,
            dual_required_blocks_verdict,
        )

        # A padded row stride (engine tables may carry wider physical rows)
        # must not change the verdict.
        kv_wide = torch.zeros((2, 8), dtype=torch.int32, device=_DEV)
        kv_wide[:, :3] = _table([[1, 2, 3], [4, 5, 6]], 3)
        state_wide = torch.zeros((2, 6), dtype=torch.int32, device=_DEV)
        state_wide[:, :2] = _table([[1, 2], [3, 4]], 2)
        kv_table = kv_wide[:, :3]
        state_table = state_wide[:, :2]
        required = torch.tensor([2, 2], dtype=torch.int32, device=_DEV)
        self.assertFalse(kv_table.is_contiguous())
        self.assertEqual(kv_table.stride(), (8, 1))
        self.assertTrue(
            dual_required_blocks_supported(kv_table, required, state_table, required)
        )
        merged = dual_required_blocks_verdict(
            kv_table,
            required,
            state_table,
            required,
            kv_pool_blocks=16,
            state_pool_blocks=16,
        )
        self.assertTrue(bool(merged.item()))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_unsupported_geometry_falls_back_to_combined_singles(self):
        from rtp_llm.models_py.model_desc.qwen4_exp_runtime.qsa_cache_views import (
            _side_tables_verdict,
        )
        from rtp_llm.models_py.modules.qwen4_exp.qsa.qsa_validation_triton import (
            dual_required_blocks_supported,
        )

        # 2 x 4096 exceeds the shared one-program lane bound for both tables.
        kv_table = torch.zeros((2, 4096), dtype=torch.int32, device=_DEV)
        kv_table[:, :2] = 1
        state_table = torch.zeros((2, 4096), dtype=torch.int32, device=_DEV)
        state_table[:, :2] = 2
        required = torch.tensor([2, 2], dtype=torch.int32, device=_DEV)
        self.assertFalse(
            dual_required_blocks_supported(kv_table, required, state_table, required)
        )
        verdict = _side_tables_verdict(
            kv_table,
            required,
            state_table,
            required,
            kv_pool_blocks=16,
            state_pool_blocks=16,
        )
        self.assertTrue(bool(verdict.item()))
        # The fallback still rejects an invalid table with the combined check.
        state_table[1, 0] = 0
        verdict = _side_tables_verdict(
            kv_table,
            required,
            state_table,
            required,
            kv_pool_blocks=16,
            state_pool_blocks=16,
        )
        self.assertFalse(bool(verdict.item()))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cpu_tables_use_the_combined_verdict(self):
        from rtp_llm.models_py.model_desc.qwen4_exp_runtime.qsa_cache_views import (
            _side_tables_verdict,
        )

        kv_table = torch.tensor([[1, 2]], dtype=torch.int32)
        state_table = torch.tensor([[1, 2]], dtype=torch.int32)
        required = torch.tensor([2], dtype=torch.int32)
        verdict = _side_tables_verdict(
            kv_table,
            required,
            state_table,
            required,
            kv_pool_blocks=4,
            state_pool_blocks=4,
        )
        self.assertTrue(bool(verdict))
        state_table[0, 0] = 0
        verdict = _side_tables_verdict(
            kv_table,
            required,
            state_table,
            required,
            kv_pool_blocks=4,
            state_pool_blocks=4,
        )
        self.assertFalse(bool(verdict))


def _geometry_fixture():
    """A CUDA runtime context whose per-layer geometry runs both verdicts."""
    import indexer_prefix_reuse_triton_test as prefix_reuse_test

    harness = prefix_reuse_test.IndexerPrefixReuseTritonTest()
    context, indexer, rope, args = harness.runtime_inputs([128, 256], [132, 7])
    return context, indexer, rope, args, args["cu"].device


class IncrementalGeometryLaunchTest(unittest.TestCase):
    def _run_geometry(self, context, indexer, rope, token_count, device):
        return context._draft_incremental_prefill_geometry(
            indexer=indexer,
            token_count=token_count,
            device=device,
            rope_config=rope,
            draft=False,
            draft_prefix_reuse=False,
        )

    def test_geometry_enqueues_two_device_kernels_per_layer(self):
        from torch.profiler import ProfilerActivity, profile

        context, indexer, rope, args, device = _geometry_fixture()
        token_count = sum(args["lengths"])
        with patch.dict(os.environ, {"RTP_LLM_QSA_STRICT_VALIDATION": "0"}):
            # Warm the plan build and the Triton compile outside the profile.
            self._run_geometry(context, indexer, rope, token_count, device)
            torch.cuda.synchronize()
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
            ) as prof:
                self._run_geometry(context, indexer, rope, token_count, device)
                torch.cuda.synchronize()
        path = os.path.join(
            tempfile.mkdtemp(prefix="qsa_dual_table_verdict_"), "t.json"
        )
        prof.export_chrome_trace(path)
        with open(path) as handle:
            events = json.load(handle)["traceEvents"]
        kernels = [event["name"] for event in events if event.get("cat") == "kernel"]
        self.assertEqual(
            len(kernels),
            2,
            "the folded verdict must enqueue exactly the dual reduction and "
            f"the async assert, got {len(kernels)}: {kernels}",
        )

    def test_equivalent_layers_enqueue_one_pair_in_each_forward_scope(self):
        from torch.profiler import ProfilerActivity, profile

        from rtp_llm.models_py.model_desc.qwen4_exp_runtime.qsa_step_plan import (
            qsa_validation_step,
        )

        context, indexer, rope, args, device = _geometry_fixture()
        token_count = sum(args["lengths"])
        with patch.dict(os.environ, {"RTP_LLM_QSA_STRICT_VALIDATION": "0"}):
            # Warm plan construction and Triton compilation outside capture.
            self._run_geometry(context, indexer, rope, token_count, device)
            torch.cuda.synchronize()
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
            ) as prof:
                for _ in range(2):
                    with qsa_validation_step():
                        for _ in range(3):
                            self._run_geometry(
                                context, indexer, rope, token_count, device
                            )
                torch.cuda.synchronize()
        path = os.path.join(tempfile.mkdtemp(prefix="qsa_validation_once_"), "t.json")
        prof.export_chrome_trace(path)
        with open(path) as handle:
            events = json.load(handle)["traceEvents"]
        kernels = [event["name"] for event in events if event.get("cat") == "kernel"]
        self.assertEqual(len(kernels), 4, kernels)
        self.assertEqual(
            sum("_dual_table_required_blocks_kernel" in k for k in kernels), 2
        )
        self.assertEqual(sum("_assert_async_cuda_kernel" in k for k in kernels), 2)

    def test_different_cuda_streams_each_enqueue_their_own_checks(self):
        from rtp_llm.models_py.model_desc.qwen4_exp_runtime import qsa_step_plan

        context, indexer, rope, args, device = _geometry_fixture()
        token_count = sum(args["lengths"])
        with patch.dict(os.environ, {"RTP_LLM_QSA_STRICT_VALIDATION": "0"}):
            self._run_geometry(context, indexer, rope, token_count, device)
            current = torch.cuda.current_stream(device)
            other = torch.cuda.Stream(device=device)
            other.wait_stream(current)  # Metadata producer precedes both streams.
            with patch.object(
                qsa_step_plan,
                "_side_tables_verdict",
                wraps=qsa_step_plan._side_tables_verdict,
            ) as verdict, qsa_step_plan.qsa_validation_step():
                self._run_geometry(context, indexer, rope, token_count, device)
                self._run_geometry(context, indexer, rope, token_count, device)
                with torch.cuda.stream(other):
                    self._run_geometry(context, indexer, rope, token_count, device)
                    self._run_geometry(context, indexer, rope, token_count, device)
                self.assertEqual(verdict.call_count, 2)
            current.wait_stream(other)
            torch.cuda.synchronize()

    def test_cuda_table_mutation_in_scope_rechecks_invalid_verdict(self):
        from rtp_llm.models_py.model_desc.qwen4_exp_runtime.qsa_step_plan import (
            qsa_validation_step,
        )

        for tag, field in (
            ("indexer_kv_inputs", "kv_cache_kernel_block_id_device"),
            ("indexer_state_inputs", "kv_cache_block_id_device"),
        ):
            with self.subTest(tag=tag):
                context, indexer, rope, args, device = _geometry_fixture()
                token_count = sum(args["lengths"])
                with patch.dict(os.environ, {"RTP_LLM_QSA_STRICT_VALIDATION": "0"}):
                    self._run_geometry(context, indexer, rope, token_count, device)
                    # Inspect the real CUDA verdict without poisoning this test
                    # process's context with a device assert; no writer is run.
                    with patch(
                        "torch._assert_async"
                    ) as assert_async, qsa_validation_step():
                        self._run_geometry(context, indexer, rope, token_count, device)
                        getattr(getattr(context, tag), field)[0, 0] = 0
                        self._run_geometry(context, indexer, rope, token_count, device)
                        self.assertEqual(assert_async.call_count, 2)
                        self.assertFalse(bool(assert_async.call_args[0][0].item()))

    def test_invalid_tables_still_assert_once_with_both_cache_names(self):
        from rtp_llm.models_py.model_desc.qwen4_exp_runtime.qsa_step_plan import (
            clear_step_plan_cache,
        )

        for table_attr, tag in (
            ("indexer_kv_inputs", "indexer_kv"),
            ("indexer_state_inputs", "indexer_state"),
        ):
            with self.subTest(cache=tag):
                clear_step_plan_cache()
                context, indexer, rope, args, device = _geometry_fixture()
                token_count = sum(args["lengths"])
                table = getattr(
                    getattr(context, table_attr),
                    (
                        "kv_cache_kernel_block_id_device"
                        if tag == "indexer_kv"
                        else "kv_cache_block_id_device"
                    ),
                )
                table[0, 0] = 0  # unallocated required column
                with patch.dict(
                    os.environ, {"RTP_LLM_QSA_STRICT_VALIDATION": "0"}
                ), patch("torch._assert_async") as assert_async:
                    # Warm the plan build (patched: no device trap), then
                    # measure a plan-hit layer call.
                    self._run_geometry(context, indexer, rope, token_count, device)
                    assert_async.reset_mock()
                    self._run_geometry(context, indexer, rope, token_count, device)
                self.assertEqual(
                    assert_async.call_count,
                    1,
                    "the folded verdict must assert exactly once per layer",
                )
                verdict, message = assert_async.call_args[0]
                self.assertFalse(bool(verdict.item()))
                self.assertIn("indexer_kv", message)
                self.assertIn("indexer_state", message)
                self.assertIn("draft incremental tail", message)

    def test_valid_tables_assert_once_true(self):
        from rtp_llm.models_py.model_desc.qwen4_exp_runtime.qsa_step_plan import (
            clear_step_plan_cache,
        )

        clear_step_plan_cache()
        context, indexer, rope, args, device = _geometry_fixture()
        token_count = sum(args["lengths"])
        with patch.dict(os.environ, {"RTP_LLM_QSA_STRICT_VALIDATION": "0"}), patch(
            "torch._assert_async"
        ) as assert_async:
            # Warm the plan build, then measure a plan-hit layer call.
            self._run_geometry(context, indexer, rope, token_count, device)
            assert_async.reset_mock()
            self._run_geometry(context, indexer, rope, token_count, device)
        self.assertEqual(assert_async.call_count, 1)
        verdict = assert_async.call_args[0][0]
        self.assertTrue(bool(verdict.item()))

    def test_strict_mode_keeps_legacy_host_diagnostics(self):
        from rtp_llm.models_py.model_desc.qwen4_exp_runtime.qsa_step_plan import (
            clear_step_plan_cache,
        )

        clear_step_plan_cache()
        context, indexer, rope, args, device = _geometry_fixture()
        token_count = sum(args["lengths"])
        context.indexer_kv_inputs.kv_cache_kernel_block_id_device[0, 0] = 0
        with patch.dict(os.environ, {"RTP_LLM_QSA_STRICT_VALIDATION": "1"}), patch(
            "torch._assert_async"
        ) as assert_async:
            with self.assertRaisesRegex(RuntimeError, "indexer_kv"):
                self._run_geometry(context, indexer, rope, token_count, device)
        assert_async.assert_not_called()


class DualTableVariantTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_no_new_variants_across_table_shapes(self):
        from rtp_llm.models_py.modules.qwen4_exp.qsa.qsa_validation_triton import (
            _dual_table_required_blocks_kernel,
            dual_required_blocks_verdict,
        )

        def variant_total(fn):
            caches = getattr(fn, "device_caches", None) or {}
            total = 0
            for cache in caches.values():
                entry = cache[0] if isinstance(cache, tuple) else cache
                total += len(entry)
            return total

        def run(kv_width, state_width, batch, pool):
            kv_row = [(i % (pool - 1)) + 1 for i in range(kv_width)]
            state_row = [(i % (pool - 1)) + 1 for i in range(state_width)]
            kv_table = _table([kv_row] * batch, kv_width)
            state_table = _table([state_row] * batch, state_width)
            required = torch.full(
                (batch,),
                min(kv_width, state_width),
                dtype=torch.int32,
                device=_DEV,
            )
            dual_required_blocks_verdict(
                kv_table,
                required,
                state_table,
                required,
                kv_pool_blocks=pool,
                state_pool_blocks=pool,
            )

        # Warm every stride-alignment class pair the contiguous engine
        # tables produce (row strides equal the widths; odd vs divisible
        # by 16) so the probes stay inside warmed classes.
        for kv_width, state_width, batch, pool in (
            (55, 110, 2, 4096),
            (64, 128, 1, 4096),
            (55, 128, 2, 4096),
            (64, 110, 1, 4096),
            (3, 4, 4, 16),
        ):
            run(kv_width, state_width, batch, pool)
        before = variant_total(_dual_table_required_blocks_kernel)
        for kv_width, state_width, batch, pool in (
            (63, 130, 2, 4096),
            (33, 96, 3, 4096),
            (96, 33, 2, 4096),
            (80, 144, 1, 4096),
            (7, 8, 2, 16),
        ):
            run(kv_width, state_width, batch, pool)
        self.assertEqual(
            variant_total(_dual_table_required_blocks_kernel),
            before,
            "table shapes must not add compiled variants (fixed lanes, "
            "do_not_specialize shape arguments)",
        )


if __name__ == "__main__":
    unittest.main()
