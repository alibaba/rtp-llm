"""Budget policy and real FP8 gather -> logits -> TopK equivalence tests."""

import json
import os
import statistics
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import TestCase, main, skipUnless
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.base.cuda.indexer_op import (
    IndexerOp,
    _fp8_mqa_chunk_workspace_bytes,
    _fp8_mqa_logits_chunk_rows,
)
from rtp_llm.ops.compute_ops import rtp_llm_ops


@contextmanager
def policy_env(settings):
    # Isolate only policy knobs; preserve PATH/HOME/CUDA for DeepGEMM JIT.
    clean = {
        k: v
        for k, v in os.environ.items()
        if k
        not in (
            "DSV4_CHUNK_TOKENS",
            "DSV4_FP8_INDEXER_SCORE_CHUNK_ROWS",
            "RTP_LLM_INDEXER_WORKSPACE_MB",
            "DSV4_PREFILL_TOPK_FORCE_RADIX",
            "DSV4_INDEXER_TOPK_CANONICALIZE",
            "RTP_INDEXER_FUSED_TOPK",
        )
    }
    clean.update(settings)
    with patch.dict(os.environ, clean, clear=True):
        yield


class IndexerWorkspacePolicyTest(TestCase):
    def test_explicit_rows_and_global_override_keep_original_behavior(self):
        for env, expected in (
            ({"DSV4_FP8_INDEXER_SCORE_CHUNK_ROWS": "13"}, 13),
            ({"DSV4_FP8_INDEXER_SCORE_CHUNK_ROWS": "0"}, 999),
            ({"DSV4_FP8_INDEXER_SCORE_CHUNK_ROWS": "-1"}, 999),
            ({"DSV4_FP8_INDEXER_SCORE_CHUNK_ROWS": "invalid"}, 999),
            ({"DSV4_CHUNK_TOKENS": "7", "DSV4_FP8_INDEXER_SCORE_CHUNK_ROWS": "13"}, 7),
        ):
            with self.subTest(env=env), policy_env(env):
                self.assertEqual(
                    _fp8_mqa_logits_chunk_rows(999, 1000000, 2048, 32), expected
                )

    def test_budget_uses_padding_dtype_and_live_writer_workspace(self):
        for heads in (8, 16, 32, 64):
            for dtype in (torch.float16, torch.float32):
                for k in (1, 255, 256, 257, 8193, 1000000):
                    for canonicalize in (False, True):
                        with self.subTest(
                            heads=heads, dtype=dtype, k=k, canonical=canonicalize
                        ):
                            env = {
                                "DSV4_INDEXER_TOPK_CANONICALIZE": str(int(canonicalize))
                            }
                            with policy_env(env):
                                rows = _fp8_mqa_logits_chunk_rows(
                                    1000000, k, 2048, heads, dtype
                                )
                            peak = _fp8_mqa_chunk_workspace_bytes(
                                rows, k, 2048, heads, dtype, canonicalize
                            )
                            self.assertLessEqual(peak, 256 * 1024 * 1024)
                            block_q = 128 // heads
                            self.assertEqual(rows % block_q, 0)
                            next_peak = _fp8_mqa_chunk_workspace_bytes(
                                rows + block_q, k, 2048, heads, dtype, canonicalize
                            )
                            self.assertGreater(next_peak, 256 * 1024 * 1024)
        # A 1-row view still retains its entire padded DeepGEMM allocation.
        self.assertEqual(
            _fp8_mqa_chunk_workspace_bytes(1, 257, 1, 32, torch.float32, False),
            4 * 3072 + 4,
        )
        with policy_env({}):
            self.assertGreater(
                _fp8_mqa_logits_chunk_rows(100000, 1000000, 512, 32, torch.float16),
                _fp8_mqa_logits_chunk_rows(100000, 1000000, 512, 32, torch.float32),
            )

    def test_non_radix_legacy_keeps_original_algorithm_boundary(self):
        with policy_env({"DSV4_PREFILL_TOPK_FORCE_RADIX": "0"}):
            self.assertEqual(_fp8_mqa_logits_chunk_rows(20000, 1000000, 512, 32), 20000)
            self.assertLess(
                _fp8_mqa_logits_chunk_rows(
                    20000, 1000000, 512, 32, topk_backend="topk_v3_tie_break"
                ),
                20000,
            )
            with patch.dict(os.environ, {"DSV4_FP8_INDEXER_SCORE_CHUNK_ROWS": "2048"}):
                self.assertEqual(
                    _fp8_mqa_logits_chunk_rows(20000, 1000000, 512, 32), 2048
                )

    def test_zero_tail_tiny_budget_and_invalid_budget(self):
        with policy_env({}):
            self.assertEqual(_fp8_mqa_logits_chunk_rows(0, 1, 512, 32), 0)
            self.assertEqual(_fp8_mqa_logits_chunk_rows(3, 100, 512, 32), 3)
            expected = _fp8_mqa_logits_chunk_rows(10000, 10000, 512, 32)
        with policy_env({"RTP_LLM_INDEXER_WORKSPACE_MB": "invalid"}):
            self.assertEqual(
                _fp8_mqa_logits_chunk_rows(10000, 10000, 512, 32), expected
            )
        with policy_env({"RTP_LLM_INDEXER_WORKSPACE_MB": "1"}):
            self.assertEqual(_fp8_mqa_logits_chunk_rows(100, 1000000, 512, 32), 4)
            self.assertEqual(_fp8_mqa_logits_chunk_rows(1, 1000000, 512, 32), 1)


def make_case(
    rows, k, topk=512, backend="dsv4_per_row", offset_dtype=torch.int32, prefix=False
):
    with patch.dict(os.environ, {"GLM5_PREFILL_INDEXER_TOPK_BACKEND": backend}):
        op = IndexerOp(32, 128, topk, 64, blocksize=64)
    q = (torch.randn(rows, 32, 128, device="cuda") * 0.25).to(torch.float8_e4m3fn)
    weights = torch.rand(rows, 32, 1, device="cuda")
    keys = torch.randn(k, 128, device="cuda", dtype=torch.bfloat16)
    cache = SimpleNamespace(
        kv_scale_base=torch.empty(
            ((k + 63) // 64, 64, 132), dtype=torch.uint8, device="cuda"
        )
    )
    slots = torch.arange(k, dtype=torch.int64, device="cuda")
    rtp_llm_ops.indexer_k_quant_and_cache(
        keys, cache.kv_scale_base, slots, 128, "ue8m0"
    )
    block_table = torch.arange(
        (k + 63) // 64, dtype=torch.int32, device="cuda"
    ).reshape(1, -1)
    inputs = SimpleNamespace(
        kv_cache_kernel_block_id_device=block_table,
        cu_kv_seqlens=torch.tensor([0, k], dtype=torch.int32, device="cuda"),
    )
    # Include rows shorter than topk, prefix reuse, and the full K tail.
    ke = (
        torch.arange(k - rows + 1, k + 1, dtype=torch.int32, device="cuda")
        if prefix
        else torch.linspace(1, k, rows, dtype=torch.int32, device="cuda")
    )
    ks = torch.zeros(rows, dtype=torch.int32, device="cuda")
    offsets = torch.arange(rows, dtype=offset_dtype, device="cuda") * (
        0 if prefix else 7
    )
    fmha = SimpleNamespace(
        prefill_total_kv_tokens=k,
        prefill_max_kv_len=k,
        ks=ks,
        ke=ke,
        expanded_seq_lens=ke,
        topk_indices_offset=offsets,
    )
    local_ids = torch.arange(rows, dtype=torch.long, device="cuda")

    def run(cp=False):
        if not cp:
            return op._get_topk_ragged(q, weights, cache, fmha, inputs)
        return op._get_topk_ragged_cp(
            q,
            weights,
            cache,
            fmha,
            inputs,
            local_ids,
            inputs.cu_kv_seqlens,
            k,
            ks,
            ke,
            ke,
            offsets,
            total_local_ids_is_identity=True,
        )

    return run, q, weights


@skipUnless(torch.cuda.is_available(), "CUDA required")
class IndexerWorkspaceGpuTest(TestCase):
    def test_actual_ragged_and_cp_topk_sets_and_canonical_order_match(self):
        torch.manual_seed(91)
        for topk, backend in (
            (512, "dsv4_per_row"),
            (2048, "dsv4_per_row"),
            (512, "topk_v3_tie_break"),
        ):
            run, q, weights = make_case(129, 8193, topk, backend, torch.int64)
            original_q, original_w = q.view(torch.uint8).clone(), weights.clone()
            for canonical in ("0", "1"):
                for fused in ("0", "1"):
                    env = {
                        "DSV4_PREFILL_TOPK_FORCE_RADIX": "1",
                        "DSV4_INDEXER_TOPK_CANONICALIZE": canonical,
                        "RTP_INDEXER_FUSED_TOPK": fused,
                        "RTP_LLM_INDEXER_WORKSPACE_MB": "1",
                    }
                    with self.subTest(
                        topk=topk, backend=backend, canonical=canonical, fused=fused
                    ), policy_env(env):
                        with patch.dict(
                            os.environ, {"DSV4_FP8_INDEXER_SCORE_CHUNK_ROWS": "0"}
                        ):
                            expected = run()
                        self.assertLess(
                            _fp8_mqa_logits_chunk_rows(129, 8193, topk, 32), 129
                        )
                        for cp in (False, True):
                            torch.cuda.synchronize()
                            live_before = torch.cuda.memory_stats()[
                                "requested_bytes.all.current"
                            ]
                            torch.cuda.reset_peak_memory_stats()
                            actual = run(cp)
                            torch.cuda.synchronize()
                            peak = (
                                torch.cuda.memory_stats()["requested_bytes.all.peak"]
                                - live_before
                            )
                            # Full K/scales and the final output are fixed storage;
                            # the remaining chunk-local live peak stays in budget.
                            fixed = 8193 * 132 + 129 * topk * 4
                            self.assertLessEqual(peak, fixed + 1024 * 1024 + 65536)
                            if canonical == "1":
                                self.assertTrue(torch.equal(actual, expected))
                            else:
                                # Neither backend promises order without
                                # canonicalization (even identical reruns differ).
                                self.assertTrue(
                                    torch.equal(
                                        actual.sort().values, expected.sort().values
                                    )
                                )
                            del actual
            self.assertTrue(torch.equal(q.view(torch.uint8), original_q))
            self.assertTrue(torch.equal(weights, original_w))

    def test_real_score_storage_padding_is_budgeted(self):
        import deep_gemm

        for rows, k in ((1, 257), (5, 8193), (17, 1025)):
            q = torch.zeros((rows, 32, 128), dtype=torch.float8_e4m3fn, device="cuda")
            kv = (
                torch.zeros((k, 128), dtype=q.dtype, device="cuda"),
                torch.ones(k, device="cuda"),
            )
            weights = torch.ones((rows, 32), device="cuda")
            starts = torch.zeros(rows, dtype=torch.int32, device="cuda")
            ends = torch.full_like(starts, k)
            scores = deep_gemm.fp8_mqa_logits(
                q, kv, weights, starts, ends, clean_logits=False
            )
            expected = _fp8_mqa_chunk_workspace_bytes(
                rows, k, 1, 32, scores.dtype, False
            )
            self.assertEqual(scores.untyped_storage().nbytes() + rows * 4, expected)

    def test_random_valid_scores_are_bit_exact_across_chunks(self):
        import deep_gemm

        torch.manual_seed(120)
        rows, k = 129, 8193
        q = torch.randn(rows, 32, 128, device="cuda").to(torch.float8_e4m3fn)
        kv = (
            torch.randn(k, 128, device="cuda").to(q.dtype),
            torch.rand(k, device="cuda"),
        )
        weights = torch.rand(rows, 32, device="cuda")
        starts = torch.zeros(rows, dtype=torch.int32, device="cuda")
        ends = torch.linspace(1, k, rows, dtype=torch.int32, device="cuda")
        valid = torch.arange(k, device="cuda")[None, :] < ends[:, None]
        expected = deep_gemm.fp8_mqa_logits(
            q, kv, weights, starts, ends, clean_logits=False
        )
        for chunk in (4, 28, 64, 128):
            actual = torch.cat(
                [
                    deep_gemm.fp8_mqa_logits(
                        q[i : i + chunk],
                        kv,
                        weights[i : i + chunk],
                        starts[i : i + chunk],
                        ends[i : i + chunk],
                        clean_logits=False,
                    )
                    for i in range(0, rows, chunk)
                ]
            )
            self.assertTrue(torch.equal(actual[valid], expected[valid]))

    @skipUnless(
        os.environ.get("RUN_INDEXER_BUDGET_BENCHMARK") == "1", "manual benchmark"
    )
    def test_benchmark_real_ragged_peak_and_latency(self):
        results = []
        policies = ("0", "512", "2048", "auto")
        for rows, k in ((8192, 8192), (8192, 16384), (4096, 65536)):
            run, _, _ = make_case(rows, k, prefix=True)
            env = {
                "DSV4_INDEXER_TOPK_CANONICALIZE": "0",
                "DSV4_PREFILL_TOPK_FORCE_RADIX": "1",
            }
            with policy_env(env):
                timings = {}
                expected = None
                for policy in policies:
                    patch_env = (
                        {}
                        if policy == "auto"
                        else {"DSV4_FP8_INDEXER_SCORE_CHUNK_ROWS": policy}
                    )
                    with patch.dict(os.environ, patch_env):
                        for _ in range(3):
                            result = run()
                        torch.cuda.synchronize()
                        if expected is None:
                            expected = result.sort().values
                        else:
                            self.assertTrue(torch.equal(expected, result.sort().values))
                        del result
                        torch.cuda.synchronize()
                        baseline = torch.cuda.memory_allocated()
                        requested_before = torch.cuda.memory_stats()[
                            "requested_bytes.all.current"
                        ]
                        torch.cuda.reset_peak_memory_stats()
                        result = run()
                        torch.cuda.synchronize()
                        peak = torch.cuda.max_memory_allocated() - baseline
                        requested_peak = (
                            torch.cuda.memory_stats()["requested_bytes.all.peak"]
                            - requested_before
                        )
                        if policy == "auto":
                            self.assertLessEqual(
                                requested_peak,
                                k * 132 + rows * 512 * 4 + 256 * 1024 * 1024 + 65536,
                            )
                        del result
                        timings[policy] = {
                            "chunk_rows": _fp8_mqa_logits_chunk_rows(rows, k, 512, 32),
                            "peak_allocated_bytes": peak,
                            "peak_requested_bytes": requested_peak,
                            "samples_ms": [],
                        }
                # Alternate policy order to reduce clock/thermal drift bias.
                for repeat in range(9):
                    order = (
                        policies[repeat % len(policies) :]
                        + policies[: repeat % len(policies)]
                    )
                    for policy in order:
                        patch_env = (
                            {}
                            if policy == "auto"
                            else {"DSV4_FP8_INDEXER_SCORE_CHUNK_ROWS": policy}
                        )
                        with patch.dict(os.environ, patch_env):
                            start, end = torch.cuda.Event(
                                enable_timing=True
                            ), torch.cuda.Event(enable_timing=True)
                            start.record()
                            for _ in range(5):
                                result = run()
                                del result
                            end.record()
                            end.synchronize()
                            timings[policy]["samples_ms"].append(
                                start.elapsed_time(end) / 5
                            )
                for values in timings.values():
                    values["gpu_ms"] = statistics.median(values["samples_ms"])
                results.append({"q": rows, "k": k, "results": timings})
                print("INDEXER_BUDGET_CASE=" + json.dumps(results[-1]), flush=True)
        print("INDEXER_BUDGET_BENCHMARK=" + json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
