"""Scheduled FP4 prefill scoring: ABI fallbacks and real SM10x integration.

The CUDA target executes native metadata, DeepGEMM and finite TopK. Its memory
check covers the production selection loop, not whole-model serving memory.
"""

import gc
import json
import os
import tempfile
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41.fp8 import _indexer_score as backend
from rtp_llm.models_py.modules.dsv41.fp8 import _v41_mqa_layout as layout
from rtp_llm.models_py.modules.dsv41.fp8 import _v41_prefill_indexer as indexer
from rtp_llm.models_py.modules.dsv41.fp8 import _v41_prefill_topk as topk
from rtp_llm.models_py.modules.dsv41.fp8 import attention_v41 as attention
from rtp_llm.models_py.modules.dsv41.fp8.test.test_v41_prefill_indexer import (
    _reference_logits,
)

ENV = "DSV41_PREFILL_INDEXER_SCHEDULED"
MIB = 1 << 20


def _vendor(doc, builder=True):
    def operation(*args, **kwargs):
        raise AssertionError("Capability checks must not launch a kernel")

    operation.__doc__ = doc
    return SimpleNamespace(
        fp8_fp4_mqa_logits=operation,
        get_mqa_logits_metadata=(lambda *args: None) if builder else None,
    )


class ScheduledIndexerCPU(unittest.TestCase):
    def test_native_signature_builder_and_architecture_gate(self):
        for doc, builder, supported in (
            ("schedule_meta: Tensor | None", True, True),
            ("max_seqlen_k: int", True, False),
            ("schedule_meta: Tensor | None", False, False),
            ("logits_dtype: torch.dtype, schedule_meta: Tensor", True, False),
        ):
            for capability in ((9, 0), (10, 0), (10, 3), (12, 0)):
                for heads in (32, 64):
                    with self.subTest(
                        doc=doc, builder=builder, capability=capability, heads=heads
                    ), patch.object(
                        backend, "_deep_gemm", _vendor(doc, builder)
                    ), patch.object(
                        torch.version, "hip", None
                    ), patch.object(
                        torch.cuda, "get_device_capability", return_value=capability
                    ):
                        self.assertEqual(
                            backend.supports_fp8_fp4_mqa_schedule(
                                torch.device("cuda"), heads
                            ),
                            supported and capability[0] == 10 and heads == 32,
                        )

    def test_cpu_and_hip_do_not_query_cuda(self):
        with patch.object(
            torch.cuda,
            "get_device_capability",
            side_effect=AssertionError("Unexpected CUDA probe"),
        ):
            self.assertFalse(
                backend.supports_fp8_fp4_mqa_schedule(torch.device("cpu"), 32)
            )
            with patch.object(torch.version, "hip", "7.0"):
                self.assertFalse(
                    backend.supports_fp8_fp4_mqa_schedule(torch.device("cuda"), 32)
                )

    def test_chunk_plan_keeps_unsupported_and_disabled_paths_unchanged(self):
        for enabled, supported, bf16, heads in (
            ("0", True, True, 32),
            ("1", False, True, 32),
            ("1", True, False, 32),
            ("1", False, True, 64),
        ):
            with patch.dict(os.environ, {ENV: enabled}), patch.object(
                backend, "supports_fp8_fp4_mqa_schedule", return_value=supported
            ):
                for keys in (0, 16384, 131072, 524224, 1048448, 1048577):
                    self.assertEqual(
                        indexer.logits_chunk_plan(
                            keys,
                            device=torch.device("cuda"),
                            num_heads=heads,
                            return_bf16=bf16,
                        ),
                        (indexer.logits_chunk_rows(keys), False),
                    )

    def test_long_plan_bounds_physical_bf16_allocation_and_keeps_short_chunks(self):
        with patch.dict(os.environ, {ENV: "1"}), patch.object(
            backend, "supports_fp8_fp4_mqa_schedule", return_value=True
        ):
            expected = {
                131072: (512, False),
                131073: (512, True),
                524224: (512, True),
                1048448: (380, True),
                1048576: (380, True),
            }
            for keys, value in expected.items():
                self.assertEqual(
                    indexer.logits_chunk_plan(
                        keys,
                        device=torch.device("cuda"),
                        num_heads=32,
                        return_bf16=True,
                    ),
                    value,
                )
            for keys in range(131073, 1048577, 4093):
                rows, scheduled = indexer.logits_chunk_plan(
                    keys, device=torch.device("cuda"), num_heads=32, return_bf16=True
                )
                self.assertTrue(scheduled)
                self.assertEqual(rows % 4, 0)
                self.assertLessEqual(rows, 512)
                # Vendor physical stride rounds KV to 384, then output to 512.
                stride = ((keys + 383) // 384 * 384 + 511) // 512 * 512
                self.assertLessEqual(rows * stride * 2, 768 * MIB)
            self.assertEqual(
                indexer.logits_chunk_plan(
                    1048577, device=torch.device("cuda"), num_heads=32, return_bf16=True
                ),
                (indexer.logits_chunk_rows(1048577), False),
            )

    def test_explicit_schedule_falls_back_for_old_wheel_and_fp32(self):
        q = torch.zeros(2, 32, 64, dtype=torch.int8)
        sf = torch.zeros(2, 32, dtype=torch.int32)
        k = torch.zeros(7, 64, dtype=torch.int8)
        ks = torch.zeros(7, dtype=torch.int32)
        bounds = torch.tensor([0, 7], dtype=torch.int32)
        result = torch.zeros(2, 7, dtype=torch.bfloat16)
        for bf16 in (True, False):
            calls = []

            def old(q, k, weights, starts, ends, *, max_seqlen_k):
                """max_seqlen_k: int"""
                calls.append(max_seqlen_k)
                return result

            vendor = SimpleNamespace(fp8_fp4_mqa_logits=old)
            with patch.object(backend, "_deep_gemm", vendor), patch.object(
                layout, "normalize_mqa_logits", return_value=result
            ), patch.object(torch.cuda, "get_device_capability", return_value=(10, 3)):
                actual = backend.fp8_fp4_mqa_indexer_score(
                    q,
                    sf,
                    k,
                    ks,
                    torch.ones(2, 32),
                    torch.zeros_like(bounds),
                    bounds,
                    max_seqlen_k=7,
                    return_bf16=bf16,
                    use_schedule=True,
                )
            self.assertIs(actual, result)
            self.assertEqual(calls, [7])

    def test_supported_metadata_failure_propagates_without_native_retry(self):
        def fail(*args):
            raise RuntimeError("metadata launch failed")

        vendor = _vendor("schedule_meta: Tensor | None")
        vendor.get_mqa_logits_metadata = fail
        with patch.object(backend, "_deep_gemm", vendor), patch.object(
            backend, "supports_fp8_fp4_mqa_schedule", return_value=True
        ), self.assertRaisesRegex(RuntimeError, "metadata launch failed"):
            backend.fp8_fp4_mqa_indexer_score(
                torch.zeros(1, 32, 64, dtype=torch.int8),
                torch.ones(1, 32, dtype=torch.int32),
                torch.zeros(7, 64, dtype=torch.int8),
                torch.ones(7, dtype=torch.int32),
                torch.ones(1, 32),
                torch.zeros(1, dtype=torch.int32),
                torch.ones(1, dtype=torch.int32),
                max_seqlen_k=7,
                return_bf16=True,
                use_schedule=True,
            )

    def test_fp32_compatibility_does_not_build_supported_metadata(self):
        result = torch.zeros(1, 7)

        def native(*args, **kwargs):
            """schedule_meta: Tensor | None"""
            self.assertNotIn("schedule_meta", kwargs)
            return result

        def metadata(*args):
            self.fail("FP32 compatibility must not build metadata")

        vendor = SimpleNamespace(
            fp8_fp4_mqa_logits=native, get_mqa_logits_metadata=metadata
        )
        with patch.object(backend, "_deep_gemm", vendor), patch.object(
            backend, "supports_fp8_fp4_mqa_schedule", return_value=True
        ), patch.object(layout, "normalize_mqa_logits", return_value=result):
            self.assertIs(
                backend.fp8_fp4_mqa_indexer_score(
                    torch.zeros(1, 32, 64, dtype=torch.int8),
                    torch.ones(1, 32, dtype=torch.int32),
                    torch.zeros(7, 64, dtype=torch.int8),
                    torch.ones(7, dtype=torch.int32),
                    torch.ones(1, 32),
                    torch.zeros(1, dtype=torch.int32),
                    torch.ones(1, dtype=torch.int32),
                    max_seqlen_k=7,
                    return_bf16=False,
                    use_schedule=True,
                ),
                result,
            )


@unittest.skipUnless(
    os.environ.get("DSV41_TEST_FUSED_INDEXER_GPU") == "1",
    "Explicit SM10x CUDA target required",
)
class ScheduledIndexerCUDA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
            raise RuntimeError("Scheduled indexer test requires a real SM10x CUDA GPU")
        cls.device = torch.device("cuda")
        if (
            not backend.supports_fp8_fp4_mqa_schedule(cls.device, 32)
            or not topk.supports_bf16()
        ):
            raise RuntimeError(
                "Native scheduled FP4 scorer and BF16 finite TopK are required"
            )

    def setUp(self):
        env = patch.dict(os.environ, {ENV: "1", "DSV41_PREFILL_DEEPSELECT": "0"})
        env.start()
        self.addCleanup(env.stop)

    def _inputs(self, rows, keys, seed=2741):
        gen = torch.Generator(device=self.device).manual_seed(seed)
        q = torch.randn(rows, 32, 128, generator=gen, device=self.device).bfloat16()
        k = torch.randn(keys, 128, generator=gen, device=self.device).bfloat16()
        weights = (
            torch.randn(rows, 32, generator=gen, device=self.device) / 64
        ).bfloat16()
        qp, qs = indexer.quantize_indexer_q(q)
        kp, ks = indexer.quantize_indexer_k_reference(k)
        return q, weights, k, qp, qs, kp, ks

    def _score(self, packed, ends, scheduled, starts=None):
        _, weights, _, qp, qs, kp, ks = packed
        starts = torch.zeros_like(ends) if starts is None else starts
        result = indexer.score_indexer_chunk(
            qp,
            qs,
            kp,
            ks,
            weights,
            ends,
            bounds=(starts, ends),
            return_bf16=True,
            use_schedule=scheduled,
        )
        stride = ((kp.shape[0] + 383) // 384 * 384 + 511) // 512 * 512
        self.assertEqual(result.stride(), (stride, 1))
        self.assertEqual(
            result.untyped_storage().nbytes(), (qp.shape[0] + 3) // 4 * 4 * stride * 2
        )
        return result

    def _topk_contract(self, scores, ends):
        columns = torch.arange(scores.shape[1], device=self.device)
        reference = scores.masked_fill(columns[None] >= ends[:, None], -torch.inf)
        expected = reference.topk(512, dim=1).values
        expected.masked_fill_(~expected.isfinite(), -torch.inf)
        output = torch.full(
            (scores.shape[0], 512), -12345, dtype=torch.int32, device=self.device
        )
        selected = topk.try_select_tokens(
            scores, ends, bounds=(torch.zeros_like(ends), ends), out=output
        )
        self.assertIs(selected, output)
        keep = selected >= 0
        self.assertTrue(bool((selected >= -1).all()))
        self.assertTrue(bool(((selected < ends[:, None]) | ~keep).all()))
        ordered = selected.sort(1).values
        self.assertFalse(
            bool(((ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] >= 0)).any())
        )
        values = scores.gather(1, selected.clamp_min(0).long()).masked_fill(
            ~keep, -torch.inf
        )
        self.assertTrue(bool((values.isfinite() | ~keep).all()))
        torch.testing.assert_close(
            values.sort(1).values, expected.sort(1).values, rtol=0, atol=0
        )
        return values.sort(1).values

    def test_real_metadata_nonmonotonic_bounds_q_tail_and_valid_scores(self):
        # A CP half transition is not a sorted request batch. Repeated descents,
        # empty rows, K512 and 64K boundaries must not change score semantics.
        rows, keys = 129, 65537
        packed = self._inputs(rows, keys)
        pattern = [keys, 0, 511, 512, 65535, 1, 65536, 17, keys - 1]
        ends = torch.tensor(
            [pattern[i % len(pattern)] for i in range(rows)],
            dtype=torch.int32,
            device=self.device,
        )
        baseline = self._score(packed, ends, False)
        self._score(packed, ends, True)  # Compile before the execution receipt.
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof:
            candidate = self._score(packed, ends, True)
            torch.cuda.synchronize()
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "scheduled.json")
            prof.export_chrome_trace(path)
            with open(path) as stream:
                kernels = [
                    event["name"]
                    for event in json.load(stream)["traceEvents"]
                    if event.get("cat") == "kernel"
                ]
        self.assertEqual(
            sum(
                "deep_gemm::sched::sm100_mqa_logits_metadata<" in name
                for name in kernels
            ),
            1,
        )
        self.assertEqual(
            sum(
                "deep_gemm::sm100_mqa_logits<32u, 128u, true," in name
                for name in kernels
            ),
            1,
        )
        self.assertFalse(
            any(
                "deep_gemm::sm100_mqa_logits<32u, 128u, false," in name
                for name in kernels
            )
        )
        clean_baseline = layout.normalize_mqa_logits(
            baseline, torch.zeros_like(ends), ends, keys, relative=True
        )
        clean_candidate = layout.normalize_mqa_logits(
            candidate, torch.zeros_like(ends), ends, keys, relative=True
        )
        torch.testing.assert_close(clean_candidate, clean_baseline, rtol=0, atol=0)
        torch.testing.assert_close(
            self._topk_contract(candidate, ends),
            self._topk_contract(baseline, ends),
            rtol=0,
            atol=0,
        )
        candidate.masked_fill_(
            torch.arange(keys, device=self.device)[None] >= ends[:, None], float("nan")
        )
        self._topk_contract(candidate, ends)
        # Metadata must be rebuilt from current bounds, not reused by shape.
        ends = ends.flip(0).contiguous()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            repeated = self._score(packed, ends, True)
        torch.cuda.current_stream().wait_stream(stream)
        reference = self._score(packed, ends, False)
        torch.testing.assert_close(
            layout.normalize_mqa_logits(
                repeated, torch.zeros_like(ends), ends, keys, relative=True
            ),
            layout.normalize_mqa_logits(
                reference, torch.zeros_like(ends), ends, keys, relative=True
            ),
            rtol=0,
            atol=0,
        )

    def test_nonzero_starts_request_relative_scores_and_small_math_oracle(self):
        packed = self._inputs(7, 4099, 2742)
        starts = torch.tensor(
            [0, 7, 511, 2048, 3, 4000, 17], dtype=torch.int32, device=self.device
        )
        ends = torch.tensor(
            [0, 519, 1024, 4099, 257, 4099, 17], dtype=torch.int32, device=self.device
        )
        default = self._score(packed, ends, False, starts)
        scheduled = self._score(packed, ends, True, starts)
        width = packed[2].shape[0]
        clean = layout.normalize_mqa_logits(
            scheduled, starts, ends, width, relative=False
        )
        torch.testing.assert_close(
            clean,
            layout.normalize_mqa_logits(default, starts, ends, width, relative=False),
            rtol=0,
            atol=0,
        )
        expected = _reference_logits(packed[0], packed[1], packed[2], ends)
        expected.masked_fill_(
            torch.arange(width, device=self.device)[None] < starts[:, None], -torch.inf
        )
        torch.testing.assert_close(clean, expected, rtol=3e-4, atol=8e-4)

    def test_selection_loop_exact_values_releases_logits_and_bounds_peak(self):
        rows, keys = 513, 524224
        q, weights, k, qp, qs, kp, ks = self._inputs(rows, keys, 2743)
        del k
        # Real CP0 positions around its front/back transition and a tail row.
        positions = torch.cat(
            (
                torch.arange(130800, 131056, device=self.device),
                torch.arange(917392, 917649, device=self.device),
            )
        )
        requests = torch.zeros(rows, dtype=torch.int32, device=self.device)
        x = torch.empty(rows, 1, device=self.device)
        shared = {
            "global": {2: [(None, indexer.PrefillIndexerKeys(kp, ks))]},
            "ced_indexer_projection": lambda value, weight: weights,
        }
        owner = SimpleNamespace(
            is_index_source=True,
            index_source_layer_id=2,
            kv_source_layer_id=2,
            layer_id=2,
            index_topk=512,
            index_n_heads=32,
            index_head_dim=128,
            rope_head_dim=0,
            compress_ratio=2,
            index_wq=None,
            index_weights=None,
            freqs_cis=torch.empty(1, 1, dtype=torch.complex64, device=self.device),
            _lin=lambda weight, value: q,
            v41_config={},
            _shared_attention=shared,
        )

        # Only the projection/quantization stage is supplied by this fixture.
        # Production loop, bounds, metadata, scorer, selector and lifetime run.
        def prepared(*args, **kwargs):
            return qp.unsqueeze(1), qs.unsqueeze(1), weights.unsqueeze(1)

        def invoke(enabled, check_lifetime=False):
            calls, previous = [], None
            real_score = indexer.score_indexer_chunk

            def tracked(*args, **kwargs):
                nonlocal previous
                if check_lifetime and previous is not None:
                    self.assertIsNone(
                        previous(), "Previous logits survived into the next allocation"
                    )
                elif not enabled and previous is not None:
                    self.assertIsNotNone(
                        previous(),
                        "OFF must retain the original overlapping allocation baseline",
                    )
                logits = real_score(*args, **kwargs)
                previous = weakref.ref(logits)
                calls.append((args[0].shape[0], kwargs.get("use_schedule", False)))
                return logits

            with patch.dict(os.environ, {ENV: str(int(enabled))}), patch.object(
                attention.indexer_q_fusion, "try_fused_indexer_q", new=prepared
            ), patch.object(indexer, "score_indexer_chunk", new=tracked):
                output = attention.AttentionV41FP8._select_indices(
                    owner, x, q, positions, requests
                )
            self.assertEqual(
                calls,
                (
                    [(512, True), (1, True)]
                    if enabled
                    else [(128, False)] * 4 + [(1, False)]
                ),
            )
            shared.pop("topk")
            return output

        baseline_ids = invoke(False)
        candidate_ids = invoke(True, True)
        # Tie membership may differ with chunk shape: compare native score
        # multisets for every row, while also checking uniqueness and bounds.
        ends = ((positions + 1) // 2).int()
        for start in range(0, rows, 128):
            stop = min(rows, start + 128)
            logits = indexer.score_indexer_chunk(
                qp[start:stop],
                qs[start:stop],
                kp,
                ks,
                weights[start:stop],
                ends[start:stop],
                return_bf16=True,
            )
            pairs = []
            for chosen in (baseline_ids[start:stop], candidate_ids[start:stop]):
                valid = chosen >= 0
                self.assertTrue(
                    bool(((chosen < ends[start:stop, None]) | ~valid).all())
                )
                sorted_ids = chosen.sort(1).values
                self.assertFalse(
                    bool(
                        (
                            (sorted_ids[:, 1:] == sorted_ids[:, :-1])
                            & (sorted_ids[:, 1:] >= 0)
                        ).any()
                    )
                )
                pairs.append(
                    logits.gather(1, chosen.clamp_min(0).long())
                    .masked_fill(~valid, -torch.inf)
                    .sort(1)
                    .values
                )
            torch.testing.assert_close(*pairs, rtol=0, atol=0)
            del logits, pairs
        del baseline_ids, candidate_ids
        peaks = []
        for enabled in (False, True):
            for _ in range(2):
                output = invoke(enabled)
                del output
            torch.cuda.synchronize()
            gc.collect()
            before = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            output = invoke(enabled, enabled)
            torch.cuda.synchronize()
            peaks.append(torch.cuda.max_memory_allocated() - before)
            del output
            torch.cuda.synchronize()
            self.assertEqual(torch.cuda.memory_allocated(), before)
        self.assertLessEqual(peaks[1] - peaks[0], 1 << 30)
        self.assertGreater(peaks[1], 512 * MIB)  # The real long BF16 allocation ran.
        print(
            json.dumps(
                {
                    "selection_loop_peak_bytes": peaks,
                    "increment_bytes": peaks[1] - peaks[0],
                    "scope": "513 query rows, full 524224-key width, real production selection loop",
                }
            )
        )


if __name__ == "__main__":
    unittest.main()
