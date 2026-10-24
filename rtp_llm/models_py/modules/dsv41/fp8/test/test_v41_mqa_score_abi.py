"""CPU call-contract tests; these do not stand in for native kernel tests."""

import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.modules.dsv41.fp8 import _indexer_score, _v41_mqa_layout


def load_backend(contiguous, paged):
    vendor = SimpleNamespace(
        fp8_fp4_mqa_logits=contiguous,
        fp8_fp4_paged_mqa_logits=paged,
        get_paged_mqa_logits_metadata=Mock(return_value=torch.zeros(2, 2)),
    )
    spec = importlib.util.spec_from_file_location(
        "rtp_llm.models_py.modules.dsv41.fp8._mqa_score_abi_case",
        _indexer_score.__file__,
    )
    backend = importlib.util.module_from_spec(spec)
    with patch.dict("sys.modules", {"deep_gemm": vendor}):
        spec.loader.exec_module(backend)
    return backend, vendor


class MqaScoreAbiTest(unittest.TestCase):
    def check_contiguous(self, legacy):
        calls = []
        result = torch.zeros(2, 11, dtype=torch.bfloat16)

        def record(q, kv, weights, starts, ends, width):
            self.assertEqual(weights.dtype, torch.bfloat16)
            self.assertTrue(weights.is_contiguous())
            self.assertEqual(q[1].dtype, torch.int32)
            self.assertEqual(kv[1].dtype, torch.int32)
            calls.append(width)
            return result

        if legacy:

            def operation(
                q,
                kv,
                weights,
                starts,
                ends,
                clean_logits=True,
                max_seqlen_k=0,
                logits_dtype=torch.float32,
                schedule_meta=None,
            ):
                """clean_logits: bool, max_seqlen_k: int, logits_dtype: torch.dtype"""
                self.assertIs(clean_logits, False)
                self.assertEqual(logits_dtype, torch.bfloat16)
                self.assertIsNone(schedule_meta)
                return record(q, kv, weights, starts, ends, max_seqlen_k)

        else:

            def operation(q, kv, weights, starts, ends, *, max_seqlen_k):
                """max_seqlen_k: int"""
                return record(q, kv, weights, starts, ends, max_seqlen_k)

        backend, _ = load_backend(operation, None)
        starts = torch.tensor([2, 5], dtype=torch.int32)
        ends = torch.tensor([7, 11], dtype=torch.int32)
        with patch.object(
            _v41_mqa_layout, "normalize_mqa_logits", return_value=result
        ) as normalize:
            for width in (0, 6):
                actual = backend.fp8_fp4_mqa_indexer_score(
                    torch.zeros(2, 32, 64, dtype=torch.int8),
                    torch.zeros(2, 32, dtype=torch.int32),
                    torch.zeros(11, 64, dtype=torch.int8),
                    torch.zeros(11, dtype=torch.int32),
                    torch.ones(2, 32),
                    starts,
                    ends,
                    max_seqlen_k=width,
                )
                self.assertIs(actual, result)
                self.assertIs(normalize.call_args.args[0], result)
                self.assertEqual(normalize.call_args.args[3], width or 11)
                self.assertEqual(normalize.call_args.kwargs, {"relative": width > 0})
        self.assertEqual(calls, [11, 6])

    def test_legacy_contiguous_uses_compressed_bf16_keywords(self):
        self.check_contiguous(True)

    def test_current_contiguous_rejects_removed_keywords(self):
        self.check_contiguous(False)

    def check_paged(self, legacy):
        calls = []
        result = torch.zeros(6, 256, dtype=torch.bfloat16)

        def record(q, cache, weights, lengths, tables, schedule, width, indices):
            self.assertEqual(tuple(q[0].shape), (6, 1, 32, 64))
            self.assertEqual(tuple(q[1].shape), (6, 1, 32))
            self.assertEqual(weights.dtype, torch.bfloat16)
            self.assertEqual(tuple(lengths.shape), (6, 1))
            self.assertEqual(tuple(cache.shape), (5, 128, 1, 68))
            self.assertEqual(tables.tolist(), [[1, 2]] * 3 + [[3, 4]] * 3)
            self.assertEqual(indices.tolist(), [0, 0, 0, 1, 1, 1])
            calls.append(width)
            return result

        if legacy:

            def operation(
                q,
                cache,
                weights,
                lengths,
                tables,
                schedule,
                width,
                clean_logits=False,
                logits_dtype=torch.float32,
                indices=None,
            ):
                """clean_logits: bool, logits_dtype: torch.dtype, indices: Tensor"""
                self.assertIs(clean_logits, False)
                self.assertEqual(logits_dtype, torch.bfloat16)
                return record(
                    q, cache, weights, lengths, tables, schedule, width, indices
                )

        else:

            def operation(
                q, cache, weights, lengths, tables, schedule, width, *, indices
            ):
                """indices: Tensor"""
                return record(
                    q, cache, weights, lengths, tables, schedule, width, indices
                )

        backend, vendor = load_backend(None, operation)
        with patch.object(backend, "_get_num_sms", return_value=148), patch.object(
            _v41_mqa_layout, "normalize_mqa_logits", return_value=result
        ) as normalize:
            actual = backend.fp8_fp4_paged_indexer_score(
                torch.zeros(2, 3, 32, 64, dtype=torch.int8),
                torch.zeros(2, 3, 32, dtype=torch.int32),
                torch.zeros(5, 128, 68, dtype=torch.uint8),
                torch.ones(6, 32),
                torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
                torch.tensor([[1, 2, 3], [130, 131, 132]], dtype=torch.int32),
                128,
                256,
            )
        self.assertIs(actual, result)
        self.assertEqual(calls, [256])
        self.assertEqual(vendor.get_paged_mqa_logits_metadata.call_count, 1)
        self.assertEqual(normalize.call_args.kwargs, {"relative": True})
        self.assertEqual(normalize.call_args.args[1].tolist(), [0] * 6)
        self.assertEqual(normalize.call_args.args[2].tolist(), [1, 2, 3, 130, 131, 132])

    def test_legacy_paged_uses_bf16_and_varlen_metadata(self):
        self.check_paged(True)

    def test_current_paged_rejects_removed_keywords(self):
        self.check_paged(False)

    def test_native_failure_is_not_retried(self):
        operation = Mock(side_effect=RuntimeError("native launch failure"))
        backend, _ = load_backend(operation, None)
        with self.assertRaisesRegex(RuntimeError, "native launch failure"):
            backend.fp8_fp4_mqa_indexer_score(
                torch.zeros(1, 32, 64, dtype=torch.int8),
                torch.zeros(1, 32, dtype=torch.int32),
                torch.zeros(1, 64, dtype=torch.int8),
                torch.zeros(1, dtype=torch.int32),
                torch.ones(1, 32),
                torch.zeros(1, dtype=torch.int32),
                torch.ones(1, dtype=torch.int32),
            )
        self.assertEqual(operation.call_count, 1)


if __name__ == "__main__":
    unittest.main()
