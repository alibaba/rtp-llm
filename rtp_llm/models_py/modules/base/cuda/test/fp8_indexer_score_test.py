"""CPU interface tests; native FP8 math/Graph requires the separate GPU gate."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.modules.base.cuda import fp8_indexer_score as score_module


class NativeFp8IndexerScoreContractTest(unittest.TestCase):
    def backend(self):
        native = SimpleNamespace(fp8_mqa_logits=Mock(), fp8_paged_mqa_logits=Mock(),
                                 get_paged_mqa_logits_metadata=Mock())
        return score_module.NativeFp8IndexerScore(native), native

    def test_missing_native_entry_does_not_fall_back(self):
        with self.assertRaisesRegex(RuntimeError, 'requires native FP8 MQA'):
            score_module.NativeFp8IndexerScore(SimpleNamespace())

    def test_cpu_or_bf16_query_never_reaches_native_score(self):
        score, native = self.backend()
        with self.assertRaises(ValueError):
            score.dense(torch.zeros(4, 32, 128, dtype=torch.bfloat16),
                        None, None, torch.zeros(4, 32), None, None, None)
        native.fp8_mqa_logits.assert_not_called()

    def test_prepare_requires_per_query_cuda_lengths(self):
        score, native = self.backend()
        with self.assertRaisesRegex(ValueError, 'Prepare CUDA'):
            score.prepare_paged_metadata(torch.zeros(2, dtype=torch.int32), num_sms=148)
        native.get_paged_mqa_logits_metadata.assert_not_called()

    def test_explicit_output_owns_scores_and_preserves_the_native_result(self):
        score, _ = self.backend()
        raw = torch.arange(8, dtype=torch.float32).view(2, 4)
        before = raw.clone()
        out = torch.empty_like(raw)
        positions = torch.arange(4, dtype=torch.int32)
        starts = torch.tensor([0, 2], dtype=torch.int32)
        ends = torch.tensor([1, 4], dtype=torch.int32)

        def reference_mask(value, columns, *, starts, ends):
            valid = ((torch.arange(4)[None, :] >= starts[:, None])
                     & (torch.arange(4)[None, :] < ends[:, None]))
            value.masked_fill_(~valid, -float('inf'))

        with patch.object(score_module, 'sanitize_indexer_scores', side_effect=reference_mask):
            actual = score._finish(raw, positions, starts, ends, (2, 4), out)
        self.assertIs(actual, out)
        torch.testing.assert_close(raw, before, rtol=0, atol=0)
        torch.testing.assert_close(actual, torch.tensor([[0., -float('inf'), -float('inf'), -float('inf')],
                                                        [-float('inf'), -float('inf'), 6., 7.]]),
                                   rtol=0, atol=0)

    def test_wrong_score_or_position_precision_is_rejected(self):
        score, _ = self.backend()
        with self.assertRaisesRegex(RuntimeError, 'incompatible score'):
            score._finish(torch.zeros(2, 4, dtype=torch.bfloat16),
                          torch.arange(4, dtype=torch.int32), None, None, (2, 4), None)
        with self.assertRaisesRegex(ValueError, 'INT32 global positions'):
            score._finish(torch.zeros(2, 4), torch.arange(4, dtype=torch.int64),
                          None, None, (2, 4), None)

    def test_owned_strided_result_is_masked_without_copy(self):
        score, _ = self.backend()
        storage = torch.arange(16, dtype=torch.float32).view(2, 8)
        raw = storage[:, :4]
        positions = torch.arange(4, dtype=torch.int32)
        starts = torch.tensor([0, 2], dtype=torch.int32)
        ends = torch.tensor([1, 4], dtype=torch.int32)

        def mask(value, positions, *, starts, ends):
            col = torch.arange(4)[None, :]
            value.masked_fill_((col < starts[:, None]) | (col >= ends[:, None]),
                               -float('inf'))

        with patch.object(score_module, 'sanitize_indexer_scores', side_effect=mask) as sanitize:
            actual = score._finish(raw, positions, starts, ends, (2, 4), None)
        self.assertIs(actual, raw)
        self.assertEqual(actual.data_ptr(), storage.data_ptr())
        self.assertEqual(actual.stride(), (8, 1))
        sanitize.assert_called_once()
        torch.testing.assert_close(actual, torch.tensor([[0., -float('inf'), -float('inf'), -float('inf')],
                                                         [-float('inf'), -float('inf'), 10., 11.]]),
                                   rtol=0, atol=0)
        torch.testing.assert_close(storage[:, 4:], torch.tensor([[4., 5., 6., 7.],
                                                                [12., 13., 14., 15.]]),
                                   rtol=0, atol=0)

    def test_original_128_token_page_cannot_be_reinterpreted_as_native64(self):
        score, native = self.backend()
        with patch.object(score, '_query', return_value=2):
            with self.assertRaisesRegex(ValueError, 'explicit64-token'):
                score.paged(torch.zeros(2, 1, 32, 128, dtype=torch.float8_e4m3fn),
                            torch.zeros(2, 128, 1, 132, dtype=torch.uint8),
                            torch.zeros(2, 32), torch.zeros(2, dtype=torch.int32),
                            torch.ones(2, 1, dtype=torch.int32), None, 128,
                            torch.arange(128, dtype=torch.int32))
        native.fp8_paged_mqa_logits.assert_not_called()

    def test_nonzero_request_starts_use_absolute_native_output_columns(self):
        score, native = self.backend()
        native.fp8_mqa_logits.return_value = torch.arange(16, dtype=torch.float32).view(2, 8)
        starts = torch.tensor([0, 4], dtype=torch.int32)
        ends = torch.tensor([3, 7], dtype=torch.int32)

        def reference_mask(value, positions, *, starts, ends):
            col = torch.arange(8)[None, :]
            value.masked_fill_((col < starts[:, None]) | (col >= ends[:, None]), -float('inf'))

        with patch.object(score, '_query', return_value=2), patch.object(
                score_module, 'sanitize_indexer_scores', side_effect=reference_mask):
            actual = score.dense(torch.zeros(2, 32, 128, dtype=torch.float8_e4m3fn),
                                 torch.zeros(8, 128, dtype=torch.float8_e4m3fn), torch.ones(8),
                                 torch.ones(2, 32), starts, ends, torch.arange(8, dtype=torch.int32))
        self.assertEqual(native.fp8_mqa_logits.call_args.kwargs['max_seqlen_k'], 0)
        self.assertTrue(torch.isneginf(actual[1, :4]).all())
        torch.testing.assert_close(actual[1, 4:7], torch.tensor([12., 13., 14.]), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
