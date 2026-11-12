"""Scorer boundary checks: raw SG FP32 and unchanged legacy BF16 routing."""

import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.modules.dsv41.backend import V41QuantizedTensor
from rtp_llm.models_py.modules.dsv41.fp8 import _indexer_score, _v41_prefill_indexer
from rtp_llm.models_py.modules.dsv41.ppu import indexer


class SGScoreIntegrationCPU(unittest.TestCase):
    def test_startup_odd_banks_decline_before_raw_score_routing(self):
        from rtp_llm.models_py.modules.dsv41.ppu.attention_layer import _local_sg_indexer_counts_supported

        for counts in ((1920,), (65536,), (131072,)):
            self.assertTrue(_local_sg_indexer_counts_supported(counts), counts)
        for counts in ((), (0,), (7,), (3841,), (4097,), (131080,)):
            self.assertFalse(_local_sg_indexer_counts_supported(counts), counts)

    @torch.no_grad()
    def test_raw_fp32_keeps_fp32_weights_and_never_normalizes_or_masks(self):
        q = V41QuantizedTensor(torch.zeros(2, 32, 64, dtype=torch.int8),
                               torch.zeros(2, 32, dtype=torch.int32), 'fp4_ue8m0')
        k = V41QuantizedTensor(torch.zeros(1024, 64, dtype=torch.int8),
                               torch.zeros(1024, dtype=torch.int32), 'fp4_ue8m0')
        weights = torch.randn(2, 32)
        positions = torch.tensor([1022, 1023])
        result = torch.empty(2, 1024, dtype=torch.float32)
        operation = Mock(return_value=result)
        operation.__doc__ = 'fp8_fp4_mqa_logits(logits_dtype: torch.dtype)'
        with patch.dict(os.environ, {'RTP_V41_PPU_CP8_DENSE_INDEX_BOUNDS': '0'}), \
             patch.object(_indexer_score, '_deep_gemm', SimpleNamespace(fp8_fp4_mqa_logits=operation)), \
             patch.object(_v41_prefill_indexer, 'score_indexer_chunk', side_effect=AssertionError('compat scorer')), \
             patch.object(torch.Tensor, 'masked_fill_', side_effect=AssertionError('full score mask')):
            self.assertIs(indexer.dense_fp4_index_score(q, k, weights, positions, 1, sg_align=True), result)
        args, kwargs = operation.call_args
        self.assertIs(args[2], weights)
        self.assertEqual(args[2].dtype, torch.float32)
        self.assertTrue(torch.equal(args[3], torch.zeros(2, dtype=torch.int32)))
        self.assertTrue(torch.equal(args[4], torch.tensor([1023, 1024], dtype=torch.int32)))
        self.assertEqual(kwargs, dict(clean_logits=False, max_seqlen_k=1024, logits_dtype=torch.float32))

    @torch.no_grad()
    def test_legacy_keeps_existing_bf16_compatibility_call(self):
        q = V41QuantizedTensor(torch.zeros(2, 32, 64, dtype=torch.int8),
                               torch.zeros(2, 32, dtype=torch.int32), 'fp4_ue8m0')
        k = V41QuantizedTensor(torch.zeros(1024, 64, dtype=torch.int8),
                               torch.zeros(1024, dtype=torch.int32), 'fp4_ue8m0')
        weights, positions = torch.randn(2, 32), torch.tensor([1022, 1023])
        result = torch.ones(2, 1024, dtype=torch.bfloat16)
        with patch.dict(os.environ, {'RTP_V41_PPU_CP8_DENSE_INDEX_BOUNDS': '0'}), \
             patch.object(_v41_prefill_indexer, 'score_indexer_chunk', return_value=result) as old:
            output = indexer.dense_fp4_index_score(q, k, weights, positions, 1)
        self.assertIs(output, result)
        self.assertIs(old.call_args.kwargs['return_bf16'], True)
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertEqual(output[0, -1].item(), -torch.inf)
        self.assertEqual(output[1, -1].item(), 1.0)


@unittest.skipUnless(os.environ.get('RTP_V41_TEST_LOCAL_WORKSPACE_DEVICE') == '1', 'explicit PPU test')
class SGScoreIntegrationDevice(unittest.TestCase):
    @torch.no_grad()
    def test_real_provider_raw_fp32_matches_sg_call_at_visible_columns(self):
        self.assertTrue(torch.cuda.is_available())
        torch.cuda.set_device(0)
        device = torch.device('cuda', 0)
        generator = torch.Generator(device=device).manual_seed(81)
        q = V41QuantizedTensor(
            torch.randint(0, 256, (4, 32, 64), dtype=torch.uint8, device=device,
                          generator=generator).view(torch.int8),
            torch.full((4, 32), 0x7f7f7f7f, dtype=torch.int32, device=device), 'fp4_ue8m0',
        )
        k = V41QuantizedTensor(
            torch.randint(0, 256, (1024, 64), dtype=torch.uint8, device=device,
                          generator=generator).view(torch.int8),
            torch.full((1024,), 0x7f7f7f7f, dtype=torch.int32, device=device), 'fp4_ue8m0',
        )
        weights = torch.rand((4, 32), device=device, generator=generator)
        positions = torch.tensor([127, 255, 511, 1023], device=device)
        ends = (positions + 1).to(torch.int32)
        expected = _indexer_score._deep_gemm.fp8_fp4_mqa_logits(
            (q.values, q.scales), (k.values, k.scales), weights,
            torch.zeros_like(ends), ends, False, 1024, logits_dtype=torch.float32,
        )
        with patch.dict(os.environ, {'RTP_V41_PPU_CP8_DENSE_INDEX_BOUNDS': '0'}):
            actual = indexer.dense_fp4_index_score(q, k, weights, positions, 1, sg_align=True)
        self.assertEqual(actual.dtype, torch.float32)
        self.assertEqual(actual.shape, (4, 1024))
        for row, end in enumerate((128, 256, 512, 1024)):
            torch.testing.assert_close(actual[row, :end], expected[row, :end], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
