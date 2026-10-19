"""CPU contract tests for the x86/ARM DeepGEMM argument adapter (no CUDA)."""
import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import torch


class IndexerApiTest(unittest.TestCase):
    def load(self, legacy):
        self.calls = []
        def prefill(*args, **kwargs):
            self.calls.append((args, kwargs))
            return torch.zeros((2, 8), dtype=torch.bfloat16)
        def paged(*args, **kwargs):
            self.calls.append((args, kwargs))
            return torch.zeros((2, 8), dtype=torch.bfloat16)
        prefill.__doc__ = paged.__doc__ = 'logits_dtype: torch.dtype' if legacy else ''
        dg = SimpleNamespace(fp8_fp4_mqa_logits=prefill, fp8_fp4_paged_mqa_logits=paged,
                             get_paged_mqa_logits_metadata=lambda *a, **kw: torch.zeros((2, 2)))
        quant = SimpleNamespace(INDEXER_ENTRY_BYTES=132, INDEXER_HEAD_DIM=128)
        normalize = SimpleNamespace(normalize_mqa_logits=lambda x, *a, **kw: x.float())
        path = Path(__file__).parents[2] / 'modules/dsv41/fp8/_indexer_score.py'
        spec = importlib.util.spec_from_file_location('mqa_api_test._indexer_score', path)
        mod = importlib.util.module_from_spec(spec)
        modules = {'deep_gemm': dg, 'rtp_llm.models_py.modules.dsv41.fp8._indexer_quant_triton': quant,
                   'mqa_api_test': ModuleType('mqa_api_test'), 'mqa_api_test._v41_mqa_layout': normalize}
        self.ctx = patch.dict(sys.modules, modules)
        self.ctx.start(); self.addCleanup(self.ctx.stop)
        spec.loader.exec_module(mod)
        mod._get_num_sms = lambda device: 148
        return mod

    def test_prefill_legacy_and_new_api(self):
        for legacy in (True, False):
            with self.subTest(legacy=legacy):
                mod = self.load(legacy)
                result = mod.fp8_fp4_mqa_indexer_score(
                    torch.zeros((2, 4, 64), dtype=torch.int8), torch.zeros((2, 4), dtype=torch.int32),
                    torch.zeros((8, 64), dtype=torch.int8), torch.zeros(8, dtype=torch.int32),
                    torch.ones((2, 4)), torch.zeros(2, dtype=torch.int32), torch.ones(2, dtype=torch.int32)*8,
                    max_seqlen_k=8)
                args, kw = self.calls[-1]
                self.assertEqual(args[2].dtype, torch.bfloat16)
                self.assertEqual(kw['max_seqlen_k'], 8)
                self.assertEqual('logits_dtype' in kw, legacy)
                if legacy:
                    self.assertEqual(kw['logits_dtype'], torch.bfloat16)
                    self.assertIs(kw['clean_logits'], False)
                self.assertEqual(result.dtype, torch.float32)
                self.ctx.stop()

    def test_paged_legacy_and_new_api(self):
        for legacy in (True, False):
            with self.subTest(legacy=legacy):
                mod = self.load(legacy)
                mod.fp8_fp4_paged_indexer_score(
                    torch.zeros((1, 2, 4, 64), dtype=torch.int8), torch.zeros((1, 2, 4), dtype=torch.int32),
                    torch.zeros((1, 128, 68), dtype=torch.uint8), torch.ones((2, 4)),
                    torch.zeros((1, 1), dtype=torch.int32), torch.ones((1, 2), dtype=torch.int32)*8, 128, 8)
                args, kw = self.calls[-1]
                self.assertEqual(args[2].dtype, torch.bfloat16)
                self.assertEqual(args[0][0].shape, (2, 1, 4, 64))
                self.assertEqual(kw['indices'].tolist(), [0, 0])
                self.assertEqual('logits_dtype' in kw, legacy)
                if legacy:
                    self.assertEqual(kw['logits_dtype'], torch.bfloat16)
                    self.assertIs(kw['clean_logits'], False)
                self.ctx.stop()


if __name__ == '__main__':
    unittest.main()
