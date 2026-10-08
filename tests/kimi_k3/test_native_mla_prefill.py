"""Native BF16 MLA arithmetic, ragged boundaries and 64K prefill checks."""
import unittest

import torch

from rtp_llm.models_py.modules.kimi_k3.native_mla_prefill import KimiK3TokenspeedPrefill


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class NativeMlaPrefillTest(unittest.TestCase):
    def plan(self, q_lens, k_lens):
        op = KimiK3TokenspeedPrefill()
        q = torch.tensor([0]+q_lens, dtype=torch.int32, device='cuda').cumsum(0).int()
        k = torch.tensor([0]+k_lens, dtype=torch.int32, device='cuda').cumsum(0).int()
        op.plan(q,k,16,16,192,128,sm_scale=192**-.5,causal=True,
                q_data_type=torch.bfloat16,kv_data_type=torch.bfloat16)
        return op

    def test_batch_and_prefix_lengths(self):
        torch.manual_seed(6501)
        lengths = [17,63,64,65,127,128,129,257,511]
        tails = [1,2,3,7,8,9,17,64,65]
        q = [(torch.randn(n,16,192,device='cuda')*.125).bfloat16() for n in lengths]
        k = [(torch.randn(n,16,192,device='cuda')*.125).bfloat16() for n in lengths]
        # Preserve a split-view V layout; the production adapter must copy it.
        packed = [(torch.randn(n,16,256,device='cuda')*.025).bfloat16() for n in lengths]
        v = [x[...,128:] for x in packed]
        singles = [self.plan([n],[n]).run(qi,ki,vi) for n,qi,ki,vi in zip(lengths,q,k,v)]
        for batch in (1,2,3,7,8,9):
            with self.subTest(batch=batch):
                full = self.plan(lengths[:batch],lengths[:batch]).run(
                    torch.cat(q[:batch]),torch.cat(k[:batch]),torch.cat(v[:batch]))
                self.assertTrue(torch.equal(full,torch.cat(singles[:batch])))
                reused = self.plan(tails[:batch],lengths[:batch]).run(
                    torch.cat([qi[-n:] for qi,n in zip(q[:batch],tails[:batch])]),
                    torch.cat(k[:batch]),torch.cat(v[:batch]))
                expected = torch.cat([out[-n:] for out,n in zip(singles[:batch],tails[:batch])])
                self.assertTrue(torch.equal(reused,expected))

    def test_64k_full_and_reused_suffix_against_uniform_oracle(self):
        torch.manual_seed(6502)
        n, tail = 65536, 7
        q = torch.zeros(n,16,192,device='cuda',dtype=torch.bfloat16)
        k = torch.zeros_like(q)
        v = torch.randint(-2,3,(n,16,128),device='cuda',dtype=torch.int8).to(torch.bfloat16)/8
        full = self.plan([n],[n]).run(q,k,v)
        reused = self.plan([tail],[n]).run(q[-tail:],k,v)
        self.assertTrue(torch.equal(full[-tail:],reused))
        # Q=0 gives uniform causal probabilities. At 65536 (a power of two),
        # the dyadic V sum/division is exact in FP32 before final BF16 rounding.
        reference = v.double().sum(0)/n
        self.assertTrue(torch.equal(full[-1],reference.bfloat16()))
        self.assertTrue(torch.isfinite(full).all())
        print('NATIVE_MLA_64K_PASS full=65536 reuse=65529+7 dtype=bf16',flush=True)

    def test_zero_length_padding_does_not_change_real_rows(self):
        torch.manual_seed(6503)
        q = torch.randn(9,16,192,device='cuda',dtype=torch.bfloat16)
        k = torch.randn(129,16,192,device='cuda',dtype=torch.bfloat16)
        v = torch.randn(129,16,128,device='cuda',dtype=torch.bfloat16)
        one = self.plan([9],[129]).run(q,k,v)
        padded = self.plan([0,9,0],[0,129,0]).run(q,k,v)
        self.assertTrue(torch.equal(one,padded))
        empty = self.plan([0],[0]).run(q[:0],k[:0],v[:0])
        self.assertEqual(tuple(empty.shape),(0,16,128))

    def test_invalid_metadata_and_precision_rejected(self):
        with self.assertRaisesRegex(ValueError,'Q <= KV'):
            self.plan([2],[1])
        op = self.plan([1],[1])
        q = torch.zeros(1,16,192,device='cuda',dtype=torch.float32)
        k, v = q.bfloat16(), q[:,:,:128].bfloat16()
        with self.assertRaisesRegex(ValueError,'dtype/shape'):
            op.run(q,k,v)


if __name__ == '__main__':
    unittest.main(verbosity=2)
