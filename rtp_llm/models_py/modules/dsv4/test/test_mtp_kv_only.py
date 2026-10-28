import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
import torch.nn as nn
from rtp_llm.models_py.modules.dsv4.fp8 import attention as attention_module
from rtp_llm.models_py.modules.dsv4.fp8.attention import AttentionFP8
from rtp_llm.models_py.modules.dsv4.block import Block

class KVOnlyTests(unittest.TestCase):
    def test_projection_matches_and_query_is_skipped(self):
        for shared in [False, True]:
            for tokens in [1, 7, 141, 1024]:
                with self.subTest(shared=shared,tokens=tokens):
                    calls=[]
                    class Linear:
                        def __init__(self,name):self.name=name
                        def quantize_input(self,x):calls.append('quant');return (x.clone(),None)
                    q,k=Linear('q'),Linear('k')
                    def linear(w,x):calls.append(w.name);return x[:,:,:8].clone()*(2 if w is k else 3)
                    obj=SimpleNamespace(wq_a=q,wkv=k,q_lora_rank=8,q_norm=None,kv_norm=None,eps=1e-6,
                        _can_reuse_qkv_input_quant=lambda:shared,_lin=linear,
                        _lin_from_shared_quant=lambda w,s,shape:linear(w,s[0].reshape(shape)),
                        _rmsnorm_weighted=lambda x,w:x)
                    common=SimpleNamespace(rd=4,freqs_cis=None,cp_on=False)
                    x=torch.randn(tokens,16)
                    with patch.object(attention_module,'fused_rmsnorm_rope',lambda x,*a,**kw:x+1):
                        normal=AttentionFP8._prefill_compute_qkv(obj,x,common)
                        calls.clear()
                        commit=AttentionFP8._prefill_compute_qkv(obj,x,common,kv_only=True)
                    self.assertTrue(torch.equal(normal.kv_full,commit.kv_full))
                    self.assertEqual(commit.qr.numel(),0)
                    self.assertNotIn('q',calls)
                    self.assertEqual(calls.count('k'),1)
                    self.assertEqual(calls.count('quant'),int(shared))

    def test_block_commit_never_runs_attention_output_or_moe(self):
        class A(AttentionFP8):
            def __init__(self):nn.Module.__init__(self);self.compress_ratio=0;self.calls=[]
            def forward_prefill_kv_only(self,x,positions,**kwargs):self.calls.append(x.clone())
        a=A();h=torch.randn(17,4,16)
        obj=SimpleNamespace(attn=a,attn_hc=SimpleNamespace(pre=lambda x:(x.sum(1),None,None)),attn_norm=nn.Identity(),
                            prefill_fast_attn_pre=lambda x:(x,x.sum(1),None,None))
        for fast in [False,True]:
            out=Block.forward_prefill_kv_only(obj,h,None,torch.arange(17),None,use_fast_path=fast)
            self.assertIs(out,h)
        self.assertEqual(len(a.calls),2)
        self.assertTrue(torch.equal(a.calls[0],a.calls[1]))

if __name__=='__main__':unittest.main()
