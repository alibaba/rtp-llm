"""Constructed BF16 sparse/latent-merge contract over the shared FlashMLA API."""
import unittest
import torch


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class SharedSfaContractTest(unittest.TestCase):
    def test_full_heads_and_latent_chunk_merge(self):
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_impl import flash_mla_sparse_fwd
        from rtp_llm.models_py.kernels.cuda.dsa_kv_chunk import normalize_sparse_partial
        from rtp_llm.models_py.triton_kernels.mla.lse_merge import merge_attention_states_in_place

        torch.manual_seed(3101)
        torch.backends.cuda.matmul.allow_tf32=False
        rows,heads,latent,suffix,topk,history=4,96,512,64,2048,4096
        q=torch.randn(rows,128,latent+suffix,device='cuda',dtype=torch.bfloat16)*.05
        q[:,heads:]=0
        kv=torch.randn(history,1,latent+suffix,device='cuda',dtype=torch.bfloat16)*.05
        indices=torch.stack([torch.randperm(history,device='cuda')[:topk] for _ in range(rows)]).to(torch.int32)
        positions=torch.tensor([0,2048,3000,4095],device='cuda')
        indices.masked_fill_(indices>positions[:,None],-1)
        indices[0]=-1
        scale=192**-.5
        valid=indices>=0
        selected=kv[indices.clamp_min(0).long(),0].float()
        logits=torch.einsum('qhd,qkd->qhk',q[:,:heads].float(),selected)*scale
        logits.masked_fill_(~valid[:,None,:],-torch.inf)
        reference_lse=torch.logsumexp(logits,dim=-1)
        probabilities=torch.where(valid.any(-1)[:,None,None],logits.softmax(-1),0)
        reference=torch.einsum('qhk,qkd->qhd',probabilities,selected[:,:,:latent])
        full,_,full_lse=flash_mla_sparse_fwd(q,kv,indices[:,None,:],scale,d_v=latent)
        canonical_lse=torch.empty(rows,heads,device='cuda')
        normalize_sparse_partial(full,full_lse,heads,canonical_lse)
        torch.testing.assert_close(full[:,:heads].float(),reference,rtol=.03,atol=.0005)
        torch.testing.assert_close(canonical_lse,reference_lse,rtol=2e-5,atol=2e-5)
        running=torch.zeros(rows,heads,latent,device='cuda')
        running_lse=torch.full((rows,heads),-torch.inf,device='cuda')
        consumed=torch.zeros_like(indices)
        for start in (0,2048):
            member=(indices>=start)&(indices<start+2048)
            local=torch.where(member,indices-start,-1).contiguous()
            consumed+=member.int()
            part,_,lse=flash_mla_sparse_fwd(q,kv[start:start+2048].contiguous(),local[:,None,:],scale,d_v=latent)
            merge_attention_states_in_place(running,running_lse,part[:,:heads],lse[:,:heads],raw_partial_lse=True)
        torch.testing.assert_close(consumed,valid.int(),rtol=0,atol=0)
        torch.testing.assert_close(running,reference,rtol=.03,atol=.0005)
        torch.testing.assert_close(running_lse,reference_lse,rtol=2e-5,atol=2e-5)
        weight=torch.randn(heads,latent,128,device='cuda',dtype=torch.bfloat16)*.02
        # Exactly one V-up after the complete latent merge.
        actual=torch.bmm(running.to(torch.bfloat16).transpose(0,1),weight).transpose(0,1)
        expected=torch.bmm(reference.to(torch.bfloat16).transpose(0,1),weight).transpose(0,1)
        torch.testing.assert_close(actual,expected,rtol=.03,atol=.0005)
        self.assertTrue((actual[0]==0).all())
        self.assertTrue(torch.isneginf(running_lse[0]).all())


if __name__=='__main__':unittest.main()
