"""GPU regression: partial acceptance, padded pool stride, slot reuse and graphs."""
import torch
from atom.model_ops.fla_ops.gdn_flydsl import mtp_rtp
from aiter.ops.flydsl.linear_attention_kernels import flydsl_gdr_mtp_hybrid
from atom.plugin.rtpllm.utils.replay_hybrid import _commit

torch.manual_seed(123)
for n in (1,2,4,8,16):
 qlen,hk,hv,d=3,8,24,128
 q=torch.randn(n,qlen,hk,d,device='cuda',dtype=torch.bfloat16);k=torch.randn_like(q)
 v=torch.randn(n,qlen,hv,d,device='cuda',dtype=torch.bfloat16)
 a=torch.randn(n,qlen,hv,device='cuda',dtype=torch.bfloat16);b=torch.randn_like(a)
 al=torch.randn(hv,device='cuda');bias=torch.randn(hv,device='cuda',dtype=torch.bfloat16)
 initial=torch.randn(n*4,hv,d,d,device='cuda')*.1
 ref=initial.clone()
 backing=torch.full((n*4,hv*d*d+512),17.5,device="cuda",dtype=torch.float32)
 candidate=backing[:,:hv*d*d].view(n*4,hv,d,d).copy_(initial)
 writes=torch.arange(n,4*n,device='cuda',dtype=torch.int32).view(n,3)
 reads=torch.arange(n,device='cuda',dtype=torch.int32)
 read_backing=torch.zeros(n,3,device='cuda',dtype=torch.int32);read_view=read_backing[:,0]
 out=torch.empty(n,qlen,hv,d,device='cuda',dtype=torch.bfloat16);expected=torch.empty_like(out)
 records=torch.empty(1,n,qlen,hv,3,d,device='cuda');ptrs=torch.tensor([candidate.data_ptr()],device='cuda',dtype=torch.int64)
 accepted=torch.arange(n,device='cuda',dtype=torch.int32)%4
 def base():
  read_view.copy_(reads)
  mtp_rtp(q.view(1,n*qlen,hk,d),k.view(1,n*qlen,hk,d),v.view(1,n*qlen,hv,d),a.view(n*qlen,hv),b.view(n*qlen,hv),ref.transpose(-1,-2),al,bias,read_view,writes,expected.view(n*qlen,hv,d))
 def hybrid():
  read_view.copy_(reads)
  flydsl_gdr_mtp_hybrid(q,k,v,a,b,bias,al,candidate,out,read_view,writes,records[0])
 def commit():_commit[(n,hv,4)](records,ptrs,reads,writes,accepted,hv,d,qlen,n,32,candidate.stride(0),num_warps=4)
 hybrid();commit();torch.cuda.synchronize()
 graph=torch.cuda.CUDAGraph()
 with torch.cuda.graph(graph):hybrid();commit()
 candidate.copy_(initial)
 for step in range(6):
  accepted.copy_((torch.arange(n,device='cuda',dtype=torch.int32)+step)%4)
  before=ref[reads.long()].clone()
  base();gold=ref[writes.gather(1,(accepted.clamp(min=1)-1)[:,None].long()).squeeze(1).long()].clone()
  gold=torch.where((accepted>0)[:,None,None,None],gold,before)
  graph.replay();torch.cuda.synchronize()
  torch.testing.assert_close(out,expected,atol=.002,rtol=.005)
  selected=torch.where(accepted>0,writes.gather(1,(accepted.clamp(min=1)-1)[:,None].long()).squeeze(1),reads)
  torch.testing.assert_close(candidate[selected.long()],gold,atol=.0001,rtol=.001)
  # The baseline writes speculative states even for accept=0; restore its committed checkpoint.
  ref[selected.long()]=gold
  reads.copy_(selected)
  torch.testing.assert_close(backing[:,hv*d*d:],torch.full_like(backing[:,hv*d*d:],17.5),atol=0,rtol=0)
  print('PASS',n,step,'accept',accepted.tolist(),flush=True)
print('ALL PASS')
