"""GPU regression for independent MTP read slots and accepted-prefix continuation.

Run after applying the pinned ATOM and AITER patches.
"""
import torch
from atom.model_ops.attention_gdn import fused_gdn_gating
from atom.model_ops.fla_ops.fused_recurrent import fused_recurrent_gated_delta_rule
from atom.model_ops.fla_ops.gdn_flydsl import mtp_rtp

torch.manual_seed(51)
for batch in (2,4):
 for width in (2,3,4):
  hk,hv,k,v=2,4,128,128
  q=torch.randn(1,batch*width,hk,k,device='cuda',dtype=torch.bfloat16)
  key=torch.randn_like(q);value=torch.randn(1,batch*width,hv,v,device='cuda',dtype=torch.bfloat16)
  a=torch.randn(batch*width,hv,device='cuda',dtype=torch.bfloat16);b=torch.randn_like(a)
  alog=torch.randn(hv,device='cuda');bias=torch.randn(hv,device='cuda',dtype=torch.bfloat16)
  slots=batch*(width+1)
  state=torch.randn(slots,hv,k,v,device='cuda')*.1
  ref=state.clone();candidate=state.transpose(-1,-2).contiguous().transpose(-1,-2)
  reads=torch.arange(batch,device='cuda',dtype=torch.int32)[:,None].expand(batch,width).contiguous()
  writes=torch.arange(batch,slots,device='cuda',dtype=torch.int32).view(batch,width)
  lengths=torch.arange(0,(batch+1)*width,width,device='cuda',dtype=torch.int32)
  accepts=torch.ones(batch,device='cuda',dtype=torch.int32)
  g,beta=fused_gdn_gating(alog,a,b,bias)
  expected,_=fused_recurrent_gated_delta_rule(q=q,k=key,v=value,g=g,beta=beta,initial_state=ref,inplace_final_state=True,cu_seqlens=lengths,ssm_state_indices=writes,ssm_state_indices_in=reads,num_accepted_tokens=accepts,use_qk_l2norm_in_kernel=True)
  out=torch.empty(batch*width,hv,v,device='cuda',dtype=torch.bfloat16)
  actual,_=mtp_rtp(q,key,value,a,b,candidate,alog,bias,reads,writes,out)
  torch.cuda.synchronize()
  print(batch,width,'output max',float((actual.float()-expected.float()).abs().max()),'state max',float((candidate-ref).abs().max()),flush=True)
  torch.testing.assert_close(actual,expected,atol=.005,rtol=.01)
  torch.testing.assert_close(candidate,ref,atol=.005,rtol=.01)
  for accepted in range(1,width+1):
   selected=writes[:,accepted-1].contiguous()
   next_reads=selected[:,None].expand_as(writes).contiguous()
   expected,_=fused_recurrent_gated_delta_rule(q=q,k=key,v=value,g=g,beta=beta,initial_state=ref,inplace_final_state=True,cu_seqlens=lengths,ssm_state_indices=writes,ssm_state_indices_in=next_reads,num_accepted_tokens=accepts,use_qk_l2norm_in_kernel=True)
   actual,_=mtp_rtp(q,key,value,a,b,candidate,alog,bias,selected,writes,out)
   torch.cuda.synchronize()
   torch.testing.assert_close(actual,expected,atol=.005,rtol=.01)
   torch.testing.assert_close(candidate,ref,atol=.005,rtol=.01)
   print('continuation',batch,width,accepted,'PASS',flush=True)
print('ALL PASS',flush=True)
