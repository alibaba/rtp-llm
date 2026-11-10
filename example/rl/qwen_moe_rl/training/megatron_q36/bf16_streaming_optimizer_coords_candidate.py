"""Private streaming BF16 Adam candidate with explicit logical SR positions.

Requires an externally validated per-parameter SRCoordinates binding. This
does not qualify real model mapping, native lease, rollback or 72-rank RL.
"""
import math
import torch
from bf16_hybrid_adam_candidate import key
from bf16_sr_coordinates_candidate import SRCoordinates,stochastic_round_coordinates
from bf16_streaming_optimizer_candidate import StreamingBf16AdamW


class StreamingBf16AdamWCoordinates(StreamingBf16AdamW):
    @torch.no_grad()
    def step(self,closure=None):
        if self.poisoned:raise RuntimeError('streaming optimizer is poisoned; discard policy and state')
        if closure is not None:raise ValueError('closure unsupported')
        if self.require_finite_ticket and not self._finite_ticket:raise RuntimeError('missing current all-rank finite ticket')
        self._finite_ticket=False
        try:
            for group in self.param_groups:
                chunk=group['chunk_numel']
                if type(chunk) is not int or chunk<1:raise ValueError('invalid chunk size')
                for p in group['params']:
                    grad=getattr(p,'decoupled_grad',None)
                    if grad is None:continue
                    coord=getattr(p,'_q38_sr_coordinates',None)
                    if not isinstance(coord,SRCoordinates):raise ValueError('missing logical SR binding')
                    coord.validate()
                    if (tuple(p.shape)!=coord.local_shape or p.device.type!='cuda' or p.dtype!=torch.bfloat16
                        or grad.device!=p.device or grad.dtype!=torch.bfloat16 or p.shape!=grad.shape
                        or not p.is_contiguous() or not grad.is_contiguous()):
                        raise ValueError('invalid CUDA BF16 local parameter/gradient shape')
                    if not self.require_finite_ticket:
                        flat=grad.reshape(-1)
                        for start in range(0,flat.numel(),chunk):
                            if not torch.isfinite(flat[start:start+chunk]).all():raise FloatingPointError('nonfinite BF16 gradient')
            for group in self.param_groups:
                beta1,beta2=group['betas']
                for p in group['params']:
                    grad=getattr(p,'decoupled_grad',None)
                    if grad is None:continue
                    state=self.state[p]
                    if not state:
                        state.update(step=0,exp_avg=torch.zeros(p.shape,dtype=torch.bfloat16,device='cpu'),
                                     exp_avg_sq=torch.zeros(p.shape,dtype=torch.bfloat16,device='cpu'))
                    step=state['step']+1;coord=p._q38_sr_coordinates
                    w,g=p.reshape(-1),grad.reshape(-1)
                    m,v=state['exp_avg'].reshape(-1),state['exp_avg_sq'].reshape(-1)
                    keys={role:key(coord.canonical_name,step,role,group['rounding_seed'])
                          for role in ('weight','exp_avg','exp_avg_sq')}
                    for start in range(0,w.numel(),group['chunk_numel']):
                        end=min(w.numel(),start+group['chunk_numel'])
                        weight=w[start:end].to(device='cpu',dtype=torch.float32,non_blocking=False)
                        gradient=g[start:end].to(device='cpu',dtype=torch.float32,non_blocking=False)
                        mean,var=m[start:end].float(),v[start:end].float()
                        weight.mul_(1-group['lr']*group['weight_decay']);mean.lerp_(gradient,1-beta1)
                        var.mul_(beta2).addcmul_(gradient,gradient,value=1-beta2)
                        denom=var.sqrt().div_(math.sqrt(1-beta2**step)).add_(group['eps'])
                        weight.addcdiv_(mean,denom,value=-group['lr']/(1-beta1**step))
                        rw=stochastic_round_coordinates(weight,keys['weight'],coord,start)
                        rm=stochastic_round_coordinates(mean,keys['exp_avg'],coord,start)
                        rv=stochastic_round_coordinates(var,keys['exp_avg_sq'],coord,start)
                        w[start:end].copy_(rw,non_blocking=False);torch.cuda.current_stream(p.device).synchronize()
                        m[start:end].copy_(rm);v[start:end].copy_(rv)
                    state['step']=step
        except BaseException:
            self.poisoned=True
            raise
