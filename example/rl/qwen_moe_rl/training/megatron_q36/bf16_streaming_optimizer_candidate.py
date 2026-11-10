"""Experimental BF16 CUDA shard / CPU BF16 Adam-state streaming optimizer.

The caller owns distributed finite checks, global clipping, synchronization,
stable shard coordinates, and checkpoint/recovery. No full CPU weight or
gradient is retained. Any step exception poisons the instance: chunks already
updated cannot be rolled back and the caller must discard the policy.
"""
import math
import torch
from bf16_hybrid_adam_candidate import key, stochastic_round


class StreamingBf16AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-6, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, chunk_numel=262144, rounding_seed=20261106,
                 require_finite_ticket=False):
        if lr < 0 or eps < 0 or not all(0 <= b < 1 for b in betas) or chunk_numel < 1:
            raise ValueError('invalid optimizer options')
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps,
                                      weight_decay=weight_decay, chunk_numel=chunk_numel,
                                      rounding_seed=rounding_seed))
        self.poisoned = False
        self.require_finite_ticket = require_finite_ticket
        self._finite_ticket = False

    def arm_finite_ticket(self):
        self._finite_ticket = True

    @torch.no_grad()
    def step(self, closure=None):
        if self.poisoned:
            raise RuntimeError('streaming optimizer is poisoned; discard policy and state')
        if closure is not None:
            raise ValueError('closure unsupported')
        if self.require_finite_ticket and not self._finite_ticket:
            raise RuntimeError('missing current all-rank finite ticket')
        self._finite_ticket = False
        try:
            # Check everything before any update. The distributed caller must
            # perform its own all-rank finite vote before invoking this method.
            for group in self.param_groups:
                chunk = group['chunk_numel']
                if type(chunk) is not int or chunk < 1:
                    raise ValueError('invalid chunk size')
                for p in group['params']:
                    grad = getattr(p, 'decoupled_grad', None)
                    if grad is None:
                        continue
                    if (not isinstance(getattr(p, '_q38_canonical_name', None), str)
                        or not p._q38_canonical_name
                        or type(getattr(p, '_q38_global_offset', None)) is not int
                        or p._q38_global_offset < 0 or p.device.type != 'cuda'
                        or p.dtype != torch.bfloat16 or grad.device != p.device
                        or grad.dtype != torch.bfloat16 or p.shape != grad.shape
                        or not p.is_contiguous() or not grad.is_contiguous()):
                        raise ValueError('invalid CUDA BF16 shard, gradient, or coordinate')
                    if not self.require_finite_ticket:
                        flat = grad.reshape(-1)
                        for start in range(0, flat.numel(), chunk):
                            if not torch.isfinite(flat[start:start + chunk]).all():
                                raise FloatingPointError('nonfinite BF16 gradient')

            for group in self.param_groups:
                beta1, beta2 = group['betas']
                for p in group['params']:
                    grad = getattr(p, 'decoupled_grad', None)
                    if grad is None:
                        continue
                    state = self.state[p]
                    if not state:
                        state.update(step=0,
                            exp_avg=torch.zeros(p.shape, dtype=torch.bfloat16, device='cpu'),
                            exp_avg_sq=torch.zeros(p.shape, dtype=torch.bfloat16, device='cpu'))
                    step = state['step'] + 1
                    w, g = p.reshape(-1), grad.reshape(-1)
                    m, v = state['exp_avg'].reshape(-1), state['exp_avg_sq'].reshape(-1)
                    keys = {role: key(p._q38_canonical_name, step, role, group['rounding_seed'])
                            for role in ('weight', 'exp_avg', 'exp_avg_sq')}
                    for start in range(0, w.numel(), group['chunk_numel']):
                        end = min(w.numel(), start + group['chunk_numel'])
                        # Pageable synchronous transfers and explicit stream
                        # completion keep transient source/destination alive.
                        weight = w[start:end].to(device='cpu', dtype=torch.float32, non_blocking=False)
                        gradient = g[start:end].to(device='cpu', dtype=torch.float32, non_blocking=False)
                        mean, var = m[start:end].float(), v[start:end].float()
                        weight.mul_(1 - group['lr'] * group['weight_decay'])
                        mean.lerp_(gradient, 1 - beta1)
                        var.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
                        denom = var.sqrt().div_(math.sqrt(1 - beta2 ** step)).add_(group['eps'])
                        weight.addcdiv_(mean, denom, value=-group['lr'] / (1 - beta1 ** step))
                        offset = p._q38_global_offset + start
                        rounded_w = stochastic_round(weight, keys['weight'], offset)
                        rounded_m = stochastic_round(mean, keys['exp_avg'], offset)
                        rounded_v = stochastic_round(var, keys['exp_avg_sq'], offset)
                        w[start:end].copy_(rounded_w, non_blocking=False)
                        torch.cuda.current_stream(p.device).synchronize()
                        m[start:end].copy_(rounded_m)
                        v[start:end].copy_(rounded_v)
                    state['step'] = step
        except BaseException:
            self.poisoned = True
            raise
