"""Qualification-only BF16 CPU AdamW for pinned Megatron HDO.

Caller must bind each CPU parameter copy to its canonical global name and
flat offset. This module is not a complete distributed optimizer integration.
"""
import hashlib
import math
import torch


def _signed64(x):
    x = int(x) & ((1 << 64) - 1)
    return x if x < 1 << 63 else x - (1 << 64)


def key(name, step, role, seed):
    data = f'q38-bf16-sr-v1\0{seed}\0{name}\0{step}\0{role}'.encode()
    return _signed64(int.from_bytes(hashlib.sha256(data).digest()[:8], 'little'))


def random16(count, seed, offset, device):
    x = torch.arange(offset, offset + count, dtype=torch.int64, device=device)
    x = x + _signed64(seed) + _signed64(0x9E3779B97F4A7C15)
    x = (x ^ ((x >> 30) & ((1 << 34) - 1))) * _signed64(0xBF58476D1CE4E5B9)
    x = (x ^ ((x >> 27) & ((1 << 37) - 1))) * _signed64(0x94D049BB133111EB)
    x = x ^ ((x >> 31) & ((1 << 33) - 1))
    return (x >> 48) & 65535


def stochastic_round(value, seed, offset):
    if value.dtype != torch.float32 or not value.is_contiguous() or not torch.isfinite(value).all():
        raise ValueError('expected finite contiguous FP32 update chunk')
    bits = value.view(torch.int32).reshape(-1).to(torch.int64) & 0xffffffff
    result = (((bits >> 16) + (random16(bits.numel(), seed, offset, value.device)
                                < (bits & 65535)).long()) & 65535).to(torch.int16)
    result = result.view(torch.bfloat16).reshape(value.shape)
    if not torch.isfinite(result).all():
        raise FloatingPointError('BF16 stochastic rounding overflow')
    return result


class Bf16ChunkedAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-6, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, chunk_numel=262144, rounding_seed=20261106, **kwargs):
        if kwargs or lr < 0 or eps < 0 or not all(0 <= b < 1 for b in betas) or chunk_numel < 1:
            raise ValueError('unsupported optimizer options')
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps,
                                      weight_decay=weight_decay, chunk_numel=chunk_numel,
                                      rounding_seed=rounding_seed))

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError('closure unsupported')
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                name = getattr(p, '_q38_canonical_name', None)
                offset = getattr(p, '_q38_global_offset', None)
                if (not isinstance(name, str) or not name or type(offset) is not int or offset < 0
                        or p.device.type != 'cpu' or p.dtype != torch.bfloat16
                        or p.grad.dtype != torch.bfloat16 or p.shape != p.grad.shape
                        or not p.is_contiguous() or not p.grad.is_contiguous()
                        or not torch.isfinite(p.grad).all()):
                    raise ValueError('invalid BF16 CPU shard, gradient, or canonical coordinate')
                state = self.state[p]
                if not state:
                    state.update(step=0, exp_avg=torch.zeros_like(p), exp_avg_sq=torch.zeros_like(p))
                state['step'] += 1
                step = state['step']
                w, g = p.reshape(-1), p.grad.reshape(-1)
                m, v = state['exp_avg'].reshape(-1), state['exp_avg_sq'].reshape(-1)
                keys = {role: key(name, step, role, group['rounding_seed'])
                        for role in ('weight', 'exp_avg', 'exp_avg_sq')}
                for start in range(0, w.numel(), group['chunk_numel']):
                    end = min(w.numel(), start + group['chunk_numel'])
                    weight, grad = w[start:end].float(), g[start:end].float()
                    mean, var = m[start:end].float(), v[start:end].float()
                    weight.mul_(1 - group['lr'] * group['weight_decay'])
                    mean.lerp_(grad, 1 - beta1)
                    var.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    denom = var.sqrt().div_(math.sqrt(1 - beta2 ** step)).add_(group['eps'])
                    weight.addcdiv_(mean, denom, value=-group['lr'] / (1 - beta1 ** step))
                    w[start:end].copy_(stochastic_round(weight, keys['weight'], offset + start))
                    m[start:end].copy_(stochastic_round(mean, keys['exp_avg'], offset + start))
                    v[start:end].copy_(stochastic_round(var, keys['exp_avg_sq'], offset + start))
