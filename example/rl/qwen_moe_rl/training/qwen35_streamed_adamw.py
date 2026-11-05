"""FP32 AdamW with CPU moments and bounded GPU workspace for sharded actors."""
from __future__ import annotations

import torch
from torch.optim.adamw import adamw


def _local(tensor):
    return tensor.to_local() if hasattr(tensor, 'to_local') else tensor


class CpuMomentAdamW(torch.optim.AdamW):
    """Run the PyTorch AdamW update on local master weights and clipped gradients.

    FSDP2 has already reduced each gradient to the owner's shard. Moments stay
    in ordinary (pageable) host RAM. Only two bounded FP32 chunks are staged
    on the parameter's device; the model and gradient never move to the CPU.
    """

    def __init__(self, params, *, chunk_bytes=32 * 1024 ** 2, **kwargs):
        super().__init__(params, **kwargs)
        if chunk_bytes < 4 or chunk_bytes % 4:
            raise ValueError('chunk_bytes must be a positive multiple of four')
        self.chunk_numel = chunk_bytes // 4
        self._last_chunk_peak_numel = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self._last_chunk_peak_numel = 0
        for group in self.param_groups:
            if group['amsgrad'] or group['differentiable'] or group['capturable']:
                raise ValueError('This training path requires ordinary non-capturable AdamW')
            for parameter in group['params']:
                if parameter.grad is None:
                    continue
                value = _local(parameter.detach())
                grad = _local(parameter.grad.detach())
                if value.numel() == 0:
                    continue
                if value.dtype != torch.float32 or grad.dtype != torch.float32:
                    raise ValueError(f'Expected FP32 master and gradient, got {value.dtype} and {grad.dtype}')
                if not value.is_contiguous() or grad.device != value.device or grad.shape != value.shape:
                    raise ValueError('Expected a contiguous master with an identically shaped local gradient')
                self._update_shard(value, grad, self.state[parameter], group)
        return loss

    def _update_shard(self, value, grad, state, group):
        if not state:
            state['step'] = torch.zeros((), dtype=torch.float32, device='cpu')
            for field in ('exp_avg', 'exp_avg_sq'):
                state[field] = torch.empty(value.shape, dtype=torch.float32, device='cpu')
        previous = int(state['step'].item())
        first = state['exp_avg'].view(-1)
        second = state['exp_avg_sq'].view(-1)
        if first.device.type != 'cpu' or second.device.type != 'cpu':
            raise RuntimeError('Persistent AdamW moments must remain on CPU')
        value, grad = value.view(-1), grad.reshape(-1)
        fused = value.device.type == 'cuda'
        for start in range(0, value.numel(), self.chunk_numel):
            end = min(value.numel(), start + self.chunk_numel)
            self._last_chunk_peak_numel = max(self._last_chunk_peak_numel, end - start)
            weight = value[start:end]
            if previous == 0:
                exp_avg = torch.zeros_like(weight)
                exp_avg_sq = torch.zeros_like(weight)
            else:
                exp_avg = first[start:end].to(value.device, copy=True)
                exp_avg_sq = second[start:end].to(value.device, copy=True)
            # The functional increments its scalar in place. Every chunk in the
            # same parameter uses the same previous step, even for short tails.
            step = torch.tensor(float(previous), device=value.device, dtype=torch.float32)
            adamw([weight], [grad[start:end]], [exp_avg], [exp_avg_sq], [], [step],
                  foreach=False, fused=fused, capturable=False, differentiable=False,
                  amsgrad=False, beta1=group['betas'][0], beta2=group['betas'][1],
                  lr=group['lr'], weight_decay=group['weight_decay'], eps=group['eps'],
                  maximize=group['maximize'])
            # Blocking copies ensure durable host state before reusing the chunk.
            first[start:end].copy_(exp_avg)
            second[start:end].copy_(exp_avg_sq)
            del exp_avg, exp_avg_sq
        state['step'].add_(1)

    def moment_placement(self):
        represented = sum(_local(param).numel() for param, state in self.state.items() if state)
        inactive = sum(_local(param).numel() for param, state in self.state.items()
                       if state and param.grad is None)
        moments = [state[name] for state in self.state.values()
                   for name in ('exp_avg', 'exp_avg_sq') if name in state]
        return {'implementation': 'torch.optim.adamw.adamw',
                'moment_devices': sorted({str(t.device) for t in moments}),
                'moment_dtypes': sorted({str(t.dtype) for t in moments}),
                'persistent_moment_bytes': sum(t.numel() * t.element_size() for t in moments),
                'moment_tensor_count': len(moments),
                'parameter_numel_with_state': represented,
                'state_numel_with_no_current_gradient': inactive,
                'max_staged_moment_bytes_per_step': 2 * self._last_chunk_peak_numel * 4,
                'configured_chunk_bytes_per_moment': self.chunk_numel * 4}

    def load_state_dict(self, state_dict):
        # Default torch Optimizer.load_state_dict migrates every moment to the
        # parameter device. Let it restore groups/counters, then restore the two
        # local tensors directly to CPU without a whole-state CUDA allocation.
        states = state_dict['state']
        reduced_states = {key: {name: item for name, item in state.items()
                                if name not in ('exp_avg', 'exp_avg_sq')}
                          for key, state in states.items()}
        super().load_state_dict(dict(state_dict, state=reduced_states))
        for saved, active in zip(state_dict['param_groups'], self.param_groups, strict=True):
            for key, param in zip(saved['params'], active['params'], strict=True):
                if key not in states:
                    continue
                for name in ('exp_avg', 'exp_avg_sq'):
                    tensor = _local(states[key][name])
                    if tensor.shape != _local(param).shape:
                        raise ValueError(f'{name} shape differs from the local master shard')
                    self.state[param][name] = tensor.detach().to('cpu', dtype=torch.float32, copy=True)
                self.state[param]['step'] = self.state[param]['step'].cpu()
