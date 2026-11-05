"""Real AdamW updates with bounded snapshots; gradients and updates are unchanged."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import socket
import time


def AuditedAdamW(*args, **kwargs):
    return _audited_adamw(False, *args, **kwargs)


def CpuMomentAuditedAdamW(*args, **kwargs):
    return _audited_adamw(True, *args, **kwargs)


def _audited_adamw(stage_moments, *args, **kwargs):
    # This factory runs after Ray has assigned each worker's CUDA device mask.
    from install_rtp_bridge import worker_setup
    worker_setup()
    from rllm.trainer.verl.patch import apply_all_verl_patches
    apply_all_verl_patches()
    import torch
    import torch.distributed as dist

    if stage_moments:
        from qwen35_streamed_adamw import CpuMomentAdamW as OptimizerBase
    else:
        OptimizerBase = torch.optim.AdamW

    class BoundedAuditedAdamW(OptimizerBase):
        def step(self, *step_args, **step_kwargs):
            pairs = []
            total_numel = tensor_count = 0
            grad_l2_sq = None
            grad_tensors = gradient_local_numel = 0
            master_dtypes, gradient_dtypes, gradient_devices = set(), set(), set()
            for group in self.param_groups:
                if float(group.get('weight_decay', 0)) != 0:
                    raise RuntimeError('Acceptance requires weight_decay=0')
                for parameter in group['params']:
                    if not parameter.requires_grad:
                        continue
                    value = parameter.detach()
                    value = value.to_local() if hasattr(value, 'to_local') else value
                    if value.numel() == 0:
                        continue
                    total_numel += value.numel()
                    master_dtypes.add(str(value.dtype))
                    tensor_count += 1
                    flat = value.reshape(-1)
                    count = min(64, flat.numel())
                    indices = torch.arange(count, device=flat.device, dtype=torch.int64)
                    if count > 1:
                        indices = indices*(flat.numel()-1)//(count-1)
                    before = flat[indices].float().cpu().clone()
                    pairs.append((parameter, indices, before))
                    if parameter.grad is not None:
                        grad_tensors += 1
                        grad = parameter.grad.detach()
                        grad = grad.to_local() if hasattr(grad, 'to_local') else grad
                        gradient_dtypes.add(str(grad.dtype))
                        gradient_devices.add(str(grad.device))
                        grad = grad.reshape(-1)
                        gradient_local_numel += grad.numel()
                        for start in range(0, grad.numel(), 4*1024**2):
                            local = grad[start:start+4*1024**2].float()
                            term = local.square().sum().double()
                            grad_l2_sq = term if grad_l2_sq is None else grad_l2_sq+term
            returned = super().step(*step_args, **step_kwargs)
            old_hash, new_hash = hashlib.sha256(), hashlib.sha256()
            changed = sample_count = 0
            sample_delta_sq = sample_max_delta = 0.0
            devices = set()
            for parameter, indices, before in pairs:
                value = parameter.detach()
                value = value.to_local() if hasattr(value, 'to_local') else value
                devices.add(str(value.device))
                after = value.reshape(-1)[indices].float().cpu()
                old_hash.update(before.numpy().tobytes())
                new_hash.update(after.numpy().tobytes())
                delta = after-before
                changed += int(torch.count_nonzero(delta).item())
                sample_count += delta.numel()
                sample_delta_sq += float(delta.square().sum().item())
                sample_max_delta = max(sample_max_delta, float(delta.abs().max().item()))
            steps = [float(v['step'].item()) for v in self.state.values() if 'step' in v]
            gradient_l2 = math.sqrt(float(grad_l2_sq.item())) if grad_l2_sq is not None else 0.0
            from qwen397_streamed_checkpoint import observed_node_ip
            record = {'schema': 'qwen36-bounded-real-optimizer-v1', 'pid': os.getpid(),
                      'hostname': socket.gethostname(), 'node_ip': observed_node_ip(),
                      'rank': dist.get_rank() if dist.is_initialized() else 0,
                      'world_size': dist.get_world_size() if dist.is_initialized() else 1,
                      'optimizer': 'AdamW', 'time_ns': time.time_ns(),
                      'trainable_local_numel': total_numel, 'tensor_count': tensor_count,
                      'gradient_tensor_count': grad_tensors, 'gradient_local_numel': gradient_local_numel,
                      'gradient_l2': gradient_l2,
                      'master_dtype': next(iter(master_dtypes)) if len(master_dtypes) == 1 else None,
                      'master_dtypes': sorted(master_dtypes),
                      'gradient_dtypes': sorted(gradient_dtypes), 'gradient_devices': sorted(gradient_devices),
                      'parameter_devices': sorted(devices),
                      'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                      'sampled_numel': sample_count, 'changed_sampled_elements': changed,
                      'sample_delta_l2': math.sqrt(sample_delta_sq), 'sample_max_abs_delta': sample_max_delta,
                      'sample_before_sha256': old_hash.hexdigest(), 'sample_after_sha256': new_hash.hexdigest(),
                      'optimizer_step_min': min(steps) if steps else None,
                      'optimizer_step_max': max(steps) if steps else None,
                      'audit_scope': 'full local gradient norm; deterministic bounded weight coordinates',
                      'audit_passed': math.isfinite(gradient_l2) and gradient_l2>0 and changed>0}
            if stage_moments:
                record['moment_placement'] = self.moment_placement()
            if any(device.startswith('cuda') for device in devices):
                props = torch.cuda.get_device_properties(torch.cuda.current_device())
                record['gpu_uuid'] = str(getattr(props, 'uuid', 'unknown'))
            path = Path(os.environ['RLLM_OPTIMIZER_AUDIT_DIR'])
            path.mkdir(parents=True, exist_ok=True)
            (path/f'optimizer-rank{record["rank"]}-{record["time_ns"]}.json').write_text(json.dumps(record, indent=2)+'\n')
            print('REAL_OPTIMIZER_AUDIT', json.dumps(record), flush=True)
            if not record['audit_passed'] and os.environ.get('RLLM_REQUIRE_REAL_UPDATE', '1') == '1':
                raise RuntimeError('No finite, nonzero real optimizer update was observed on this rank')
            return returned

    return BoundedAuditedAdamW(*args, **kwargs)
