"""Quiescent partial BF16 master offload through public FSDP module conversion.

No forward is legal while any selected chunk is on CPU. Native inference must
be closed and its contexts independently checked before restoring train state.
"""
from __future__ import annotations

from contextlib import contextmanager
import gc
import hashlib
import inspect
import torch
from qwen38_bf16_optimizer import local


@contextmanager
def preserve_parameter_identity():
    previous = torch.__future__.get_swap_module_params_on_conversion()
    torch.__future__.set_swap_module_params_on_conversion(True)
    try:yield
    finally:torch.__future__.set_swap_module_params_on_conversion(previous)


def _fingerprint(parameter):
    value = local(parameter.detach()).reshape(-1)
    count = min(64, value.numel())
    indices = torch.arange(count, device=value.device)
    if count > 1:indices = indices*(value.numel()-1)//(count-1)
    return hashlib.sha256(value[indices].cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def disable_prefetch_and_bound_reduction(model):
    from torch.distributed.fsdp import FSDPModule
    count = 0
    for module in model.modules():
        if not isinstance(module, FSDPModule):continue
        module.set_modules_to_backward_prefetch([])
        module.set_modules_to_forward_prefetch([])
        module.set_reduce_scatter_max_input_buffers(1, recurse=False)
        count += 1
    return count


class MasterPhaseManager:
    def __init__(self, model, gradients, optimizer, *, leaf_predicate, retained_bytes):
        if retained_bytes < 0:raise ValueError('Master GPU cache cannot be negative')
        self.model = model; self.gradients = gradients; self.optimizer = optimizer
        self.retained_bytes = retained_bytes; self.leaf_predicate = leaf_predicate
        self.offloaded = []; self.phase = 'train'; self.transition_records = []
        self._parameter_ids = {name:id(p) for name,p in model.named_parameters()}

    def _identity(self):
        current = {name:id(p) for name,p in self.model.named_parameters()}
        if current != self._parameter_ids:
            raise RuntimeError('FSDP conversion replaced optimizer-owned parameter identities')
        expected = {id(p) for p in self.model.parameters()}
        actual = {id(p) for group in self.optimizer.param_groups for p in group['params']}
        if expected != actual:raise RuntimeError('Optimizer references no longer cover actual model parameters')

    def _gpu_bytes(self):
        return sum(local(p).numel()*local(p).element_size() for p in self.model.parameters()
                   if local(p).device.type == 'cuda')

    def prepare_rollout(self):
        if self.phase != 'train':raise RuntimeError('Training masters are already offloaded')
        if self.gradients.slots or any(p.grad is not None for p in self.model.parameters()):
            raise RuntimeError('Commit optimizer update and clear gradients before entering rollout')
        from torch.distributed.fsdp import FSDPModule
        self.gradients.remove_hooks(); saved = {}
        try:
            for name,module in self.model.named_modules():
                if self._gpu_bytes() <= self.retained_bytes:break
                if not self.leaf_predicate(module):continue
                if not isinstance(module,FSDPModule):raise TypeError('Only explicit leaf FSDP modules can be offloaded')
                if any(self.leaf_predicate(child) for child in list(module.modules())[1:]):
                    raise ValueError('Nested offload units can alias or repeat master storage')
                module.reshard()
                before = {key:_fingerprint(p) for key,p in module.named_parameters()}
                with preserve_parameter_identity():module.to(device='cpu')
                self._identity()
                after = {key:_fingerprint(p) for key,p in module.named_parameters()}
                if before != after:raise RuntimeError('CPU offload changed sampled BF16 master bits')
                if any(local(p).device.type != 'cpu' for p in module.parameters()):
                    raise RuntimeError('An offload unit retained real GPU parameter storage')
                self.offloaded.append((name,module)); saved[name] = before
            if self._gpu_bytes() > self.retained_bytes:
                raise RuntimeError('Explicit offload units cannot meet the native-refit master-cache budget')
            self.phase = 'rollout'; self._offload_fingerprints = saved
            torch.cuda.synchronize();torch.cuda.empty_cache()
            self.transition_records.append({'phase':'rollout','gpu_master_bytes':self._gpu_bytes(),
                                           'offloaded_modules':len(self.offloaded),'sample_bits_verified':True})
            return self.transition_records[-1]
        except BaseException:
            # Keep the actual phase visible. Never attempt inference or training
            # with an unreviewed half-restored model after a conversion failure.
            self.phase = 'failed';raise

    def restore_training(self, *, native_contexts_released, refit_sources_released=False):
        if self.phase != 'rollout':raise RuntimeError('No committed rollout phase can be restored')
        if native_contexts_released is not True:
            raise RuntimeError('Real native-process/context cleanup proof is required before restoring masters')
        if refit_sources_released is not True:
            raise RuntimeError('All refit source views and the leased CUDA backing must be released before training')
        device = torch.device('cuda',torch.cuda.current_device())
        try:
            gc.collect();torch.cuda.synchronize(device)
            torch.cuda.ipc_collect();torch.cuda.empty_cache()
            before_restore=torch.cuda.memory_allocated(device)
            for name,module in self.offloaded:
                with preserve_parameter_identity():module.to(device=device)
                self._identity()
                if {key:_fingerprint(p) for key,p in module.named_parameters()} != self._offload_fingerprints[name]:
                    raise RuntimeError('Restored BF16 masters changed sampled values')
            self.offloaded.clear();self.gradients.register_hooks(self.model.parameters())
            for name,p in self.model.named_parameters():p._qwen36_parameter_name = name
            self.phase = 'train';torch.cuda.synchronize()
            self.transition_records.append({'phase':'train','gpu_master_bytes':self._gpu_bytes(),
                                           'refit_source_release_proven':True,
                                           'cuda_ipc_collect_after_native_cleanup':True,
                                           'cuda_allocated_bytes_before_master_restore':before_restore,
                                           'parameter_and_optimizer_identity_verified':True,
                                           'sample_bits_verified':True})
            return self.transition_records[-1]
        except BaseException:self.phase = 'failed';raise

    def require_training(self):
        if self.phase != 'train':raise RuntimeError('FSDP forward is forbidden during native rollout/refit')
