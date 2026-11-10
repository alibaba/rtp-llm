"""Guard resident nonsharded DDP weights for verl's HF export path."""
import functools
import contextvars
import json
import time


def install(module):
    import torch
    from megatron.core.distributed import DistributedDataParallel
    original = module.load_megatron_model_to_gpu
    if getattr(original, '_q36_resident_nonsharded', False): return
    original_export = module.MegatronEngine.get_per_tensor_param
    allowed = contextvars.ContextVar('q36_resident_export', default=False)

    @functools.wraps(original)
    def load(models, load_grad=True, load_frozen_params=True):
        if not allowed.get():
            return original(models, load_grad=load_grad, load_frozen_params=load_frozen_params)
        if not models or not all(isinstance(chunk, DistributedDataParallel) for chunk in models):
            return original(models, load_grad=load_grad, load_frozen_params=load_frozen_params)
        if any(getattr(chunk.ddp_config, 'use_distributed_optimizer', True) for chunk in models):
            return original(models, load_grad=load_grad, load_frozen_params=load_frozen_params)
        if load_grad:
            raise RuntimeError('nonsharded resident export does not implement gradient migration')
        start = time.monotonic(); device = torch.cuda.current_device(); count = 0
        for chunk in models:
            buffers = list(chunk.buffers) + list(chunk.expert_parallel_buffers)
            if not buffers or any(buffer.param_data is not None for buffer in buffers):
                raise RuntimeError('unexpected nonsharded DDP parameter buffer layout')
            for name, param in chunk.named_parameters():
                if param.device.type != 'cuda' or param.device.index != device:
                    raise RuntimeError('parameter is not resident on this CUDA device: ' + name)
                if not param.is_contiguous():
                    raise RuntimeError('parameter storage is not contiguous: ' + name)
                need = (param.storage_offset() + param.numel()) * param.element_size()
                if param.untyped_storage().nbytes() < need:
                    raise RuntimeError('parameter storage is incomplete: ' + name)
                count += 1
        if not count: raise RuntimeError('empty nonsharded parameter set')
        print('Q36_RESIDENT_PARAMETER_PREFLIGHT ' + json.dumps({'parameters': count,
              'cuda_device': device, 'seconds': time.monotonic()-start}), flush=True)

    load._q36_resident_nonsharded = True
    module.load_megatron_model_to_gpu = load

    @functools.wraps(original_export)
    def export(self, *args, **kwargs):
        if getattr(self, '_is_offload_param', True) or getattr(self.engine_config, 'param_offload', True):
            raise RuntimeError('resident HF export requires parameter offload disabled')
        token = allowed.set(True)
        try: return original_export(self, *args, **kwargs)
        finally: allowed.reset(token)

    module.MegatronEngine.get_per_tensor_param = export
