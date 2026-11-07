"""Observe actual micro boundaries and release only empty allocator segments.

This optional dedicated-profile hook does not replace loss/gradient tensors or
change FSDP's micro-batch loop. It runs before the original forward step, after
any preceding backward has returned to that loop.
"""
from __future__ import annotations
import functools,json,os,time
from pathlib import Path


def _write_boundary(engine, event, micro, *, exception=None, released_cache=False):
    import torch
    import torch.distributed as dist
    from qwen38_bf16_optimizer import local
    parameters=list(engine.module.parameters())
    shadow=[local(p.grad) for p in parameters if p.grad is not None]
    store=engine._q38_gradient_store
    device=torch.device('cuda',torch.cuda.current_device())
    free,total=torch.cuda.mem_get_info(device)
    proc_status={}
    for line in Path('/proc/self/status').read_text().splitlines():
        fields=line.split()
        if fields and fields[0] in ('VmRSS:','RssAnon:','RssFile:','RssShmem:'):
            proc_status[fields[0][:-1]+'_bytes']=int(fields[1])*1024
    record=dict(schema='q38-actual-engine-micro-memory-boundary-v1',event=event,
        rank=dist.get_rank(),world_size=dist.get_world_size(),pid=os.getpid(),time_ns=time.time_ns(),
        actual_training_micro_index=micro,optimizer_steps=engine.optimizer.audit()['steps'],
        gradient_store=store.audit(),parameter_shadow_grad_count=len(shadow),
        parameter_shadow_grad_bytes=sum(t.numel()*t.element_size() for t in shadow),
        cuda_allocated_bytes=torch.cuda.memory_allocated(device),
        cuda_reserved_bytes=torch.cuda.memory_reserved(device),
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
        cuda_free_bytes=free,cuda_total_bytes=total,process_CPU_resident_memory=proc_status,
        CPU_cgroup_limit_or_peak_inferred=False,empty_allocator_cache_released=released_cache,
        tensors_loss_rewards_gradients_modified=False,exception=exception)
    root=Path(os.environ['Q38_TRAINING_AUDIT_DIR']);root.mkdir(parents=True,exist_ok=True)
    path=root/f'micro-memory-rank{record["rank"]}-{event}-{record["time_ns"]}.json'
    with path.open('x') as out:json.dump(record,out,allow_nan=False);out.write('\n')
    return record


def install():
    if os.environ.get('Q38_MICRO_BOUNDARY_RELEASE_CACHE')!='1':return False
    import torch
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead
    original=FSDPEngineWithLMHead.forward_step
    if getattr(original,'_q38_micro_memory_boundary',False):return True

    @functools.wraps(original)
    def forward(self,micro_batch,loss_function,forward_only):
        if forward_only or not hasattr(self,'_q38_gradient_store'):
            return original(self,micro_batch,loss_function,forward_only)
        # The original forward_backward_batch calls the next forward only
        # after its preceding loss.backward() returns. Synchronizing here
        # drains stream-delayed frees; empty_cache releases unused segments.
        # Live policy masters and retained FP32 gradient slots remain intact.
        self._q38_training_micro_index=getattr(self,'_q38_training_micro_index',0)+1
        micro=self._q38_training_micro_index
        _write_boundary(self,'before-cache-release',micro)
        torch.cuda.synchronize();torch.cuda.empty_cache()
        _write_boundary(self,'enter-forward',micro,released_cache=True)
        try:
            result=original(self,micro_batch,loss_function,forward_only)
        except BaseException as exc:
            _write_boundary(self,'forward-exception',micro,exception=f'{type(exc).__name__}: {exc}')
            raise
        _write_boundary(self,'exit-forward-before-backward',micro)
        return result

    forward._q38_micro_memory_boundary=True
    FSDPEngineWithLMHead.forward_step=forward
    return True
