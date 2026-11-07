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


def _observe_failure(engine,event,micro,exc):
    """Exception-only allocator metadata; never synchronize or replace failure."""
    import torch
    try:
        _write_boundary(engine,event,micro,exception=f'{type(exc).__name__}: {exc}')
        if (os.environ.get('Q38_OOM_ALLOCATOR_AUDIT')!='1'
                or not isinstance(exc,torch.OutOfMemoryError)
                or getattr(exc,'_q38_allocator_audit_saved',False)):
            return
        # The normal path does not call these APIs or record allocation history.
        # Take the snapshot before any cache release/synchronization can erase
        # the distinction between inactive blocks and delayed stream frees.
        stats=torch.cuda.memory_stats()
        selected={key:value for key,value in stats.items()
            if key.startswith(('allocated_bytes.','active_bytes.','reserved_bytes.','inactive_split_bytes.',
                               'allocation.','active.','segment.','inactive_split.','requested_bytes.'))
            or key in ('num_alloc_retries','num_ooms','oversize_allocations.current','oversize_segments.current')}
        segments=torch.cuda.memory_snapshot();state_bytes={};state_blocks={};largest={}
        for segment in segments:
            for block in segment.get('blocks',[]):
                state=block.get('state','unknown');size=int(block['size'])
                state_bytes[state]=state_bytes.get(state,0)+size
                state_blocks[state]=state_blocks.get(state,0)+1
                largest[state]=max(largest.get(state,0),size)
        record=dict(schema='q38-exception-only-CUDA-allocator-block-metadata-v1',event=event,
            pid=os.getpid(),time_ns=time.time_ns(),actual_training_micro_index=micro,
            original_exception=f'{type(exc).__name__}: {exc}',stats=selected,
            allocator_backend=torch.cuda.memory.get_allocator_backend(),
            active_allocator_settings=torch._C._accelerator_getAllocatorSettings(),
            block_bytes_by_state=state_bytes,block_count_by_state=state_blocks,
            largest_block_bytes_by_state=largest,segment_count=len(segments),
            expandable_segment_count=sum(s.get('is_expandable') is True for s in segments),
            nonexpandable_segment_count=sum(s.get('is_expandable') is False for s in segments),
            unknown_expandability_segment_count=sum(not isinstance(s.get('is_expandable'),bool) for s in segments),
            inactive_split_stat_excludes_expandable_segments=True,
            full_segment_and_block_metadata=segments,
            snapshot_is_metadata_only=True,model_optimizer_or_gradient_tensor_payload_saved=False,
            observer_synchronized_or_released_cache=False,original_exception_will_be_reraised_unchanged=True)
        root=Path(os.environ['Q38_TRAINING_AUDIT_DIR']);root.mkdir(parents=True,exist_ok=True)
        path=root/f'allocator-oom-pid{os.getpid()}-{record["time_ns"]}.json'
        with path.open('x') as out:json.dump(record,out,allow_nan=False);out.write('\n')
        exc._q38_allocator_audit_saved=True
    except BaseException as observer_error:
        # Diagnostic failures, including a read-only/full disk, cannot mask the
        # original CUDA/Python exception or grant any successful step/admission.
        try:
            root=Path(os.environ['Q38_TRAINING_AUDIT_DIR']);root.mkdir(parents=True,exist_ok=True)
            path=root/f'allocator-observer-failed-pid{os.getpid()}-{time.time_ns()}.json'
            with path.open('x') as out:
                json.dump(dict(original_exception=repr(exc),observer_exception=repr(observer_error)),out)
        except BaseException:
            pass


def install():
    if os.environ.get('Q38_MICRO_BOUNDARY_RELEASE_CACHE')!='1':return False
    import torch
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngine,FSDPEngineWithLMHead
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
            _observe_failure(self,'forward-exception',micro,exc)
            raise
        _write_boundary(self,'exit-forward-before-backward',micro)
        return result

    forward._q38_micro_memory_boundary=True
    FSDPEngineWithLMHead.forward_step=forward
    if os.environ.get('Q38_OOM_ALLOCATOR_AUDIT')=='1':
        original_batch=FSDPEngine.forward_backward_batch
        @functools.wraps(original_batch)
        def batch(self,*args,**kwargs):
            try:
                return original_batch(self,*args,**kwargs)
            except BaseException as exc:
                if hasattr(self,'_q38_gradient_store'):
                    _observe_failure(self,'forward-backward-exception',
                        getattr(self,'_q38_training_micro_index',0),exc)
                raise
        FSDPEngine.forward_backward_batch=batch
    return True
