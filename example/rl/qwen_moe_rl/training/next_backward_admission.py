"""Private candidate: world CPU admission at an explicit train-batch boundary.

Not installed by default. Actual verl dispatch, source hashes, Gloo and GPU
behavior require separate qualification before deployment. No model tensor
is modified or written by this module.
"""
from __future__ import annotations
import functools,hashlib,inspect,json,os,socket,time
from pathlib import Path
from cpu_admission import evaluate,GIB


def future_spill_upper_bytes(owned_numel, gpu_cache_bytes, largest_shard_bytes):
    values=(owned_numel,gpu_cache_bytes,largest_shard_bytes)
    if any(isinstance(x,bool) or not isinstance(x,int) or x<0 for x in values):
        raise ValueError('Expected nonnegative integer capacity inputs')
    if largest_shard_bytes>owned_numel*4 or largest_shard_bytes%4 or gpu_cache_bytes%4:
        raise ValueError('Invalid FP32 shard/cache bounds')
    return min(owned_numel*4,max(0,owned_numel*4-gpu_cache_bytes+largest_shard_bytes))


def _exchange(dist,group,value,world):
    rows=[None]*world;dist.all_gather_object(rows,value,group=group);return rows


def _collect_local(engine,dist,group,world,run_id,host):
    import torch
    from qwen38_bf16_optimizer import local
    if dist.get_world_size()!=world or str(dist.get_backend(group)).lower()!='gloo':
        raise ValueError('World/Gloo group changed')
    if dist.get_world_size(engine.get_data_parallel_group())!=world or engine.get_data_parallel_size()!=world:
        raise ValueError('Train batch is not dispatched on the full DP world')
    if getattr(engine,'_q38_update_poisoned',False) or getattr(engine.optimizer,'_reuse_failed',False):
        raise RuntimeError('Optimizer update is poisoned')
    engine._q38_phases.require_training()
    store=engine._q38_gradient_store
    if store.slots or store.cpu_spilled_bytes or store.retained_bytes:
        raise ValueError('Previous gradients were not fully released at batch entry')
    parameters=[p for p in engine.module.parameters() if p.requires_grad]
    locals_=[local(p) for p in parameters]
    owned=sum(p.numel() for p in locals_);largest=max((4*p.numel() for p in locals_),default=0)
    if owned*4!=engine._q38_planned_moment_bytes:
        raise ValueError('Actual trainable shards disagree with the persistent moment plan')
    audit=engine.optimizer.audit();steps=audit['steps'];allocated=engine.optimizer.moments.bytes
    if (not isinstance(steps,list) or len(steps)>1 or any(isinstance(s,bool) or not isinstance(s,int) or s<1 for s in steps)):
        raise ValueError('Optimizer steps are missing or inconsistent')
    if not steps and allocated!=0:raise ValueError('Initial batch has pre-existing optimizer moments')
    if steps and (allocated!=owned*4 or not getattr(engine,'_q38_native_contexts_released',False)):
        raise ValueError('Incomplete moments or native contexts still live before next backward')
    if audit.get('moment_dtype')!='bfloat16' or audit.get('gradient_dtype')!='float32' or audit.get('disk_offload_bytes')!=0:
        raise ValueError('Actual optimizer precision/offload contract changed')
    nodes=json.loads(Path(os.environ['Q38_NODES_FILE']).read_text())['nodes']
    matching=[n for n in nodes if n['container_ip']==socket.gethostbyname(host)]
    if len(matching)!=1:raise ValueError('Current node IP not unique in the frozen allocation')
    proof=store.bind_gpu_capture_contract(parameters,torch.device('cuda',torch.cuda.current_device()))
    if (proof['total_bytes']!=owned*4 or proof['largest_bytes']!=largest or proof['cache_bytes']!=store.limit
            or proof['cpu_upper_bytes']!=future_spill_upper_bytes(owned,store.limit,largest)):
        raise ValueError('Frozen GPU capture proof disagrees with the planned spill bound')
    return dict(rank=dist.get_rank(),pid=os.getpid(),hostname=host,container_id=matching[0]['gpu_container'],
        run_id=run_id,optimizer_steps=steps,planned_moment_bytes=owned*4,allocated_moment_bytes=allocated,
        cpu_gradient_bytes=0,next_cpu_gradient_upper_bytes=future_spill_upper_bytes(owned,store.limit,largest),
        gpu_gradient_cache_limit=store.limit,largest_local_gradient_bytes=largest,capture_contract=proof,error=None)


def next_batch_admission(engine,dist,group,*,world=72):
    if not dist.is_initialized():raise RuntimeError('Distributed group is not initialized')
    rank=dist.get_rank();host=socket.gethostname();run_id=os.environ['Q38_RUN_ID']
    root=Path(os.environ['Q38_RUN_ROOT']);audit=Path(os.environ['Q38_TRAINING_AUDIT_DIR'])
    try:local=_collect_local(engine,dist,group,world,run_id,host)
    except Exception as exc:local=dict(rank=rank,hostname=host,error=type(exc).__name__+': '+str(exc))
    rows=_exchange(dist,group,local,world)
    if (len(rows)!=world or {r.get('rank') for r in rows}!=set(range(world)) or any(r.get('error') for r in rows)
            or len({tuple(r['optimizer_steps']) for r in rows})!=1):
        raise MemoryError('Batch entrance metadata disagreed across the world; no backward started')
    groups={}
    for row in rows:groups.setdefault(row['hostname'],[]).append(row)
    if (len(groups)*4!=world or any(len(x)!=4 or len({p['container_id'] for p in x})!=1 for x in groups.values())
            or len({x[0]['container_id'] for x in groups.values()})!=len(groups)):
        raise MemoryError('Batch entrance placement disagreed; no backward started')
    if not rows[0]['optimizer_steps']:
        # Every rank participates even on batch 1, avoiding a conditional
        # Gloo call when any other rank reports a different optimizer step.
        return dict(phase='initial_batch',rank=rank,all_world_steps_equal=True)
    if not hasattr(engine,'_q38_reclaim_cursor_holder'):engine._q38_reclaim_cursor_holder={}
    reserve=int(os.environ.get('Q38_CPU_RESERVE_BYTES_PER_NODE',60*GIB))
    if os.environ.get('Q38_CPU_CACHE_RECLAIM_ENABLED')=='1':
        from distributed_reclaim import reclaim_world
        reclaim_world(dist,group,local,rows,root=root,run_id=run_id,world=world,reserve=reserve,
                      cursor_holder=engine._q38_reclaim_cursor_holder,phase='next_backward')
    peers=groups[host]
    def decision():
        sample=json.loads((root/'cpu-monitor/CURRENT.json').read_text())
        result=evaluate(sample,peers,run_id=run_id,hostname=host,expected_cid=local['container_id'],
            reserve=reserve,expected_time_namespace=os.readlink('/proc/self/ns/time'),
            maximum_age_seconds=5,phase='next_backward')
        return sample,result
    def vote_now():
        try:
            sample,result=decision()
            return dict(rank=rank,passed=result['passed'],sample=sample,decision=result,error=None)
        except Exception as exc:return dict(rank=rank,passed=False,error=type(exc).__name__+': '+str(exc))
    votes=_exchange(dist,group,vote_now(),world)
    passed=all(v['passed'] for v in votes)
    try:
        audit.mkdir(parents=True,exist_ok=True)
        path=audit/f'next-backward-admission-rank{rank}-{time.time_ns()}.json'
        holder=engine._q38_reclaim_cursor_holder;cursor=holder.get('cursor')
        record=dict(schema='q38-next-backward-admission-private-v1',rank=rank,run_id=run_id,
            optimizer_steps=local['optimizer_steps'],votes=votes,passed=passed,
            coordinator_completed_advice_bytes=cursor.total if cursor else None,
            coordinator_rounds=cursor.round if cursor else None,
            actual_current_only=True,existing_moments_not_added_twice=True)
        with path.open('x') as out:json.dump(record,out);out.write('\n')
        saved=dict(rank=rank,passed=True)
    except Exception as exc:saved=dict(rank=rank,passed=False,error=type(exc).__name__+': '+str(exc))
    saved_votes=_exchange(dist,group,saved,world)
    final_local=vote_now() if passed and all(x['passed'] for x in saved_votes) else dict(rank=rank,passed=False)
    final_votes=_exchange(dist,group,final_local,world)
    age=(time.monotonic()-final_local['sample']['sample_started_monotonic']) if final_local.get('passed') else None
    fresh=_exchange(dist,group,dict(rank=rank,passed=bool(age is not None and 0<=age<=5),age=age),world)
    if not all(x.get('passed') for x in final_votes) or not all(x.get('passed') for x in fresh):
        raise MemoryError('Next-backward cgroup, audit or freshness gate denied; no backward started')
    try:
        final_path=audit/f'next-backward-final-rank{rank}-{time.time_ns()}.json'
        post_age=time.monotonic()-final_local['sample']['sample_started_monotonic']
        with final_path.open('x') as out:
            json.dump(dict(schema='q38-next-backward-final-private-v1',rank=rank,run_id=run_id,
                final_votes=final_votes,freshness=fresh,post_collective_age_seconds=post_age,
                no_atomic_memory_lease_after_collective=True),out);out.write('\n')
        after_write_age=time.monotonic()-final_local['sample']['sample_started_monotonic']
        final_saved=dict(rank=rank,passed=bool(0<=after_write_age<=5),age_after_write=after_write_age)
    except Exception as exc:final_saved=dict(rank=rank,passed=False,error=type(exc).__name__+': '+str(exc))
    saved_final=_exchange(dist,group,final_saved,world)
    after_saved_vote_age=time.monotonic()-final_local['sample']['sample_started_monotonic']
    final_age_votes=_exchange(dist,group,dict(rank=rank,passed=bool(0<=after_saved_vote_age<=5),
        age_after_saved_vote=after_saved_vote_age),world)
    if not all(row.get('passed') for row in saved_final) or not all(row.get('passed') for row in final_age_votes):
        raise MemoryError('Next-backward final evidence failed or became stale; no backward started')
    return record


def install_batch_boundary(engine_class,dist,*,expected_train_batch_sha256,on_failure,is_target,world=72):
    """Candidate only: do not call without actual-source/dispatch qualification."""
    original=engine_class.train_batch
    actual=hashlib.sha256(inspect.getsource(original).encode()).hexdigest()
    if actual!=expected_train_batch_sha256:raise ValueError('Unreviewed actual train_batch implementation')
    if list(inspect.signature(original).parameters)!=['self','data','loss_function']:
        raise ValueError('Unreviewed actual train_batch signature')
    if not callable(on_failure) or not callable(is_target):raise ValueError('Engine target and poison handlers are required')
    @functools.wraps(original)
    def checked(self,data,loss_function):
        try:
            target=is_target(self)
        except BaseException:
            try:on_failure(self)
            except BaseException:self._q38_update_poisoned=True
            raise
        if not target:return original(self,data,loss_function)
        try:
            if getattr(self,'_q38_batch_in_progress',False):
                raise RuntimeError('A prior batch has not completed its world vote')
            self._q38_batch_in_progress=True
            next_batch_admission(self,dist,self._q38_cpu_admission_group,world=world)
        except BaseException:
            try:on_failure(self)
            except BaseException:self._q38_update_poisoned=True
            raise
        result=None;failure=None
        try:result=original(self,data,loss_function)
        except BaseException as exc:
            failure=exc
            try:on_failure(self)
            except BaseException:self._q38_update_poisoned=True
        def world_vote(phase,error):
            group=self._q38_batch_completion_group
            if str(dist.get_backend(group)).lower()!='gloo' or dist.get_world_size(group)!=world:
                raise RuntimeError('Batch-completion group is not the full Gloo world')
            rows=_exchange(dist,group,dict(rank=dist.get_rank(),phase=phase,passed=error is None,
                error=None if error is None else type(error).__name__),world)
            if ({r.get('rank') for r in rows}!=set(range(world))
                    or any(r.get('phase')!=phase or r.get('passed') is not True for r in rows)):
                raise RuntimeError('A rank failed '+phase+'; no global rollback is implied')
        try:
            world_vote('train-batch-complete',failure)
            if failure is not None:raise failure
            finish_error=None
            try:self._q38_gradient_store.finish_gpu_capture_contract()
            except BaseException as exc:finish_error=exc
            world_vote('capture-contract-finished',finish_error)
            if finish_error is not None:raise finish_error
            self._q38_batch_in_progress=False
            return result
        except BaseException:
            try:on_failure(self)
            except BaseException:self._q38_update_poisoned=True
            if failure is not None:raise failure
            raise
    checked._q38_next_batch_gate=True
    engine_class.train_batch=checked
    return actual
