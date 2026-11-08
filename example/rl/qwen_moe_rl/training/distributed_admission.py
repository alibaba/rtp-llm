"""First-Adam world-wide CPU cgroup admission, using an explicit Gloo group."""
from pathlib import Path
import hashlib,json,os,socket,time
from cpu_admission import evaluate

def _exchange(dist,group,value,world):
    rows=[None]*world
    dist.all_gather_object(rows,value,group=group)
    return rows

def first_adam_admission(engine,dist,group,*,world=72):
    """All ranks enter; every rank agrees before any first-step moment write.

    Local read/evaluation/audit failures are exchanged as data, so a single
    failed rank does not silently let another begin Adam. A later process or
    machine failure cannot be made transactional by this preflight.
    """
    if not dist.is_initialized() or dist.get_world_size()!=world:
        raise RuntimeError('Admission requires the actual world')
    if str(dist.get_backend(group)).lower()!='gloo':
        raise RuntimeError('Admission metadata requires a CPU Gloo group')
    rank=dist.get_rank();host=socket.gethostname();run_id=os.environ['Q38_RUN_ID']
    root=Path(os.environ['Q38_RUN_ROOT']);audit_dir=Path(os.environ['Q38_TRAINING_AUDIT_DIR'])
    reuse_enabled=os.environ.get('Q38_FIRST_ADAM_REUSE_CPU_SPILL')=='1'
    reuse_plan=None
    try:
        snapshot=json.loads((root/'cpu-monitor/CURRENT.json').read_text())
        nodes=json.loads(Path(os.environ['Q38_NODES_FILE']).read_text())['nodes']
        ip=socket.gethostbyname(host)
        matching=[row for row in nodes if row['container_ip']==ip]
        if len(matching)!=1:raise ValueError('Current node IP not unique in the frozen allocation')
        cid=matching[0]['gpu_container']
        local=dict(rank=rank,pid=os.getpid(),hostname=host,container_id=cid,planned_moment_bytes=engine._q38_planned_moment_bytes,
            allocated_moment_bytes=engine.optimizer.moments.bytes,
            cpu_gradient_bytes=engine._q38_gradient_store.cpu_spilled_bytes,
            snapshot=snapshot,error=None)
        if reuse_enabled:
            reuse_plan=engine.optimizer.prepare_reuse_plan()
            if reuse_plan['planned_moment_bytes']!=local['planned_moment_bytes'] or reuse_plan['reusable_bytes']!=local['cpu_gradient_bytes']:
                raise ValueError('Actual reusable slots differ from planned moments/CPU spill')
            engine.optimizer.bind_reuse_plan(reuse_plan)
            audit_dir.mkdir(parents=True,exist_ok=True)
            gradients=engine._q38_gradient_store.audit()
            if (not gradients['all_gradients_clipped'] or gradients['gradient_numel']!=engine._q38_owned_numel
                    or gradients['cpu_spilled_bytes']!=reuse_plan['reusable_bytes']):
                raise ValueError('Pre-update complete clipped gradient audit differs from frozen plan')
            record=dict(schema='q38-first-adam-pre-update-gradient-v1',run_id=run_id,rank=rank,pid=os.getpid(),
                world_size=world,recorded_unix=time.time(),generation=reuse_plan['generation'],
                expected_local_numel=engine._q38_owned_numel,gradient_store=gradients,slot_plan=reuse_plan,
                captured_before_any_optimizer_write=True,tensor_payload_included=False)
            raw=(json.dumps(record,sort_keys=True,indent=2)+'\n').encode()
            path=audit_dir/f'pre-adam-reuse-rank{rank}-{time.time_ns()}.json'
            with path.open('xb') as stream:stream.write(raw)
            digest=hashlib.sha256(raw).hexdigest()
            engine.optimizer.attach_reuse_evidence(digest,path)
            local.update(reuse_plan_verified=True,reusable_cpu_gradient_bytes=reuse_plan['reusable_bytes'],
                reuse_plan_sha256=digest,reuse_plan_path=str(path))
    except Exception as exc:
        local=dict(rank=rank,hostname=host,error=type(exc).__name__+': '+str(exc))
    rows=_exchange(dist,group,local,world)
    if os.environ.get('Q38_CPU_CACHE_RECLAIM_ENABLED')=='1':
        if len({row['rank'] for row in rows})!=world or any(row.get('error') for row in rows):
            raise MemoryError('A rank could not report state before bounded reclaim')
        from distributed_reclaim import reclaim_world
        reclaim_world(dist,group,local,rows,root=root,run_id=run_id,world=world,
            reserve=int(os.environ.get('Q38_CPU_RESERVE_BYTES_PER_NODE',60*1024**3)))
    try:
        if {row['rank'] for row in rows}!=set(range(world)) or any(row.get('error') for row in rows):
            raise ValueError('At least one rank could not report its real state')
        groups={}
        for row in rows:groups.setdefault(row['hostname'],[]).append(row)
        if len(groups)*4!=world or any(len(peers)!=4 for peers in groups.values()):
            raise ValueError('Current world does not consist of four ranks per node')
        if len({peers[0]['container_id'] for peers in groups.values()})!=len(groups):
            raise ValueError('Two nodes report the same container identity')
        peers=groups[host]
        if len({row['container_id'] for row in peers})!=1:raise ValueError('Node CID disagreement')
        # Always read a fresh actual sample after potential reclaim; never
        # reuse the initial pre-reclaim snapshot.
        current=json.loads((root/'cpu-monitor/CURRENT.json').read_text())
        decision=evaluate(current,peers,run_id=run_id,hostname=host,expected_cid=local['container_id'],
            expected_time_namespace=os.readlink('/proc/self/ns/time'),reuse_enabled=reuse_enabled,
            reserve=int(os.environ.get('Q38_CPU_RESERVE_BYTES_PER_NODE',60*1024**3)))
        result=dict(rank=rank,hostname=host,passed=decision['passed'],decision=decision,error=None)
    except Exception as exc:
        result=dict(rank=rank,hostname=host,passed=False,error=type(exc).__name__+': '+str(exc))
    votes=_exchange(dist,group,result,world)
    admitted=all(row.get('passed') is True for row in votes)
    record=dict(schema='q38-first-adam-cgroup-admission-v1',run_id=run_id,rank=rank,
        recorded_unix=time.time(),world_size=world,budget_votes_passed=admitted,local=result,
        failed_votes=[row for row in votes if row.get('passed') is not True],
        no_moments_or_weights_written_by_gate=True)
    try:
        audit_dir.mkdir(parents=True,exist_ok=True)
        path=audit_dir/f'cpu-admission-rank{rank}-{time.time_ns()}.json'
        with path.open('x') as stream:json.dump(record,stream,indent=2);stream.write('\n')
        audit=dict(rank=rank,passed=True,path=str(path))
    except Exception as exc:
        audit=dict(rank=rank,passed=False,error=type(exc).__name__+': '+str(exc))
    audit_votes=_exchange(dist,group,audit,world)
    if not admitted or not all(row['passed'] for row in audit_votes):
        raise MemoryError('First Adam denied by world-wide cgroup admission or audit failure; no update committed')
    # Re-read actual charged bytes after the potentially slow audit/vote.
    # No assumed reclaim is deducted; both this final sample and its age are
    # checked immediately before a final world-wide vote.
    try:
        if reuse_enabled and engine.optimizer.prepare_reuse_plan()!=reuse_plan:
            raise ValueError('First-Adam reuse slot identity changed before final vote')
        final_snapshot=json.loads((root/'cpu-monitor/CURRENT.json').read_text())
        final=evaluate(final_snapshot,peers,run_id=run_id,hostname=host,expected_cid=local['container_id'],
            expected_time_namespace=os.readlink('/proc/self/ns/time'),
            reserve=int(os.environ.get('Q38_CPU_RESERVE_BYTES_PER_NODE',60*1024**3)),maximum_age_seconds=5,reuse_enabled=reuse_enabled)
        final_vote=dict(rank=rank,passed=final['passed'],decision=final,snapshot=final_snapshot,error=None)
    except Exception as exc:
        final_vote=dict(rank=rank,passed=False,error=type(exc).__name__+': '+str(exc))
    final_votes=_exchange(dist,group,final_vote,world)
    final_passed=all(row['passed'] for row in final_votes)
    final_record=dict(schema='q38-final-cgroup-admission-v1',run_id=run_id,rank=rank,
        recorded_unix=time.time(),all_budget_votes_passed=final_passed,world_size=world,
        final_votes=final_votes,no_moments_or_weights_written_by_gate=True)
    try:
        final_path=audit_dir/f'cpu-admission-final-rank{rank}-{time.time_ns()}.json'
        with final_path.open('x') as stream:json.dump(final_record,stream,indent=2);stream.write('\n')
        persisted=dict(rank=rank,passed=True,path=str(final_path))
    except Exception as exc:
        persisted=dict(rank=rank,passed=False,error=type(exc).__name__+': '+str(exc))
    persisted_votes=_exchange(dist,group,persisted,world)
    # Re-check local sample age after persisted final votes; each rank sends
    # its own host-relative age, so clock offsets between nodes are irrelevant.
    # A slow final collective or a subsequent process fault is not a
    # transaction guarantee and remains visible in the caller's outcome.
    age=(time.monotonic()-final_snapshot['sample_started_monotonic']) if final_vote.get('passed') else None
    fresh_vote=dict(rank=rank,passed=bool(final_passed and all(x['passed'] for x in persisted_votes)
        and age is not None and 0<=age<=5),sample_age_seconds=age)
    freshness=_exchange(dist,group,fresh_vote,world)
    if not all(row['passed'] for row in freshness):
        try:
            with (audit_dir/f'cpu-admission-final-denied-rank{rank}-{time.time_ns()}.json').open('x') as stream:
                json.dump(dict(schema='q38-final-cgroup-admission-denied-v1',run_id=run_id,
                    rank=rank,final_votes=final_votes,persistence=persisted_votes,freshness=freshness),stream,indent=2)
                stream.write('\n')
        except Exception:pass
        raise MemoryError('Final first-Adam cgroup, persistence or freshness check denied; no update committed')
    # Persist the distinct freshness vote as well as the final sample vote.
    # This is an observation at the recorded instant, not a lease preventing
    # a later allocation by an external process or a process failure.
    post_gather_age=time.monotonic()-final_snapshot['sample_started_monotonic']
    success=dict(schema='q38-final-cgroup-freshness-v1',run_id=run_id,rank=rank,
        recorded_unix=time.time(),final_sample_votes=final_votes,
        final_record_persistence_votes=persisted_votes,freshness_votes=freshness,
        local_sample_age_after_freshness_collective=post_gather_age,
        no_atomic_lease_after_collective=True,no_moments_or_weights_written_by_gate=True)
    try:
        success_path=audit_dir/f'cpu-admission-freshness-rank{rank}-{time.time_ns()}.json'
        with success_path.open('x') as stream:json.dump(success,stream,indent=2);stream.write('\n')
        saved=dict(rank=rank,passed=bool(0<=post_gather_age<=5),path=str(success_path))
    except Exception as exc:
        saved=dict(rank=rank,passed=False,error=type(exc).__name__+': '+str(exc))
    saved_votes=_exchange(dist,group,saved,world)
    if not all(row['passed'] for row in saved_votes):
        raise MemoryError('Freshness evidence failed or was stale; no Adam update committed')
    return success
