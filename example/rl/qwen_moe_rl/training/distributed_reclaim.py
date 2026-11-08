"""World-coordinated, single-leader bounded cache reclamation before Adam."""
import hashlib,json,os,time
from pathlib import Path
from cpu_admission import evaluate
from bounded_reclaim import build_plan,ReclaimCursor,PinnedSources

MAX_ROUNDS=4096

def post_advice_sample_ready(sample,finished_monotonic,expected_time_namespace):
    started=sample.get('sample_started_monotonic');completed=sample.get('sample_completed_monotonic')
    return (sample.get('same_time_namespace') is True and sample.get('time_namespace')==expected_time_namespace
        and isinstance(started,(int,float)) and isinstance(completed,(int,float))
        and started>finished_monotonic and completed>=started)

def _gather(dist,group,value,world):
    rows=[None]*world;dist.all_gather_object(rows,value,group=group);return rows

def reclaim_world(dist,group,local,all_rows,*,root,run_id,world,reserve,
                  action_group,cursor_holder=None,phase='first_adam'):
    if action_group is None or action_group is group:
        raise ValueError('Reclaim action requires its own bounded Gloo group')
    if str(dist.get_backend(action_group)).lower()!='gloo' or dist.get_world_size(action_group)!=world:
        raise ValueError('Reclaim action group must contain the full Gloo world')
    holder={} if cursor_holder is None else cursor_holder
    try:
        return _reclaim_world(dist,group,local,all_rows,root=root,run_id=run_id,world=world,reserve=reserve,
                              action_group=action_group,cursor_holder=holder,phase=phase)
    except BaseException:
        cursor=holder.get('cursor')
        if cursor is not None:
            try:cursor.close()
            except BaseException:pass
        raise

def _reclaim_world(dist,group,local,all_rows,*,root,run_id,world,reserve,
                   action_group,cursor_holder=None,phase='first_adam'):
    """Reclaim only while measured budget fails, before any Adam mutation.

    Every node elects the minimum rank once; other ranks never call fadvise.
    Each collective round performs at most one ≤4GiB batch per node; the
    configured Gloo group has a finite timeout. An unreported/ambiguous batch
    is a failure and the exclusive local claim forbids automatic replay.
    """
    host=local['hostname'];rank=local['rank'];root=Path(root)
    nodes={}
    for row in all_rows:nodes.setdefault(row['hostname'],[]).append(row)
    if len(nodes)*4!=world or any(len(v)!=4 for v in nodes.values()):
        raise ValueError('Node/rank grouping mismatch')
    peers=nodes[host];leader=min(p['rank'] for p in peers)
    if cursor_holder is None:cursor_holder={}
    cursor=cursor_holder.get('cursor');not_before=0;completed=[]
    for round_index in range(MAX_ROUNDS):
        try:
            if cursor_holder.get('identity', (run_id,host,os.getpid()))!=(run_id,host,os.getpid()):
                raise RuntimeError('Reclaim holder identity changed across admission boundaries')
            cursor_holder.setdefault('identity',(run_id,host,os.getpid()))
            deadline=time.monotonic()+30
            while True:
                sample=json.loads((root/'cpu-monitor/CURRENT.json').read_text())
                if post_advice_sample_ready(sample,not_before,os.readlink('/proc/self/ns/time')):break
                if time.monotonic()>=deadline:raise TimeoutError('No fresh post-advice cgroup sample')
                time.sleep(.2)
            options=dict(run_id=run_id,hostname=host,expected_cid=local['container_id'],
                         expected_time_namespace=os.readlink('/proc/self/ns/time'),reserve=reserve,maximum_age_seconds=5)
            if phase!='first_adam':options['phase']=phase
            else:options['reuse_enabled']=os.environ.get('Q38_FIRST_ADAM_REUSE_CPU_SPILL')=='1'
            decision=evaluate(sample,peers,**options)
            vote=dict(rank=rank,hostname=host,passed=decision['passed'],error=None,decision=decision)
        except Exception as exc:
            vote=dict(rank=rank,hostname=host,passed=False,error=type(exc).__name__+': '+str(exc))
        votes=_gather(dist,group,vote,world)
        if any(row.get('error') for row in votes):raise RuntimeError('World reclaim precheck failed; no Adam update')
        if all(row['passed'] for row in votes):
            return dict(passed=True,rounds=round_index,actual_current_only=True,advice_is_not_reclaim_proof=True)
        local_needed=not all(row['passed'] for row in votes if row['hostname']==host)
        action=dict(rank=rank,hostname=host,leader=False,needed=local_needed,error=None)
        if local_needed and rank==leader:
            action['leader']=True
            try:
                if cursor is None:
                    pinned=PinnedSources()
                    try:
                        plan=build_plan(root,peers,host,pinned=pinned);cursor=ReclaimCursor(plan,pinned=pinned);cursor_holder['cursor']=cursor
                        with (cursor.directory/'PLAN.json').open('x') as stream:json.dump(plan,stream,indent=2);stream.write('\n')
                    except BaseException:
                        try:pinned.close()
                        except BaseException as close_error:action['source_close_error_type']=type(close_error).__name__
                        raise
                batch=cursor.next_batch()
                if batch is None:raise MemoryError('Bounded source/wider reclaim budget exhausted')
                receipt=cursor.directory/('batch-%05d.json'%batch['round'])
                action.update(batch_summary={k:batch.get(k) for k in ('stage','round','advised_bytes',
                    'syscall_advised_bytes','total_advised_bytes','wide_scanned_bytes')},
                    receipt_path=str(receipt),receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest(),
                    finished_unix=time.time(),finished_monotonic=time.monotonic())
            except Exception as exc:
                action['error']=type(exc).__name__+': '+str(exc)
        actions=_gather(dist,action_group,action,world)
        if any(row.get('error') for row in actions):raise MemoryError('A node reclaim action failed; no Adam update')
        local_actions=[row for row in actions if row['hostname']==host and row.get('leader')]
        if local_needed:
            if len(local_actions)!=1:raise RuntimeError('Node did not have exactly one reclaim coordinator')
            not_before=local_actions[0]['finished_monotonic']
            completed.append(dict(round=round_index,finished_monotonic=not_before,
                finished_unix=local_actions[0]['finished_unix']))
        else:not_before=0
    raise MemoryError('Bounded reclaim round limit exceeded; no Adam update')
