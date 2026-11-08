"""Pure, conservative cgroup admission arithmetic for a first Adam update.

This module neither changes cgroup state nor allocates model memory. A caller
must make the decision unanimous across all training ranks before any rank
starts an optimizer step. No file-cache bytes are credited before they have
actually left the kernel's current cgroup charge.
"""
from __future__ import annotations
import time
from pathlib import PurePosixPath

GIB = 1024**3


def _nat(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError('Invalid nonnegative integer: '+name)
    return value


def eligible_inactive_file(stat):
    keys=('file','inactive_file','file_dirty','file_writeback','file_mapped','shmem','unevictable')
    values={key:_nat(stat[key],key) for key in keys}
    gross=min(values['file'],values['inactive_file'])
    excluded=sum(values[key] for key in ('file_dirty','file_writeback','file_mapped','shmem','unevictable'))
    return max(0,gross-excluded)


def evaluate(snapshot, peers, *, run_id, hostname, expected_cid, reserve=60*GIB,
             expected_time_namespace,maximum_age_seconds=15, now=None,monotonic_now=None,reuse_enabled=False):
    """Return a detailed decision or raise on missing/untrusted accounting.

    peers must contain exactly four distinct rank reports on this host. The
    first-step gate intentionally rejects already-allocated moments: a resumed
    or partially-written optimizer requires an independently reviewed path.
    """
    now=time.time() if now is None else now
    monotonic_now=time.monotonic() if monotonic_now is None else monotonic_now
    _nat(reserve,'reserve')
    if snapshot.get('run_id')!=run_id or snapshot.get('hostname')!=hostname:
        raise ValueError('Sampler run/hostname mismatch')
    if snapshot.get('container_id')!=expected_cid or snapshot.get('identity_verified') is not True:
        raise ValueError('Sampler container identity mismatch')
    if snapshot.get('same_time_namespace') is not True or snapshot.get('time_namespace')!=expected_time_namespace:
        raise ValueError('Sampler and rank time namespaces differ')
    started=snapshot.get('sample_started_monotonic');completed=snapshot.get('sample_completed_monotonic')
    if (not isinstance(started,(int,float)) or isinstance(started,bool)
            or not isinstance(completed,(int,float)) or isinstance(completed,bool)
            or not 0<=started<=completed<=monotonic_now or monotonic_now-started>maximum_age_seconds):
        raise ValueError('Sampler monotonic interval missing, future or stale')
    sample=snapshot.get('sample_unix')
    if not isinstance(sample,(int,float)) or isinstance(sample,bool) or not 0<=now-sample<=maximum_age_seconds:
        raise ValueError('Sampler snapshot missing, future or stale')
    if len(peers)!=4 or len({row.get('rank') for row in peers})!=4:
        raise ValueError('Exactly four distinct local rank reports required')
    if any(row.get('hostname')!=hostname for row in peers):
        raise ValueError('Peer hostname mismatch')
    spill=sum(_nat(row['cpu_gradient_bytes'],'cpu_gradient_bytes') for row in peers)
    allocated=sum(_nat(row['allocated_moment_bytes'],'allocated_moment_bytes') for row in peers)
    pending=sum(_nat(row['planned_moment_bytes'],'planned_moment_bytes') for row in peers)
    reuse=0
    if reuse_enabled:
        for row in peers:
            amount=_nat(row.get('reusable_cpu_gradient_bytes'),'reusable_cpu_gradient_bytes')
            if (row.get('reuse_plan_verified') is not True or not isinstance(row.get('reuse_plan_sha256'),str)
                    or len(row['reuse_plan_sha256'])!=64 or any(ch not in '0123456789abcdef' for ch in row['reuse_plan_sha256'])
                    or amount!=row['cpu_gradient_bytes']
                    or amount>row['planned_moment_bytes']):
                raise ValueError('Unverified or incomplete first-Adam reuse plan')
            reuse+=amount
        pending-=reuse
    if allocated!=0:
        raise ValueError('First-step admission forbids pre-existing moment allocation')
    groups=snapshot.get('cgroups')
    if not isinstance(groups,list) or not groups:
        raise ValueError('No cgroup sample')
    if not groups[0].get('is_exact_container') or any(not group.get('identity_verified') for group in groups):
        raise ValueError('Container and every ancestor must be identity verified')
    child_path=PurePosixPath(groups[0]['path'])
    if not child_path.is_absolute() or len({group['path'] for group in groups})!=len(groups):
        raise ValueError('Invalid or duplicate cgroup paths')
    if any(PurePosixPath(group['path']) not in child_path.parents for group in groups[1:]):
        raise ValueError('A sampled parent is not an actual ancestor')
    child_current=_nat(groups[0]['current_bytes'],'container_current')
    child_anon=_nat(groups[0]['stat']['anon'],'container_anon')
    container_non_state=max(0,min(child_anon,child_current)-spill-allocated)
    decisions=[]
    for group in groups:
        events=group.get('events_delta')
        if not isinstance(events,dict) or any(key not in events for key in ('oom','oom_kill')):
            raise ValueError('Missing cgroup OOM event baseline or delta')
        if any(_nat(events[key],key)!=0 for key in ('oom','oom_kill')):
            raise ValueError('New cgroup OOM event since sampler baseline')
        if not isinstance(group.get('pressure'),str):
            raise ValueError('Missing cgroup pressure telemetry')
        limit=_nat(group['limit_bytes'],'limit_bytes')
        current=_nat(group['current_bytes'],'current_bytes')
        if not limit or current>limit:
            raise ValueError('Invalid actual cgroup maximum/current')
        # This value is an upper bound, not proof of immediately available
        # capacity. Record it for diagnosis but take zero credit for it.
        inactive_upper=min(eligible_inactive_file(group['stat']),current)
        charged=current
        accounted=spill+allocated
        # File cache may disappear under reclaim and never offsets reserve.
        # cgroup v2 ancestors include this exact child's charged anonymous
        # bytes. Credit at most that known child subset, never sibling anon.
        anon=_nat(group['stat']['anon'],'anon')
        if current<child_current or anon<child_anon:
            raise ValueError('Non-atomic cgroup snapshot contradicts hierarchy')
        non_state=container_non_state
        reserve_shortfall=max(0,reserve-non_state)
        required=charged+pending+reserve_shortfall
        decisions.append(dict(path=group['path'],limit_bytes=limit,current_bytes=current,
            reclaimable_inactive_upper_bytes=inactive_upper,credited_cache_bytes=0,
            non_state_charged_bytes=non_state,
            reserve_shortfall_bytes=reserve_shortfall,existing_cpu_gradient_bytes=spill,
            existing_moment_bytes=allocated,pending_moment_bytes=pending,
            approved_reusable_cpu_gradient_bytes=reuse,
            required_bytes=required,headroom_bytes=limit-required,passed=required<=limit))
    return dict(passed=all(row['passed'] for row in decisions),run_id=run_id,hostname=hostname,
        sample_unix=sample,evaluated_unix=now,sample_started_monotonic=started,
        sample_completed_monotonic=completed,evaluated_monotonic=monotonic_now,
        peers=sorted(row['rank'] for row in peers),
        reserve_bytes=reserve,cgroups=decisions,active_file_credited=False)
