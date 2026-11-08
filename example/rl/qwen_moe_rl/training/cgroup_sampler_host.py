"""Owned, read-only cgroup sampler; emits metadata into the selected container.

Execute on the authorized node through the wt connection container. This is a
candidate only: no sampler or training process is started by importing it.
"""
from pathlib import Path
import argparse,json,os,re,subprocess,time

FIELDS=('anon','file','inactive_file','active_file','file_dirty','file_writeback',
        'file_mapped','shmem','unevictable','pgscan','pgsteal','pgmajfault')

def identity(cid):
    p=subprocess.run(['pouch','inspect',cid],stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                     universal_newlines=True,timeout=10,check=True)
    data=json.loads(p.stdout)[0]
    if data['Id']!=cid or not data['State']['Running'] or 'luolinglong' not in data['Name'] or 'k8s_workdir_' not in data['Name']:
        raise RuntimeError('Container identity/ownership mismatch')
    pid=int(data['State']['Pid']);proc=Path('/proc')/str(pid)
    ticks=(proc/'stat').read_text().split(') ',1)[1].split()[19]
    return pid,ticks,proc

def cgroups(proc,cid):
    membership=[line.split(':',2)[2] for line in (proc/'cgroup').read_text().splitlines() if line.startswith('0::')]
    if len(membership)!=1:raise RuntimeError('Missing cgroup v2 membership')
    rows=[]
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        fields=line.split();index=fields.index('-')
        if fields[index+1]!='cgroup2':continue
        mount=Path(fields[4]);relative=os.path.relpath(membership[0],fields[3])
        if relative=='..' or relative.startswith('../'):continue
        current=mount/relative;first=True
        while True:
            limit=current/'memory.max'
            if limit.is_file():
                raw=limit.read_text().strip()
                if raw!='max':
                    stats=dict(line.split(' ',1) for line in (current/'memory.stat').read_text().splitlines())
                    stats={key:int(stats[key]) for key in FIELDS}
                    def optional(name):
                        p=current/name
                        return p.read_text().strip() if p.is_file() else None
                    rows.append(dict(path=str(current),is_exact_container=first,identity_verified=True,
                        limit_bytes=int(raw),current_bytes=int((current/'memory.current').read_text()),
                        stat=stats,events=optional('memory.events'),pressure=optional('memory.pressure'),
                        swap_current=optional('memory.swap.current')))
            if current==mount:break
            first=False;current=current.parent
    if not rows or not rows[0]['is_exact_container'] or cid not in rows[0]['path']:
        raise RuntimeError('Exact bounded container cgroup not found')
    return rows

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--container-id',required=True)
    parser.add_argument('--run-id',required=True);parser.add_argument('--node',required=True)
    parser.add_argument('--interval',type=float,default=1);args=parser.parse_args()
    if not re.fullmatch(r'[0-9a-f]{64}',args.container_id):raise ValueError('Invalid CID')
    if not re.fullmatch(r'q38-rl72-bf16-[a-z0-9-]{8,80}',args.run_id):raise ValueError('Invalid run')
    if not re.fullmatch(r'node(?:[1-9]|1[0-8])',args.node):raise ValueError('Invalid node')
    if not 0.5<=args.interval<=10:raise ValueError('Invalid sample interval')
    pid,ticks,proc=identity(args.container_id)
    host_time_ns=os.readlink('/proc/self/ns/time');container_time_ns=os.readlink(str(proc/'ns/time'))
    if host_time_ns!=container_time_ns:raise RuntimeError('Host/container monotonic time namespaces differ')
    root=proc/'root';host=(root/'etc/hostname').read_text().strip()
    # Each GPU container has a distinct filesystem, so the same run-scoped
    # path is local to each node and usable by all four colocated ranks.
    output=root/'home/admin/qwen38-72gpu-bf16-state-rl-20261106/runs'/args.run_id/'cpu-monitor'
    # Exclusive directory prevents silent sampler restart, baseline reset or
    # a second writer replacing this run's current snapshot.
    output.mkdir(parents=True,exist_ok=False)
    sampler_ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]
    with (output/'CLAIM.json').open('x') as stream:
        json.dump(dict(run_id=args.run_id,node=args.node,container_id=args.container_id,
            sampler_pid=os.getpid(),sampler_start_ticks=sampler_ticks,created_unix=time.time()),stream)
    baseline_events=None
    while not (output/'STOP').exists():
        sample_started=time.monotonic()
        if identity(args.container_id)[:2]!=(pid,ticks):raise RuntimeError('Container was replaced')
        sampled=cgroups(proc,args.container_id)
        sample_completed=time.monotonic()
        parsed={row['path']:{k:int(v) for k,v in (line.split() for line in row['events'].splitlines())} for row in sampled}
        if baseline_events is None:baseline_events=parsed
        if set(parsed)!=set(baseline_events):raise RuntimeError('Bounded cgroup hierarchy changed')
        for row in sampled:
            current=parsed[row['path']];initial=baseline_events[row['path']]
            if set(current)!=set(initial) or any(current[k]<initial[k] for k in current):
                raise RuntimeError('cgroup memory event counters changed identity or decreased')
            row['events_delta']={k:current[k]-initial[k] for k in current}
        record=dict(schema='q38-cgroup-live-snapshot-v1',run_id=args.run_id,node=args.node,
          hostname=host,container_id=args.container_id,identity_verified=True,init_pid=pid,
          init_start_ticks=ticks,sampler_pid=os.getpid(),sampler_start_ticks=sampler_ticks,
          same_time_namespace=True,time_namespace=host_time_ns,sample_started_monotonic=sample_started,
          sample_completed_monotonic=sample_completed,sample_unix=time.time(),cgroups=sampled)
        text=json.dumps(record,separators=(',',':'))+'\n'
        temp=output/'CURRENT.json.tmp';temp.write_text(text);os.replace(temp,output/'CURRENT.json')
        with (output/'samples.jsonl').open('a') as stream:stream.write(text)
        time.sleep(args.interval)
    print('CGROUP_SAMPLER_NORMAL_STOP',json.dumps(dict(run_id=args.run_id,node=args.node)),flush=True)

if __name__=='__main__':main()
