"""Create a claim from a fresh, already-running task-owned Ray allocation.

No Ray cluster is started or stopped and no GPU is allocated by this program.
The caller must first start the explicitly scoped 4+4 supervisors, and is
responsible for their eventual identity-checked cleanup.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


def inspect_node(run_id, expected_ip, role, status_file):
    """Read locally observed supervisor and physical GPU identities on a node."""
    import psutil
    import ray
    if ray.util.get_node_ip_address() != expected_ip:
        raise RuntimeError('claim inspection ran on the wrong node')
    path = Path(status_file)
    if path.is_symlink() or not path.is_file():
        raise RuntimeError('supervisor status must be a regular local file')
    status = json.loads(path.read_bytes())
    if (status.get('run_id') != run_id or status.get('role') != role or
            status.get('state') != 'RUNNING' or status.get('ray_start_returncode') is not None):
        raise RuntimeError('supervisor status is not the claimed running allocation')
    pid = status['ray_start_pid']
    monitor_pid = status['supervisor_pid']
    if type(pid) is not int or type(monitor_pid) is not int:
        raise RuntimeError('supervisor process identities must be integer PIDs')
    proc, monitor = psutil.Process(pid), psutil.Process(monitor_pid)
    if (proc.ppid() != monitor_pid or not proc.is_running() or not monitor.is_running() or
            abs(monitor.create_time() - float(status['supervisor_create_time'])) >= .001):
        raise RuntimeError('supervisor or launcher identity changed')
    output = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader,nounits'],
        text=True, timeout=15)
    gpu_uuids = [line.strip() for line in output.splitlines() if line.strip()]
    if len(gpu_uuids) != 4 or len(set(gpu_uuids)) != 4 or any(not u.startswith('GPU-') for u in gpu_uuids):
        raise RuntimeError('node must expose exactly four distinct physical GPU UUIDs')
    return {'gpu_uuids': gpu_uuids, 'supervisor': {
        'status_file': str(path), 'role': role, 'pid': pid, 'create_time': proc.create_time(),
        'monitor_pid': monitor_pid, 'monitor_create_time': monitor.create_time()}}


def assemble_claim(run_id, session, node_rows, observed, expected_ips):
    if len(node_rows) != 2 or len(observed) != 2 or len(set(expected_ips.values())) != 2:
        raise RuntimeError('expected exactly two distinct nodes')
    by_ip = {row.get('NodeManagerAddress'): row for row in node_rows if row.get('Alive')}
    if len(by_ip) != 2 or set(by_ip) != set(expected_ips.values()):
        raise RuntimeError('live Ray nodes differ from the two authorized roles')
    claim = {'schema': 'q36-owned-ray-4plus4-v1', 'run_id': run_id, 'session': session}
    for index, role in enumerate(('training', 'rollout')):
        ip = expected_ips[role]
        row = by_ip[ip]
        if row.get('Resources', {}).get('GPU') != 4 or not row.get('NodeID'):
            raise RuntimeError('each Ray node must expose exactly four GPUs')
        claim[role] = {'ip': ip, 'node_id': row['NodeID'], **observed[index]}
    if claim['training']['node_id'] == claim['rollout']['node_id']:
        raise RuntimeError('Ray node IDs overlap')
    return claim


def install_peer_claim(run_root, expected_ip, data, expected_sha):
    """Atomically install and read back exact claim bytes on rollout node."""
    import ray
    if ray.util.get_node_ip_address() != expected_ip:
        raise RuntimeError('claim copy ran on the wrong node')
    if not isinstance(data, bytes) or len(data) > 1024 * 1024 or hashlib.sha256(data).hexdigest() != expected_sha:
        raise RuntimeError('incoming claim content differs')
    run = Path(run_root).resolve(strict=True)
    final = run / 'ray-claim.json'
    if final.exists() or final.is_symlink():
        raise FileExistsError('rollout claim already exists; do not overwrite')
    fd, name = tempfile.mkstemp(prefix='.ray-claim-peer-', dir=run)
    temporary = Path(name)
    try:
        with os.fdopen(fd, 'wb') as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.link(temporary, final, follow_symlinks=False)
        readback = final.read_bytes()
        if readback != data or hashlib.sha256(readback).hexdigest() != expected_sha:
            raise RuntimeError('rollout claim readback differs')
    finally:
        temporary.unlink(missing_ok=True)
    return {'node_ip': expected_ip, 'claim_path': str(final), 'claim_sha256': expected_sha}


def make_claim(ray, guard, env):
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
    run = Path(env['Q36_RUN_ROOT']).resolve(strict=True)
    run_id = run.name
    supervisor_root = Path(env['Q36_RAY_SUPERVISOR_ROOT']).resolve(strict=True)
    expected_ips = guard.EXPECTED_ROLE_IPS
    nodes = ray.nodes()
    by_ip = {row.get('NodeManagerAddress'): row for row in nodes if row.get('Alive')}
    if len(nodes) != 2 or set(by_ip) != set(expected_ips.values()):
        raise RuntimeError('Ray allocation has unexpected nodes')
    ctx = ray.get_runtime_context()
    address = ray._private.worker.global_worker.node.address_info
    session = {'name': ctx.get_session_name(), 'session_dir': address['session_dir'],
               'gcs_address': address['gcs_address'],
               'cluster_id': str(ray._private.worker.global_worker.node.cluster_id)}
    task = ray.remote(num_cpus=0)(inspect_node)
    refs = []
    for role, supervisor_role in (('training', 'head'), ('rollout', 'worker')):
        ip = expected_ips[role]
        status_file = supervisor_root / run_id / ('ray-' + supervisor_role + '.status.json')
        refs.append(task.options(scheduling_strategy=NodeAffinitySchedulingStrategy(
            by_ip[ip]['NodeID'], soft=False)).remote(run_id, ip, supervisor_role, str(status_file)))
    observed = ray.get(refs, timeout=30)
    claim = assemble_claim(run_id, session, nodes, observed, expected_ips)
    data = (json.dumps(claim, sort_keys=True, indent=2) + '\n').encode()
    digest = hashlib.sha256(data).hexdigest()
    final = run / 'ray-claim.json'
    if final.exists() or final.is_symlink():
        raise FileExistsError('a run claim already exists; do not replay or overwrite')
    fd, temp_name = tempfile.mkstemp(prefix='.ray-claim-', dir=run)
    temp = Path(temp_name)
    try:
        with os.fdopen(fd, 'wb') as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        gate_env = dict(env, Q36_RAY_CLAIM_JSON=str(temp), Q36_RAY_CLAIM_SHA256=digest)
        loaded = guard.load_claim(gate_env)
        guard.verify_attached_session(ray, loaded)
        guard.verify_supervisors(ray, loaded)
        # hard-link is atomic and refuses to replace an existing claim
        os.link(temp, final, follow_symlinks=False)
    finally:
        temp.unlink(missing_ok=True)
    if final.read_bytes() != data or hashlib.sha256(final.read_bytes()).hexdigest() != digest:
        raise RuntimeError('training claim readback differs')
    peer_task = ray.remote(num_cpus=0)(install_peer_claim)
    peer = ray.get(peer_task.options(scheduling_strategy=NodeAffinitySchedulingStrategy(
        claim['rollout']['node_id'], soft=False)).remote(str(run), claim['rollout']['ip'], data, digest), timeout=30)
    expected_peer = {'node_ip': claim['rollout']['ip'], 'claim_path': str(final), 'claim_sha256': digest}
    if peer != expected_peer:
        raise RuntimeError('rollout claim identity or readback differs')
    ready = {'run_id': run_id, 'claim_sha256': digest, 'claim_path': str(final),
             'training_node_ip': claim['training']['ip'], 'rollout_readback': peer}
    # An absent ready receipt means this run must not be launched or retried.
    with (run / 'ray-claim-ready.json').open('x') as output:
        json.dump(ready, output, sort_keys=True)
        output.flush()
        os.fsync(output.fileno())
    return {'claim_path': str(final), 'claim_sha256': digest, 'run_id': run_id,
            'session_name': session['name'], 'cluster_id': session['cluster_id'],
            'rollout_claim_readback': peer}


def main():
    import ray
    from q36_4plus4_claim_loader import load_guard
    address = os.environ['Q36_RAY_ADDRESS']
    if address != '10.44.49.51:31100':
        raise RuntimeError('an explicit task-owned head address is required')
    guard = load_guard()
    ray.init(address=address, logging_level='ERROR')
    try:
        print(json.dumps(make_claim(ray, guard, os.environ), sort_keys=True), flush=True)
    finally:
        ray.shutdown()  # disconnect this driver only; does not stop the cluster


if __name__ == '__main__':
    main()
