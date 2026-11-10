"""Strict task-owned 4+4 Ray identity gate, private candidate.

Read-only until explicitly called with an already attached Ray instance. It
does not start Ray, allocate a GPU, kill a process, or create a claim.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import uuid

EXPECTED_ROLE_IPS = {'training': '10.44.49.51', 'rollout': '10.44.33.31'}


def _uuid(value):
    value = str(value).removeprefix('GPU-').removeprefix('gpu-')
    return str(uuid.UUID(value))


def load_claim(env=None):
    env = os.environ if env is None else env
    run = Path(env['Q36_RUN_ROOT']).resolve(strict=True)
    run_id = run.name
    if not re.fullmatch(r'q36-disagg-[a-z0-9-]+-[0-9a-f]{10}', run_id):
        raise ValueError('fresh run name must end in a ten-hex identifier')
    path = Path(env['Q36_RAY_CLAIM_JSON'])
    if path.is_symlink() or not path.is_file() or path.resolve().parent != run:
        raise ValueError('task-owned Ray claim must be a regular file inside the run root')
    raw = path.read_bytes()
    digest = env['Q36_RAY_CLAIM_SHA256']
    if not re.fullmatch(r'[0-9a-f]{64}', digest) or hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError('task-owned Ray claim digest differs')
    claim = json.loads(raw)
    if claim.get('schema') != 'q36-owned-ray-4plus4-v1' or claim.get('run_id') != run_id:
        raise ValueError('Ray claim belongs to another run')
    session = claim.get('session', {})
    head = claim.get('training', {}).get('supervisor', {})
    if (session.get('name') != env.get('Q36_RAY_SESSION_NAME') or
        session.get('gcs_address') != env.get('Q36_RAY_ADDRESS') or
        not str(session.get('name', '')).endswith('_' + str(head.get('pid', ''))) or
        not session.get('cluster_id')):
        raise ValueError('Ray claim session identity is incomplete or inconsistent')
    expected_dir = str(Path('/tmp') / ('q36r-' + run_id.rsplit('-', 1)[-1]) / session['name'])
    if session.get('session_dir') != expected_dir:
        raise ValueError('Ray session directory is outside this run')
    roles = [claim.get('training', {}), claim.get('rollout', {})]
    if any(claim.get(name, {}).get('ip') != ip for name, ip in EXPECTED_ROLE_IPS.items()):
        raise ValueError('4+4 claim must bind the prepared training and rollout nodes')
    if len({row.get('ip') for row in roles}) != 2 or len({row.get('node_id') for row in roles}) != 2:
        raise ValueError('training and rollout must use two distinct Ray nodes')
    allowed = {item.strip() for item in env.get('Q36_DISAGG_ALLOWED_NODE_IPS', '').split(',') if item.strip()}
    if {row.get('ip') for row in roles} != allowed or len(allowed) != 2:
        raise ValueError('Ray role nodes differ from the explicit two-node allowlist')
    devices = []
    for role, row in zip(('head', 'worker'), roles):
        if not row.get('node_id') or len(row.get('gpu_uuids', [])) != 4:
            raise ValueError('each Ray role requires one node and four GPU UUIDs')
        devices.extend(_uuid(item) for item in row['gpu_uuids'])
        supervisor = row.get('supervisor', {})
        if (supervisor.get('role') != role or type(supervisor.get('pid')) is not int or
            supervisor['pid'] <= 0 or not isinstance(supervisor.get('create_time'), (int, float)) or
            supervisor['create_time'] <= 0 or type(supervisor.get('monitor_pid')) is not int or
            supervisor['monitor_pid'] <= 0 or not isinstance(supervisor.get('monitor_create_time'), (int, float)) or
            supervisor['monitor_create_time'] <= 0):
            raise ValueError('Ray supervisor identity is incomplete')
        task_root = Path(env['Q36_RAY_SUPERVISOR_ROOT']).resolve(strict=True)
        expected_status = task_root / run_id / ('ray-' + role + '.status.json')
        if Path(supervisor.get('status_file', '')).resolve() != expected_status:
            raise ValueError('Ray supervisor status path differs from this run')
    if len(set(devices)) != 8:
        raise ValueError('training and rollout GPU UUIDs overlap or repeat')
    return claim


def verify_attached_session(ray, claim):
    ctx = ray.get_runtime_context()
    session = claim['session']
    address = ctx.address_info
    if (ctx.get_session_name() != session['name'] or address.get('session_dir') != session['session_dir'] or
        address.get('gcs_address') != session['gcs_address'] or
        str(ray._private.worker.global_worker.node.cluster_id) != session['cluster_id']):
        raise RuntimeError('attached Ray session does not match the frozen task claim')
    nodes = [row for row in ray.nodes() if row.get('Alive')]
    roles = [claim['training'], claim['rollout']]
    if len(nodes) != 2 or {r['NodeManagerAddress'] for r in nodes} != {r['ip'] for r in roles}:
        raise RuntimeError('attached Ray cluster is not exactly the claimed two nodes')
    by_ip = {row['NodeManagerAddress']: row for row in nodes}
    for role in roles:
        row = by_ip[role['ip']]
        if row.get('NodeID') != role['node_id'] or row.get('Resources', {}).get('GPU', 0) != 4:
            raise RuntimeError('attached Ray node identity or GPU resource count differs')


def inspect_supervisor(run_id, node_ip, session_name, session_dir, supervisor):
    """Execute as a zero-GPU, node-affined Ray task in the requested container."""
    import psutil
    import ray
    if ray.util.get_node_ip_address() != node_ip:
        raise RuntimeError('supervisor inspection ran on the wrong node')
    path = Path(supervisor['status_file'])
    if path.is_symlink() or not path.is_file():
        raise RuntimeError('supervisor identity record is missing')
    status = json.loads(path.read_bytes())
    pid, created, role = supervisor['pid'], supervisor['create_time'], supervisor['role']
    if (status.get('run_id') != run_id or status.get('role') != role or status.get('state') != 'RUNNING' or
        status.get('ray_start_pid') != pid or status.get('ray_start_returncode') is not None or
        status.get('supervisor_pid') != supervisor['monitor_pid'] or
        abs(float(status.get('supervisor_create_time', 0)) - supervisor['monitor_create_time']) >= .001):
        raise RuntimeError('Ray supervisor status does not match the task claim')
    monitor = psutil.Process(supervisor['monitor_pid'])
    if (abs(monitor.create_time() - supervisor['monitor_create_time']) >= .001 or not monitor.is_running() or
            monitor.status() in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD)):
        raise RuntimeError('Ray monitor identity has exited or changed')
    process = psutil.Process(pid)
    if (abs(process.create_time() - float(created)) >= .001 or not process.is_running() or
            process.status() in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD) or process.ppid() != monitor.pid):
        raise RuntimeError('Ray launcher identity has exited or changed')
    descendants = {(child.pid, child.create_time()) for child in process.children(recursive=True)
                   if child.is_running()}
    required = {'raylet', 'gcs_server'} if role == 'head' else {'raylet'}
    recorded = [row for row in status.get('live', []) if row.get('name') in required]
    if {row['name'] for row in recorded} != required:
        raise RuntimeError('Ray supervisor omitted required services')
    for row in recorded:
        child = psutil.Process(int(row['pid']))
        if ((child.pid, child.create_time()) not in descendants or
            abs(child.create_time() - float(row['create_time'])) >= .001 or
            not child.is_running() or child.status() in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD)):
            raise RuntimeError('Ray service is not a live verified descendant')
    if Path(session_dir).name != session_name or not Path(session_dir).is_dir():
        raise RuntimeError('Ray supervisor session directory is absent')
    return {'role': role, 'ip': node_ip, 'pid': pid, 'create_time': created,
            'service_pids': sorted(row['pid'] for row in recorded)}


def verify_supervisors(ray, claim):
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
    task = ray.remote(num_cpus=0)(inspect_supervisor)
    futures = []
    for key in ('training', 'rollout'):
        role = claim[key]
        futures.append(task.options(scheduling_strategy=NodeAffinitySchedulingStrategy(role['node_id'], soft=False)).remote(
            claim['run_id'], role['ip'], claim['session']['name'], claim['session']['session_dir'], role['supervisor']))
    return ray.get(futures, timeout=30)


def validate_role_ownership(rows, role, claim):
    if role not in ('training', 'rollout'):
        raise ValueError('unsupported 4+4 role')
    expected = claim[role]
    if (len(rows) != 4 or {row['node_ip'] for row in rows} != {expected['ip']} or
        {str(row['node_id']) for row in rows} != {expected['node_id']} or
        {_uuid(row['gpu_uuid']) for row in rows} != {_uuid(value) for value in expected['gpu_uuids']}):
        raise RuntimeError(role + ' worker allocation does not match the task-owned GPU claim')


def validate_pool_ownership(ownership, claim):
    for role in ('training', 'rollout'):
        validate_role_ownership(ownership[role], role, claim)
