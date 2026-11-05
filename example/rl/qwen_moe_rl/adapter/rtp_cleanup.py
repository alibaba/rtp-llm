"""Driver cleanup scoped to one run's recorded RTP process identities.

This module deliberately imports neither Torch nor Ray at module import time.
It can run after a trainer constructor or Ray actor failed before normal cleanup.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import time

import psutil


def _write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def _live(process):
    try:
        return process.is_running() and process.status() not in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD)
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return False


def _owned(identity):
    result = {int(identity['pid']): float(identity['create_time'])}
    for item in identity.get('descendants', []):
        result[int(item['pid'])] = float(item['create_time'])
    # Descendants can be discovered only through an identity-verified process,
    # never by command-line substring, CUDA device, or another Ray worker.
    for pid, created in list(result.items()):
        try:
            process = psutil.Process(pid)
            if process.create_time() != created or not _live(process):
                continue
            for child in process.children(recursive=True):
                result[child.pid] = child.create_time()
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
    return result


def _terminate_identity(identity):
    owned = _owned(identity)
    targets = []
    for pid, created in owned.items():
        try:
            process = psutil.Process(pid)
            # A reused PID is unrelated. Do not signal it.
            if process.create_time() == created and _live(process):
                if pid == identity['pid'] and os.getpgid(pid) != identity['pgid']:
                    raise RuntimeError('RTP leader changed process group; refusing termination')
                targets.append(process)
        except (psutil.NoSuchProcess, psutil.ZombieProcess, ProcessLookupError):
            continue
    for process in targets:
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(targets, timeout=20)
    for process in alive:
        try:
            # psutil validates process creation time when signalling.
            process.kill()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(alive, timeout=10)
    survivors = [process.pid for process in alive if _live(process)]
    if survivors:
        raise RuntimeError(f'Owned native RTP processes survived cleanup: {survivors}')
    unknown = []
    for process in psutil.process_iter(['pid', 'status', 'create_time']):
        try:
            # Observe this original group, but never kill unrecorded members.
            if (_live(process) and os.getpgid(process.pid) == identity['pgid']
                    and process.info['create_time'] >= identity['create_time']):
                unknown.append(process.pid)
        except (psutil.NoSuchProcess, psutil.ZombieProcess, ProcessLookupError):
            continue
    if unknown:
        raise RuntimeError(f'Unverified processes remain in original RTP group: {unknown}')
    return {'process_tree_terminated': True, 'owned_pids': sorted(owned),
            'pid': identity['pid'], 'create_time': identity['create_time'],
            'pgid': identity['pgid'], 'finished_at_unix': time.time()}


def _cleanup_multinode(identity, root):
    """Remote PIDs are verified and signalled only by their local owner actor."""
    import ray
    if not ray.is_initialized():
        raise RuntimeError('A live Ray cluster is required to verify cross-node native cleanup')
    timeout=float(os.environ.get('RLLM_RTP_CLEANUP_RPC_TIMEOUT','90'))
    actor=identity['ray_actor']
    controller_error=None
    try:
        handle=ray.get_actor(actor['name'],namespace=actor['namespace'])
        result=ray.get(handle.shutdown_if_owned.remote(int(identity['pid']),
            float(identity['create_time']),str(root)),timeout=timeout)
        if (not result.get('process_tree_terminated') or len(result.get('nodes',[]))!=identity['node_owner_count']
                or any(not node.get('process_tree_terminated') or node.get('owned_gpu_processes_after')
                       for node in result['nodes'])):
            raise RuntimeError('Cluster controller did not acknowledge cleanup on every native node')
        return dict(method='owned_multinode_controller_rpc',rpc_receipt=result)
    except Exception as error:
        controller_error=repr(error)
    owners=identity.get('node_owners',[])
    if len(owners)!=identity['node_owner_count']:
        raise RuntimeError('Cluster controller is unavailable and local owner identities are incomplete: '+controller_error)
    pending=[]
    errors=[]
    for owner in owners:
        try:
            if Path(owner['audit_dir']).resolve()!=root:
                raise ValueError('Native owner belongs to another audit directory')
            actor=owner['ray_actor']
            handle=ray.get_actor(actor['name'],namespace=actor['namespace'])
            pending.append(handle.stop_if_owned.remote(int(owner['pid']),float(owner['create_time']),str(root)))
        except Exception as error:
            errors.append(repr(error))
    results=[]
    for future in pending:
        try:
            results.append(ray.get(future,timeout=timeout))
        except Exception as error:
            errors.append(repr(error))
    if errors:
        raise RuntimeError('Native node cleanup failed after attempting every owner: '+str(errors))
    if (len({item['node_index'] for item in results})!=len(owners)
            or any(not item.get('process_tree_terminated') or item.get('owned_gpu_processes_after') for item in results)):
        raise RuntimeError('Individual owners did not acknowledge complete cross-node cleanup')
    return dict(method='owned_multinode_owner_rpc',controller_error=controller_error,
                rpc_receipt=dict(process_tree_terminated=True,nodes=results))


def _discover_required_clusters(root, expected_replicas):
    """Copy missing owned controller identities from live Ray to the driver.

    Node-local audit directories do not imply that the trainer driver received
    a replica's identity. Discover only the explicitly expected named actors in
    this driver's namespace, and verify deployment/run before recording them.
    """
    if not 1 <= expected_replicas <= 32:
        raise ValueError('Invalid explicit native replica count')
    import ray
    if not ray.is_initialized():
        raise RuntimeError('Live Ray is required to discover distributed cleanup identities')
    namespace = ray.get_runtime_context().namespace
    deployment = os.environ.get('RLLM_RTP_DEPLOY_ROOT') or os.environ.get('RTP_DEPLOY_ROOT')
    if not deployment:
        raise ValueError('An explicit deployment root is required for native discovery')
    errors = []
    for replica in range(expected_replicas):
        path = root/f'rtp_cluster_{replica}.json'
        if path.exists():
            continue
        try:
            name = f'rtp_server_{replica}_0'
            actor = ray.get_actor(name, namespace=namespace)
            identity = ray.get(actor.prepare.remote(), timeout=float(
                os.environ.get('RLLM_RTP_CLEANUP_RPC_TIMEOUT', '90')))
            if (identity['replica_rank'] != replica or Path(identity['root']).resolve() != Path(deployment).resolve()
                    or Path(identity['audit_dir']).resolve() != root
                    or identity['ray_actor'] != dict(name=name, namespace=namespace)
                    or identity['node_owner_count'] != 4 or len(identity['node_owners']) != 4):
                raise ValueError('Discovered native controller belongs to another deployment, run, or topology')
            _write(path, identity)
        except Exception as error:
            errors.append(f'cluster discovery replica {replica}: {error!r}')
    return errors


def cleanup_owned_rtp_processes(audit_dir):
    """Synchronously stop only this run's native RTP replicas, or raise.

    Call before ray.shutdown(). Named actor shutdown is preferred. Its native
    leader identity must match this run's record before it can terminate anything.
    If Ray is unavailable, process cleanup uses PID AND creation time. A durable
    driver-cleanup.json is written on success and failure; failures are not hidden.
    """
    root = Path(audit_dir).resolve()
    configured = os.environ.get('RLLM_RTP_ACCEPTANCE_DIR')
    if configured is None or root != Path(configured).resolve():
        raise ValueError('Cleanup is restricted to the current RLLM_RTP_ACCEPTANCE_DIR')
    root.mkdir(parents=True, exist_ok=True)
    receipt = {'audit_dir': str(root), 'started_at_unix': time.time(), 'replicas': []}
    errors = []
    expected = os.environ.get('RLLM_RTP_EXPECTED_REPLICAS')
    if expected is not None:
        try:
            errors.extend(_discover_required_clusters(root, int(expected)))
        except Exception as error:
            errors.append(f'cluster discovery: {error!r}')
        receipt['expected_native_replicas'] = int(expected)
    for path in sorted(root.glob('rtp_cluster_*.json')):
        if not re.fullmatch(r'rtp_cluster_\d+\.json',path.name):
            continue
        item={'identity_file':str(path)}
        try:
            identity=json.loads(path.read_text())
            if path.name!=f"rtp_cluster_{int(identity['replica_rank'])}.json":
                raise ValueError('Cluster identity filename mismatch')
            expected_deployment=os.environ.get('RLLM_RTP_DEPLOY_ROOT') or os.environ.get('RTP_DEPLOY_ROOT')
            if expected_deployment and Path(identity['root']).resolve()!=Path(expected_deployment).resolve():
                raise ValueError('Recorded cluster deployment differs from this deployment')
            if Path(identity['audit_dir']).resolve()!=root:
                raise ValueError('Recorded cluster audit directory differs from the current run')
            item.update(_cleanup_multinode(identity,root),passed=True)
        except Exception as error:
            item.update(passed=False,error=repr(error))
            errors.append(f'{path.name}: {error}')
        receipt['replicas'].append(item)
    for path in sorted(root.glob('rtp_process_*.json')):
        item = {'identity_file': str(path)}
        try:
            identity = json.loads(path.read_text())
            if path.name != f"rtp_process_{int(identity['replica_rank'])}.json":
                raise ValueError('Replica identity filename mismatch')
            expected_deployment = os.environ.get('RLLM_RTP_DEPLOY_ROOT') or os.environ.get('RTP_DEPLOY_ROOT')
            if expected_deployment and Path(identity['root']).resolve() != Path(expected_deployment).resolve():
                raise ValueError('Recorded RTP deployment root differs from current deployment')
            if identity.get('audit_dir') and Path(identity['audit_dir']).resolve() != root:
                raise ValueError('Recorded RTP audit directory differs from current run')
            rpc_error = None
            actor = identity.get('ray_actor')
            if actor:
                try:
                    import ray
                    if not ray.is_initialized():
                        raise RuntimeError('Ray runtime is no longer initialized')
                    handle = ray.get_actor(actor['name'], namespace=actor['namespace'])
                    result = ray.get(handle.shutdown_if_owned.remote(
                        int(identity['pid']), float(identity['create_time']), str(root)),
                        timeout=float(os.environ.get('RLLM_RTP_CLEANUP_RPC_TIMEOUT', '45')))
                    if not result.get('process_tree_terminated'):
                        raise RuntimeError('Native shutdown RPC omitted termination acknowledgment')
                    item['rpc_receipt'] = result
                    item['method'] = 'owned_actor_rpc'
                except Exception as exc:
                    rpc_error = repr(exc)
            if rpc_error is not None:
                item['rpc_error'] = rpc_error
            # This is idempotent after a successful RPC and verifies its claim.
            item['process_receipt'] = _terminate_identity(identity)
            item.setdefault('method', 'identity_checked_process_fallback')
            item['passed'] = True
        except Exception as exc:
            item['passed'] = False
            item['error'] = repr(exc)
            errors.append(f'{path.name}: {exc}')
        receipt['replicas'].append(item)
    receipt.update(passed=not errors, finished_at_unix=time.time())
    _write(root / 'driver-cleanup.json', receipt)
    if errors:
        raise RuntimeError('Native RTP cleanup incomplete: ' + '; '.join(errors))
    return receipt
