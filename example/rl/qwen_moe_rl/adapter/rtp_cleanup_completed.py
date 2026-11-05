"""Read-only proof of a previously completed native multi-node shutdown.

This fallback never imports Torch, creates a CUDA context, sends a signal, or
uses a missing actor as proof of cleanup. Every native identity is checked on
its original physical Ray node before the prior shutdown can be reused.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time

import psutil


_MAX_JSON_BYTES = 64 * 1024 * 1024


def _require(condition, message):
    if not condition:
        raise RuntimeError('Completed native shutdown verification: ' + message)


def _uuid(value):
    return str(value).lower().removeprefix('gpu-').replace('-', '')


def _record(path, root):
    path, root = Path(path).resolve(strict=True), Path(root).resolve(strict=True)
    _require(path.is_relative_to(root), 'metadata path escapes the original audit directory')
    before = path.stat()
    _require(0 < before.st_size <= _MAX_JSON_BYTES, 'metadata size is out of bounds')
    data = path.read_bytes()
    after = path.stat()
    _require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
             'metadata changed during inspection')
    return json.loads(data), dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def _identity_shape(identity, root):
    root = Path(root).resolve(strict=True)
    replica = identity['replica_rank']
    _require(type(replica) is int and 0 <= replica < 32, 'invalid replica identity')
    _require(Path(identity['audit_dir']).resolve() == root, 'identity belongs to another run')
    deployment = Path(identity['root']).resolve(strict=True)
    _require(root != deployment and root.is_relative_to(deployment), 'audit directory is outside deployment')
    owners = identity['node_owners']
    topology = identity['topology']
    groups = topology['node_groups']
    count = identity['node_owner_count']
    _require(type(count) is int and count > 0 and len(owners) == len(groups) == count,
             'incomplete node-owner topology')
    _require(topology['replica_rank'] == replica, 'topology replica differs')
    ranks = [rank for group in groups for rank in group['ranks']]
    _require(sorted(ranks) == list(range(topology['tp_size'])), 'TP rank topology is incomplete')
    _require(len({group['host'] for group in groups}) == count, 'owner hosts are not unique')
    for index, (owner, group) in enumerate(zip(owners, groups, strict=True)):
        _require(owner['node_index'] == index and owner['owner_ip'] == group['host'],
                 'node owner differs from recorded topology')
        _require(Path(owner['audit_dir']).resolve() == root and Path(owner['root']).resolve() == deployment,
                 'node owner belongs to another deployment or run')
    return root, replica, owners, groups


def _canonical_bundle(identity, root):
    """Run only on the replica leader; return original hashes plus compact bindings."""
    root, replica, owners, groups = _identity_shape(identity, root)
    recorded, cluster_file = _record(root/f'rtp_cluster_{replica}.json', root)
    _require(recorded == identity, 'cluster identity changed after the driver recorded it')
    canonical_path=root/f'rtp_cluster_shutdown_{replica}.json'
    if not canonical_path.exists():
        return None
    canonical, canonical_file = _record(canonical_path, root)
    _require(canonical.get('replica_rank') == replica, 'canonical shutdown belongs to another replica')
    if (canonical.get('process_tree_terminated') is not True or
            not isinstance(canonical.get('nodes'), list) or len(canonical['nodes']) != len(owners) or
            any(item.get('process_tree_terminated') is not True or item.get('owned_gpu_processes_after') != []
                for item in canonical['nodes'])):
        return None  # No completed proof: preserve the existing owned stop-RPC path.
    boots = []
    for path in root.glob(f'rtp_boot_{replica}_step*.json'):
        match = re.fullmatch(rf'rtp_boot_{replica}_step(\d+)\.json', path.name)
        if match:
            value, original = _record(path, root)
            _require(value['target_global_step'] == int(match.group(1)), 'boot filename/step differs')
            boots.append((int(value['native_boot']), int(value['target_global_step']), value, original))
    _require(bool(boots), 'there is no original successful native boot')
    _, _, boot, boot_file = max(boots, key=lambda item: item[:2])
    _require(boot.get('passed') is True and boot['replica_rank'] == replica,
             'latest native boot did not pass for this replica')
    _require(boot['manifest_count'] == identity['topology']['tp_size'], 'boot TP manifest coverage differs')
    _require(canonical.get('process_tree_terminated') is True and canonical.get('pause_error') is None,
             'canonical shutdown did not finish cleanly')
    _require((canonical['replica_rank'], canonical['native_boot'], canonical['native_incarnation']) ==
             (replica, boot['native_boot'], boot['native_incarnation']),
             'canonical shutdown is stale or belongs to another incarnation')
    syncs = []
    for path in root.glob(f'rtp_sync_rank{replica}_step*.json'):
        match = re.fullmatch(rf'rtp_sync_rank{replica}_step(\d+)\.json', path.name)
        if match:
            value, original = _record(path, root)
            _require(value['global_step'] == int(match.group(1)), 'sync filename/step differs')
            syncs.append((int(value['global_step']), value, original))
    _require(bool(syncs), 'there is no original committed policy sync')
    _, sync, sync_file = max(syncs, key=lambda item: item[0])
    version = sync['model_version']
    _require((sync['replica_rank'], sync['native_boot'], sync['native_incarnation']) ==
             (replica, boot['native_boot'], boot['native_incarnation']),
             'latest committed sync and native boot differ')
    _require(version.startswith(boot['native_incarnation'] + ':') and
             canonical['previous_committed_version'] == version,
             'canonical shutdown is not bound to the latest committed policy')
    _require(re.fullmatch('[0-9a-f]{64}', sync['full_source_sha256']) is not None,
             'latest committed source digest is invalid')
    statuses = sync['rank_status']
    _require(sorted(item['tp_rank'] for item in statuses) == list(range(identity['topology']['tp_size'])) and
             all(item['model_version'] == version for item in statuses), 'latest commit rank ACKs are incomplete')
    _require(sync['hf_tensor_count'] > 0 and sync['native_received_weights'] == sync['hf_tensor_count'] and
             sync['native_applied_weights'] > 0, 'latest sync omitted complete native apply acknowledgement')
    _require(canonical['finished_at_unix'] >= sync['finished_unix'], 'shutdown predates the latest sync')
    _require(len(canonical['nodes']) == len(owners) == len(boot['nodes']), 'canonical node coverage is incomplete')
    shutdown_nodes = {item['node_index']: item for item in canonical['nodes']}
    boot_nodes = {item['node_index']: item for item in boot['nodes']}
    _require(set(shutdown_nodes) == set(boot_nodes) == set(range(len(owners))), 'duplicate or missing node indices')
    nodes = []
    for index, (owner, group) in enumerate(zip(owners, groups, strict=True)):
        stopped, started = shutdown_nodes[index], boot_nodes[index]
        process = started['process_identity']
        _require(stopped.get('process_tree_terminated') is True and stopped.get('owned_gpu_processes_after') == [],
                 'a canonical node did not release all owned native contexts')
        for key in ('pid', 'create_time', 'pgid'):
            _require(stopped[key] == process[key], 'canonical native process identity differs from boot')
        _require((stopped['owner_ip'], stopped['owner_hostname'], stopped['boot_id']) ==
                 (owner['owner_ip'], owner['owner_hostname'], boot['native_boot']), 'canonical node placement differs')
        _require(stopped['finished_at_unix'] >= sync['finished_unix'], 'node shutdown predates latest policy')
        _require((process['replica_rank'], process['boot_id'], process['incarnation'], process['node_index']) ==
                 (replica, boot['native_boot'], boot['native_incarnation'], index), 'boot process binding differs')
        ranks = started['rank_manifests']
        _require(sorted(row['tp_rank'] for row in ranks) == sorted(group['ranks']), 'boot local TP ranks differ')
        proofs = {row['tp_rank']: row for row in started['nccl_scope_proofs']}
        _require(set(proofs) == set(group['ranks']), 'native rank creation-time evidence is incomplete')
        native = []
        for rank in ranks:
            proof = proofs[rank['tp_rank']]
            _require(proof['pid'] == rank['pid'] and _uuid(proof['gpu_uuid']) == _uuid(rank['gpu_uuid']) and
                     proof['native_incarnation'] == boot['native_incarnation'] and
                     proof['owner_ip'] == owner['owner_ip'], 'native rank PID/UUID/incarnation binding differs')
            native.append(dict(tp_rank=rank['tp_rank'], pid=rank['pid'], create_time=proof['create_time'],
                               gpu_uuid=rank['gpu_uuid'], tp_size=rank['tp_size']))
        nodes.append(dict(node_index=index, owner=owner, group=group, canonical_node=stopped,
                          process_identity=process, native_ranks=native))
    return dict(schema='q397-completed-native-shutdown-binding-v1', replica_rank=replica,
                audit_dir=str(root), native_boot=boot['native_boot'], native_incarnation=boot['native_incarnation'],
                model_version=version, latest_sync_step=sync['global_step'],
                latest_source_sha256=sync['full_source_sha256'], latest_sync_finished_unix=sync['finished_unix'],
                canonical_shutdown=canonical, canonical_file=canonical_file, cluster_file=cluster_file,
                boot_file=boot_file, sync_file=sync_file, nodes=nodes)


def _gpu_inventory():
    devices = subprocess.run(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, check=True, timeout=8).stdout
    apps = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader,nounits'],
                          capture_output=True, text=True, check=True, timeout=8).stdout
    uuids = [line.strip() for line in devices.splitlines() if line.strip()]
    rows = []
    for parts in csv.reader(apps.splitlines()):
        if not parts:
            continue
        _require(len(parts) == 2 and parts[0].strip().isdigit(), 'unparseable NVML compute process row')
        rows.append(dict(pid=int(parts[0]), gpu_uuid=parts[1].strip()))
    _require(bool(uuids), 'NVML did not report the node physical GPUs')
    return dict(gpu_uuids=uuids, compute_processes=rows)


def _inspect_completed_node(identity, root, index, expected_source_sha256, bundle=None):
    """CPU-only, no-write, no-signal task pinned to one original native host."""
    _require('torch' not in sys.modules, 'CPU inspector was started in a Torch-importing worker')
    source = Path(__file__).read_bytes()
    _require(hashlib.sha256(source).hexdigest() == expected_source_sha256, 'inspector source SHA differs on this node')
    root, replica, owners, groups = _identity_shape(identity, root)
    owner = owners[index]
    _require(socket.gethostname() == owner['owner_hostname'] and
             owner['owner_ip'] in socket.gethostbyname_ex(socket.gethostname())[2], 'inspector is not on original native node')
    if bundle is None:
        _require(index == 0, 'canonical shutdown can only be read on recorded replica leader')
        bundle = _canonical_bundle(identity, root)
        if bundle is None:
            return dict(proof=None, binding=None, canonical_available=False)
    _require((bundle['audit_dir'], bundle['replica_rank']) == (str(root), replica), 'inspection bundle belongs to another run')
    local_boot, boot_file = _record(bundle['boot_file']['path'], root)
    _require(boot_file == bundle['boot_file'] and local_boot['native_incarnation'] == bundle['native_incarnation'],
             'node-local boot original differs from leader binding')
    # An in-progress later boot may have no published boot proof yet. Its
    # persisted launcher identity still makes the old completed proof stale.
    for path in root.glob(f'boots/boot-*/node-*/rtp_process_{replica}.json'):
        candidate, _ = _record(path, root)
        _require(candidate['boot_id'] <= bundle['native_boot'] and
                 (candidate['boot_id'] != bundle['native_boot'] or candidate['incarnation'] == bundle['native_incarnation']),
                 'a later or different native process incarnation has already started')
    binding = bundle['nodes'][index]
    _require(binding['node_index'] == index and binding['owner'] == owner, 'node index binding differs')
    own_root = root/f'boots/boot-{bundle["native_boot"]:03d}'/f'node-{index:02d}'
    process, process_file = _record(own_root/f'rtp_process_{replica}.json', root)
    stopped, stopped_file = _record(own_root/f'rtp_shutdown_{replica}.json', root)
    original = binding['process_identity']
    for key in ('pid', 'create_time', 'pgid', 'replica_rank', 'boot_id', 'incarnation', 'node_index', 'owner_ip', 'owner_hostname'):
        _require(process[key] == original[key], 'local persisted native process identity differs from boot')
    _require(Path(process['root']).resolve() == Path(identity['root']).resolve() and
             Path(process['audit_dir']).resolve() == own_root.resolve(), 'local native identity belongs to another run')
    canonical_node = binding['canonical_node']
    for key in ('pid', 'create_time', 'pgid', 'owned_pids', 'process_tree_terminated'):
        _require(stopped[key] == canonical_node[key], 'local shutdown receipt differs from canonical shutdown')
    _require(bundle['latest_sync_finished_unix'] <= stopped['finished_at_unix'] <= canonical_node['finished_at_unix'],
             'local shutdown receipt timing differs')
    owned = {process['pid']: process['create_time']}
    for child in process.get('descendants', []):
        _require(child['pid'] not in owned, 'duplicate persisted descendant PID')
        owned[child['pid']] = child['create_time']
    _require(set(owned) == set(canonical_node['owned_pids']), 'canonical owned PID list lacks persisted creation times')
    for rank in binding['native_ranks']:
        _require(owned.get(rank['pid']) == rank['create_time'], 'native rank is outside original PID/creation-time ownership')
        manifest, _ = _record(own_root/f'rtp_weight_manifest_{replica}'/f'rank-{rank["tp_rank"]:05d}.json', root)
        _require(all(manifest[key] == rank[key] for key in ('tp_rank', 'tp_size', 'pid')) and
                 _uuid(manifest['gpu_uuid']) == _uuid(rank['gpu_uuid']), 'local native rank manifest differs')
    observed = []
    for pid, created in sorted(owned.items()):
        try:
            current = psutil.Process(pid)
            actual_created = current.create_time()
            status = current.status()
            if actual_created == created:
                _require(status in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD),
                         'an original owned native PID is still present')
                state = 'nonlive'
            else:
                state = 'pid_reused'
            observed.append(dict(pid=pid, expected_create_time=created, state=state,
                                 observed_create_time=actual_created, observed_status=status))
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            observed.append(dict(pid=pid, expected_create_time=created, state='absent'))
    group_members = []
    for current in psutil.process_iter(['pid', 'status']):
        try:
            if current.info['status'] not in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD) and os.getpgid(current.pid) == process['pgid']:
                group_members.append(current.pid)
        except (psutil.NoSuchProcess, psutil.ZombieProcess, ProcessLookupError):
            continue
    _require(not group_members, 'the original native PGID still has live members; no process was signalled')
    inventory = _gpu_inventory()
    _require({_uuid(row['gpu_uuid']) for row in binding['native_ranks']} <= {_uuid(value) for value in inventory['gpu_uuids']},
             'original physical GPU UUIDs are no longer present')
    reused = {item['pid'] for item in observed if item['state'] == 'pid_reused'}
    owned_contexts = [row for row in inventory['compute_processes'] if row['pid'] in owned and row['pid'] not in reused]
    _require(not owned_contexts, 'an owned native PID still has a GPU context')
    _require('torch' not in sys.modules, 'CPU inspection unexpectedly imported Torch')
    proof = dict(schema='q397-completed-native-node-inspection-v1', passed=True, replica_rank=replica,
                 native_boot=bundle['native_boot'], native_incarnation=bundle['native_incarnation'],
                 model_version=bundle['model_version'], latest_sync_step=bundle['latest_sync_step'],
                 latest_source_sha256=bundle['latest_source_sha256'], node_index=index,
                 owner_ip=owner['owner_ip'], owner_hostname=socket.gethostname(), inspector_pid=os.getpid(),
                 inspector_source_sha256=expected_source_sha256, inspected_unix=time.time(),
                 native_identities=observed, original_pgid=process['pgid'], live_original_group_members=group_members,
                 owned_native_gpu_contexts=owned_contexts, inventory=inventory, torch_imported=False,
                 signals_sent=False, metadata=dict(process=process_file, shutdown=stopped_file, boot=boot_file),
                 canonical_shutdown_sha256=bundle['canonical_file']['sha256'], latest_sync_file=bundle['sync_file'])
    return dict(proof=proof, binding=bundle if index == 0 else None)


def verify_completed_multinode_shutdown(identity, root, timeout):
    """Return a positive prior-shutdown proof, or None if no completed receipt exists."""
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    root, replica, owners, _ = _identity_shape(identity, root)
    matches = {}
    for node in ray.nodes():
        if node['Alive']:
            matches.setdefault(node['NodeManagerAddress'], []).append(node['NodeID'])
    _require(all(len(matches.get(owner['owner_ip'], [])) == 1 for owner in owners),
             'an original native node is missing or ambiguous in the current Ray cluster')
    inspector = ray.remote(num_cpus=0.1, num_gpus=0, max_calls=1, max_retries=0)(_inspect_completed_node)
    environment = {'CUDA_VISIBLE_DEVICES': '', 'PYTHONPATH': str(Path(__file__).parent) + os.pathsep + os.environ.get('PYTHONPATH', '')}
    source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    def submit(index, bundle=None):
        node_id = matches[owners[index]['owner_ip']][0]
        return inspector.options(scheduling_strategy=NodeAffinitySchedulingStrategy(node_id, soft=False),
                                 runtime_env={'env_vars': environment}).remote(
                                     identity, str(root), index, source_sha256, bundle)
    first = ray.get(submit(0), timeout=timeout)
    bundle = first['binding']
    if bundle is None:
        return None
    remaining = [submit(index, bundle) for index in range(1, len(owners))]
    results = [first] + (ray.get(remaining, timeout=timeout) if remaining else [])
    proofs = [result['proof'] for result in results]
    _require(sorted(item['node_index'] for item in proofs) == list(range(len(owners))) and
             all(item.get('passed') is True and item['canonical_shutdown_sha256'] == bundle['canonical_file']['sha256'] and
                 item['latest_sync_file'] == bundle['sync_file'] and item['latest_source_sha256'] == bundle['latest_source_sha256']
                 for item in proofs), 'not every original native node acknowledged the same completed shutdown')
    return dict(process_tree_terminated=True, nodes=bundle['canonical_shutdown']['nodes'],
                schema='q397-verified-completed-native-shutdown-v1', replica_rank=replica,
                verification_only=True, canonical_shutdown=bundle['canonical_file'], boot=bundle['boot_file'],
                latest_sync=bundle['sync_file'], latest_sync_step=bundle['latest_sync_step'],
                latest_source_sha256=bundle['latest_source_sha256'], native_boot=bundle['native_boot'],
                native_incarnation=bundle['native_incarnation'], model_version=bundle['model_version'],
                node_inspections=proofs, inspected_unix=time.time(), signals_sent=False)
