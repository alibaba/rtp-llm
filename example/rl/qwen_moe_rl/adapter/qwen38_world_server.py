"""One native EP72 owner tree with nine attention-DP request routes.

CPU prepare allocates no native process or CUDA buffer. Native bootstrap must
implement START_PAUSED and refuse RESUME/admission until a complete refit.
This controller cannot make an unqualified native build safe by itself.
"""
from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import struct
import subprocess
import threading
import time
import uuid

import psutil
import ray

from rtp_native_server import RTPServer, _audit_root, _deployment_root, _source, _write_json, _check_native_http_health
from qwen38_world_contract import (
    CONTROL_TOPOLOGY, World72Topology, canonical_uuid, require, fixed_layer_source_map,
    validate_native_manifest, validate_native_library_proof, validate_world_manifests,
)
from qwen38_refit_contract import validate_policy_probe_sources


def _probe_attempt_payload(values, hf, prompt, target, request_ids, context):
    """Record returned scores before validation; never grant admission here.

    Final protobuf payloads are preserved separately by NativeRpcJournal. These
    FP32 encodings describe the parsed score values, not network frames.
    """
    rows = []
    for dp, value in enumerate(values):
        row = dict(dp_rank=dp, generation_world_rank=dp * 8,
                   request_id=request_ids[dp], returned=not isinstance(value, BaseException))
        if isinstance(value, BaseException):
            row['error'] = repr(value)
        else:
            row['raw_log_probs'] = [x if math.isfinite(x) else None for x in value]
            row['nonfinite_indices'] = [i for i, x in enumerate(value) if not math.isfinite(x)]
            row['parsed_fp32_bits_hex'] = struct.pack(f'<{len(value)}f', *value).hex()
        rows.append(row)
    finite = (len(values) == 9 and not any(isinstance(row, BaseException) for row in values)
              and all(len(row) == len(target) and all(math.isfinite(x) for x in row) for row in values))
    return dict(schema='q38-native-HF-probe-attempt-v1',
        numeric_gates_not_yet_evaluated=True, acceptance_proven_here=False,
        generation_admission_granted_here=False,
        native_DP_tolerance=None if os.environ.get('Q38_LOGPROB_REPORT_ONLY') == '1' else .05,
        prompt_ids=prompt, response_ids=target, probes=rows,
        hf_raw_log_probs=hf['raw_log_probs'], hf_raw_fp32_bits_hex=hf.get('raw_fp32_bits_hex'),
        all_nine_scores_finite_and_complete=finite,
        observed_max_dp_error=(max(abs(a-b) for row in values for a, b in zip(values[0], row)) if finite else None),
        observed_max_HF_error=(max(abs(a-b) for row in values for a, b in zip(hf['raw_log_probs'], row)) if finite else None),
        **context)


def _check_native_port_available(host, port):
    # Match restartable TCP listeners: expired connections may retain a
    # TIME_WAIT tuple after every owned process has actually exited.
    # No SO_REUSEPORT: a live listener must remain an exclusive conflict.
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind((host, port))


def gpu_inventory():
    devices = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.total,memory.used',
                                       '--format=csv,noheader,nounits'], text=True)
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_gpu_memory',
                                         '--format=csv,noheader,nounits'], text=True)
    return dict(hostname=socket.gethostname(),
                devices=[dict(index=int(a), uuid=b.strip(), total_mib=int(c), used_mib=int(d))
                         for a, b, c, d in csv.reader(devices.splitlines())],
                processes=[dict(pid=int(a), uuid=b.strip(), used_mib=c.strip())
                           for a, b, c in csv.reader(processes.splitlines()) if a.strip().isdigit()])


def _sources(model_path):
    from qwen38_training_contract import load_metadata
    _config, _index, policy, _mtp = load_metadata(model_path)
    require(len(policy) == 1590, 'the pinned Qwen3.8 main policy must have 1590 sources')
    return set(policy)


def _status_rows(replies):
    fields = ('protocol_version', 'world_rank', 'world_size', 'tp_rank', 'tp_size',
              'dp_rank', 'dp_size', 'ep_rank', 'ep_size', 'phase', 'model_version',
              'global_step', 'update_id', 'received_weights', 'applied_weights',
              'quiescent', 'kv_cache_ready', 'kv_cache_bytes')
    return [{name: getattr(reply, name) for name in fields} for reply in replies]


async def _control(topology, *, bootstrap_deadline=None):
    _source()
    from rtp_llm.RL.replica_rpc_client import ReplicaRpcClient
    while True:
        control = ReplicaRpcClient(topology.control_addresses,
            timeout=float(os.environ.get('RLLM_RTP_CONTROL_TIMEOUT', '600')), topology=CONTROL_TOPOLOGY)
        try:
            await control.status()
            return control
        except BaseException as error:
            await control.close()
            cause = error.__cause__
            import grpc
            bootstrap_wait = (bootstrap_deadline is not None
                              and isinstance(cause, grpc.aio.AioRpcError)
                              and cause.code() == grpc.StatusCode.UNAVAILABLE
                              and cause.details() == 'RL bootstrap is waiting for complete-world quiescence')
            if not bootstrap_wait or time.monotonic() >= bootstrap_deadline:
                raise
            # A poisoned client is never reused. This is exclusively the
            # native constructor's documented read-only readiness response;
            # no state-changing operation is replayed.
            await asyncio.sleep(.25)


def _native_nccl(pid, created):
    expected_sha = os.environ.get('RLLM_Q38_NCCL_LIBRARY_SHA256', '')
    require(re.fullmatch('[0-9a-f]{64}', expected_sha) is not None,
            'current native NCCL SHA must be supplied by the qualified launcher')
    process = psutil.Process(pid)
    require(process.create_time() == created, 'native PID changed before NCCL inspection')
    paths = set()
    for line in (Path('/proc') / str(pid) / 'maps').read_text().splitlines():
        fields = line.split(None, 5)
        if len(fields) == 6 and Path(fields[-1].removesuffix(' (deleted)')).name.startswith('libnccl.so'):
            require(not fields[-1].endswith(' (deleted)'), 'native mapped NCCL was deleted')
            paths.add(str(Path(fields[-1]).resolve(strict=True)))
    require(len(paths) == 1, 'native must map exactly one NCCL library')
    library = Path(next(iter(paths)))
    configured = os.environ.get('RLLM_Q38_NCCL_LIBRARY_PATH')
    require(not configured or library == Path(configured).resolve(strict=True),
            'native NCCL mapped path differs from the current launcher binding')
    h = hashlib.sha256()
    with library.open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024**2), b''):
            h.update(block)
    require(h.hexdigest() == expected_sha and process.create_time() == created,
            'native NCCL bytes or process identity differ from the current launcher binding')
    return dict(pid=pid, create_time=created, library_path=str(library), library_sha256=h.hexdigest(),
                evidence='actual native PID mapped-file SHA; no compile-header version inference')


class WorldNodeOwner(RTPServer):
    """Reuse the existing identity-bound process-tree cleanup, never global pkill."""
    def __init__(self, model_path, topology, node_index):
        self.topology = World72Topology.from_dict(topology)
        self.node_index = int(node_index)
        self.node = self.topology.node_groups[self.node_index]
        self.main_audit_root = _audit_root().resolve()
        self.required_sources = _sources(model_path)
        self.boot = -1
        self.incarnation = None
        self._local_manifests = []
        self._process_lock = threading.RLock()
        self._stop_requested_for_boot = None
        super().__init__(0, model_path, ','.join(self.node['gpu_ids']), 8)
        self.owner_identity = dict(pid=os.getpid(), create_time=psutil.Process().create_time(),
            owner_hostname=socket.gethostname(), owner_ip=ray.util.get_node_ip_address(),
            node_index=self.node_index, ray_node_id=ray.get_runtime_context().get_node_id(),
            audit_dir=str(self.main_audit_root), root=str(_deployment_root()),
            ray_actor=dict(name=f'q38_native_world_owner_{self.node_index}',
                           namespace=ray.get_runtime_context().namespace))
        require(self.owner_identity['owner_ip'] == self.node['host'], 'Ray owner is on a different node')

    def _capture_process_tree(self):
        with self._process_lock:
            return super()._capture_process_tree()

    def _terminate(self):
        with self._process_lock:
            return super()._terminate()

    async def identity(self):
        inventory = gpu_inventory()
        selectors = {str(row['index']): canonical_uuid(row['uuid']) for row in inventory['devices']}
        available = {canonical_uuid(row['uuid']) for row in inventory['devices']}
        selected = []
        for value in self.node['gpu_ids']:
            physical = selectors.get(value)
            if physical is None:
                physical = canonical_uuid(value)
            require(physical in available, 'Ray GPU selector is outside the current node allocation')
            selected.append(physical)
        require(len(selected) == len(set(selected)) == 4, 'native owner requires four different physical GPUs')
        return dict(self.owner_identity, allocated_gpu_uuids=selected, inventory=inventory)

    async def bind_world(self, topology):
        bound = World72Topology.from_dict(topology)
        require(bound.rank_hosts == self.topology.rank_hosts and bound.gpu_ids == self.topology.gpu_ids
                and bound.start_port == self.topology.start_port and len(bound.gpu_uuids) == 72,
                'world binding changed the CPU-prepared allocation')
        own = await self.identity()
        require(own['allocated_gpu_uuids'] == bound.node_groups[self.node_index]['gpu_uuids'],
                'world binding UUIDs differ from current owner inventory')
        self.topology = bound
        self.node = bound.node_groups[self.node_index]
        _write_json(self.main_audit_root / 'rtp_topology_0.json', bound.to_dict())
        return dict(node_index=self.node_index, bound=True, allocated_gpu_uuids=own['allocated_gpu_uuids'])

    async def start(self, boot, incarnation):
        require(not self.poisoned, 'native owner was permanently stopped')
        require(len(self.topology.gpu_uuids) == 72, 'native start requires the complete physical world binding')
        library_sha256 = os.environ.get('RTP_RL_NATIVE_LIBRARY_SHA256', '')
        require(len(library_sha256) == 64 and all(c in '0123456789abcdef' for c in library_sha256),
                'pin the fresh native library SHA256 before any native process allocation')
        require(self.process is None or self.process.poll() is not None, 'a native incarnation is still alive')
        if self.process is not None and self._cleanup_receipt is None:
            self._terminate()
        self.boot, self.incarnation = int(boot), str(incarnation)
        self._stop_requested_for_boot = None
        self._cleanup_receipt = None
        self._owned_processes = {}
        self._local_manifests = []
        self.process = self.process_identity = self.log = None
        own_root = self.main_audit_root / f'world-boots/boot-{self.boot:03d}/node-{self.node_index:02d}'
        own_root.mkdir(parents=True, exist_ok=False)
        physical_map_path=own_root/'fixed-phy2log.json'
        _write_json(physical_map_path, fixed_layer_source_map())
        os.environ['RLLM_RTP_ACCEPTANCE_DIR'] = str(own_root)
        for port in range(self.topology.start_port - 11, self.topology.start_port + 4 * self.topology.rank_port_stride + 12):
            _check_native_port_available(self.node['host'], port)
        environment = dict(os.environ)
        for key in ('RANK', 'WORLD_RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE',
                    'MASTER_ADDR', 'MASTER_PORT', 'DISTRIBUTE_CONFIG_FILE'):
            environment.pop(key, None)
        environment.update(self.topology.node_environment(self.node_index, incarnation))
        environment.update(RTP_MODEL_PATH=self.model_path,
                           RTP_PHY2LOG_PATH=str(physical_map_path),
                           RTP_WEIGHT_MANIFEST_DIR=str(own_root / 'rtp_weight_manifest_0'))
        self.log = (own_root / 'native-world.log').open('xb')
        self.process = subprocess.Popen(['/bin/bash', os.environ['RTP_SERVER_LAUNCHER']],
                                        env=environment, stdout=self.log, stderr=subprocess.STDOUT,
                                        start_new_session=True)
        leader = psutil.Process(self.process.pid)
        self.process_identity = dict(pid=leader.pid, create_time=leader.create_time(), pgid=os.getpgid(leader.pid),
            replica_rank=0, boot_id=self.boot, incarnation=incarnation, node_index=self.node_index,
            owner_hostname=socket.gethostname(), owner_ip=self.node['host'],
            owned_world_ranks=self.node['ranks'], tp_size=8, world_size=72,
            root=str(_deployment_root()), audit_dir=str(own_root), started=time.time())
        self._owned_processes[leader.pid] = leader.create_time()
        _write_json(own_root / 'rtp_process_0.json', self.process_identity)
        deadline = time.monotonic() + int(os.environ.get('RLLM_RTP_STARTUP_TIMEOUT', '5400'))
        try:
            while time.monotonic() < deadline:
                require(not self.poisoned and self._stop_requested_for_boot != self.boot,
                        'native owner startup was cancelled by its controller')
                self._capture_process_tree()
                require(self.process.poll() is None, f'native node exited {self.process.returncode}')
                paths = [own_root / f'rtp_weight_manifest_0/rank-{rank:05d}.json' for rank in self.node['ranks']]
                library_paths = [own_root / f'rtp_weight_manifest_0/native-library-rank-{rank:05d}.json'
                                 for rank in self.node['ranks']]
                if not all(path.exists() for path in paths + library_paths):
                    await asyncio.sleep(.5)
                    continue
                manifests = []
                for rank, path, library_path in zip(self.node['ranks'], paths, library_paths):
                    record = json.loads(path.read_text())
                    validate_native_manifest(record, self.topology, rank, self.required_sources, incarnation=incarnation)
                    process = psutil.Process(record['pid'])
                    require(self._owned_processes.get(process.pid) == process.create_time() and self._running(process),
                            'native world rank is not a live identity-bound child')
                    library = validate_native_library_proof(json.loads(library_path.read_text()), record, library_sha256)
                    record.update(owner_hostname=socket.gethostname(), owner_ip=self.node['host'],
                                  owner_node_index=self.node_index, boot_id=self.boot, incarnation=incarnation,
                                  metadata_bootstrap=True, bootstrap_native_library=library)
                    manifests.append(record)
                try:
                    health = await asyncio.gather(*[
                        asyncio.to_thread(_check_native_http_health, self.node['host'],
                            self.topology.start_port + local * self.topology.rank_port_stride + 5, rank, 2)
                        for local, rank in enumerate(self.node['ranks'])])
                except (OSError, TimeoutError, ConnectionError):
                    await asyncio.sleep(.5)
                    continue
                inventory = gpu_inventory()
                native_pids = {row['pid'] for row in manifests}
                require({row['pid'] for row in inventory['processes'] if row['pid'] in native_pids} == native_pids,
                        'every local native rank must own an actual GPU context')
                self._local_manifests = manifests
                proof = dict(node_index=self.node_index, native_boot=self.boot,
                    native_incarnation=incarnation, rank_manifests=manifests, process_identity=self.process_identity,
                    hostname=socket.gethostname(), node_ip=self.node['host'], http_health=health,
                    inventory=inventory, fixed_physical_expert_map_path=str(physical_map_path),
                    fixed_physical_expert_map_sha256=hashlib.sha256(physical_map_path.read_bytes()).hexdigest(),
                    logical_experts=512, physical_expert_slots=576, never_routed_padding_slots=64,
                    ready_unix=time.time())
                _write_json(own_root / 'local-startup.json', proof)
                return proof
            raise TimeoutError('native local HTTP/manifest startup did not complete')
        except BaseException:
            await asyncio.to_thread(self._terminate)
            raise

    async def publish(self, manifests, topology, boot_proof):
        require(boot_proof['native_boot'] == self.boot and boot_proof['native_incarnation'] == self.incarnation,
                'world publication belongs to another native boot')
        ordered = validate_world_manifests(manifests, self.topology, self.required_sources,
                                          boot=self.boot, incarnation=self.incarnation)
        for row in ordered:
            _write_json(self.main_audit_root / f'rtp_weight_manifest_0/rank-{row["world_rank"]:05d}.json', row)
        _write_json(self.main_audit_root / 'rtp_topology_0.json', topology)
        _write_json(self.main_audit_root / f'rtp_boot_0_step{boot_proof["target_global_step"]}.json', boot_proof)
        nccl = [_native_nccl(row['pid'], self._owned_processes[row['pid']]) for row in self._local_manifests]
        _write_json(self.main_audit_root / f'world-boots/boot-{self.boot:03d}/node-{self.node_index:02d}/actual-nccl.json',
                    dict(node_index=self.node_index, ranks=self.node['ranks'], actual_mappings=nccl))
        return dict(node_index=self.node_index, published=True, manifest_count=72, actual_nccl_mappings=nccl)

    async def stop_for_training(self):
        self._stop_requested_for_boot = self.boot
        if self.process is not None:
            await asyncio.to_thread(self._terminate)
        if self.log is not None:
            self.log.close()
        result = dict(self._cleanup_receipt or dict(process_tree_terminated=True, owned_pids=[]))
        result.update(node_index=self.node_index, owner_ip=self.node['host'], owner_hostname=socket.gethostname(),
                      native_boot=self.boot, native_incarnation=self.incarnation,
                      checked_native_ranks=list(self.node['ranks']),
                      manifested_native_ranks=[row['world_rank'] for row in self._local_manifests])
        inventory = gpu_inventory()
        available = {canonical_uuid(row['uuid']) for row in inventory['devices']}
        require(set(self.node['gpu_uuids']).issubset(available) and len(self.node['gpu_uuids']) == 4,
                'cleanup cannot verify the original four physical rank slots')
        # NVML may briefly retain a context after the process exits. Any row
        # with an owned numeric PID still present is conservatively a blocker;
        # a reused PID is never signalled by this independent observation.
        remaining = [row for row in inventory['processes'] if row['pid'] in set(result['owned_pids'])]
        require(not remaining, 'owned native GPU contexts remain after actual process-tree cleanup')
        result.update(owned_gpu_processes_after=remaining, inventory_after=inventory,
                      remaining_owned_gpu_contexts=remaining, finished_at_unix=time.time())
        return result

    async def inventory(self):
        return gpu_inventory()

    async def stop_if_owned(self, expected_pid, expected_create_time, expected_audit_dir):
        identity = self.owner_identity
        require((expected_pid, expected_create_time, str(Path(expected_audit_dir).resolve()))
                == (identity['pid'], identity['create_time'], identity['audit_dir']),
                'shutdown does not own this world node actor')
        self.poisoned = True
        return await self.stop_for_training()


class World72RTPServer(RTPServer):
    def __init__(self, model_path, topology, owners, node_ids):
        self.topology = World72Topology.from_dict(topology)
        self.owners = list(owners)
        self.node_ids = list(node_ids)
        self.main_audit_root = _audit_root().resolve()
        self.required_sources = _sources(model_path)
        self.boot = -1
        self.incarnation = None
        self._started = self._owners_active = self._generation_ready = False
        self._incarnation_failed = False
        self._committed_version = self._committed_step = None
        self._owner_identities = []
        self._last_suspend = None
        self._lifecycle_lock = asyncio.Lock()
        self._admission = asyncio.Condition()
        self._inflight = 0
        self._next_dp = 0
        self.clients = []
        super().__init__(0, model_path, '', 8)
        self.controller_identity = dict(pid=os.getpid(), create_time=psutil.Process().create_time(),
            replica_rank=0, logical_model_replicas=1, world_size=72, tp_size=8, dp_size=9, ep_size=72,
            owner_hostname=socket.gethostname(), audit_dir=str(self.main_audit_root), root=str(_deployment_root()),
            ray_actor=dict(name='rtp_server_0_0', namespace=ray.get_runtime_context().namespace),
            node_owner_count=18, node_owners=[], topology=self.topology.to_dict())

    def _terminate(self):
        # Explicit async shutdown awaits all 18 owners before Ray shutdown.
        return None

    async def prepare(self):
        async with self._lifecycle_lock:
            if self._owner_identities:
                return self.controller_identity
            require(len(self.owners) == len(self.node_ids) == len(set(self.node_ids)) == 18,
                    'the controller must own 18 distinct Ray nodes')
            identities = await asyncio.gather(*[owner.identity.remote() for owner in self.owners])
            for index, row in enumerate(identities):
                require(row['node_index'] == index and row['owner_ip'] == self.topology.node_groups[index]['host']
                        and row['ray_node_id'] == self.node_ids[index] and row['audit_dir'] == str(self.main_audit_root),
                        'native owner identity differs from the CPU-prepared allocation')
            physical = [value for row in identities for value in row['allocated_gpu_uuids']]
            if self.topology.gpu_uuids:
                require([canonical_uuid(x) for x in self.topology.gpu_uuids] == physical,
                        'native owner UUIDs changed from the recorded allocation')
            self.topology = self.topology.with_gpu_uuids(physical)
            bound = await asyncio.gather(*[owner.bind_world.remote(self.topology.to_dict()) for owner in self.owners])
            require(len(bound) == 18 and all(row['bound'] for row in bound), 'not all owners accepted the physical world binding')
            self._owner_identities = identities
            self.controller_identity.update(node_owners=identities, topology=self.topology.to_dict())
            _write_json(self.main_audit_root / 'rtp_topology_0.json', self.topology.to_dict())
            _write_json(self.main_audit_root / 'rtp_cluster_0.json', self.controller_identity)
            return self.controller_identity

    async def launch_server(self):
        await self.prepare()
        return self.topology.generation_addresses[0]

    async def owned_identity(self):
        # Emergency cleanup discovery must not wait behind native startup's
        # lifecycle lock. CPU identity is immutable after prepare completed.
        return self.controller_identity

    async def _close_admission(self, reason):
        async with self._admission:
            self._generation_ready = False
            for client in self.clients:
                client.close_admission(reason)
            await asyncio.wait_for(self._admission.wait_for(lambda: self._inflight == 0),
                                   timeout=float(os.environ.get('RLLM_RTP_DRAIN_TIMEOUT', '600')))

    async def _close_clients(self):
        if self.clients:
            outcomes = await asyncio.gather(*(client.close() for client in self.clients), return_exceptions=True)
            self.clients = []
            self.client = None
            require(not any(isinstance(row, BaseException) for row in outcomes), 'a native DP token client failed to close')

    async def _stop_owners(self, *, permanent=False):
        if permanent:
            require(len(self._owner_identities) == len(self.owners) == 18,
                    'failed startup cleanup requires all18 prepared owner identities')
            calls = [owner.stop_if_owned.remote(row['pid'], row['create_time'], row['audit_dir'])
                     for owner, row in zip(self.owners, self._owner_identities)]
        else:
            calls = [owner.stop_for_training.remote() for owner in self.owners]
        return await asyncio.gather(*calls, return_exceptions=True)

    async def ensure_native_running(self, global_step, previous_committed_version):
        if not self._owner_identities:
            await self.prepare()
        async with self._lifecycle_lock:
            require(not self.poisoned, 'world controller was permanently shut down')
            require(self._committed_version == previous_committed_version, 'refit does not reference the last committed policy')
            await self._close_admission('policy_refit')
            if self._started:
                require(not self._incarnation_failed, 'the failed native incarnation cannot be reused')
                return dict(native_boot=self.boot, native_incarnation=self.incarnation, restarted=False)
            self.boot += 1
            self.incarnation = uuid.uuid4().hex
            self._incarnation_failed = False
            self._owners_active = True
            try:
                # First failed owner must stop the whole boot immediately. The
                # cleanup below permanently poisons identity-bound owners, so
                # queued start RPCs cannot allocate after the cleanup receipt.
                results = await asyncio.gather(*[owner.start.remote(self.boot, self.incarnation) for owner in self.owners])
                require(not any(isinstance(row, BaseException) for row in results), 'native world local startup failed')
                manifests = validate_world_manifests([row for proof in results for row in proof['rank_manifests']],
                    self.topology, self.required_sources, boot=self.boot, incarnation=self.incarnation)
                control = await _control(self.topology, bootstrap_deadline=time.monotonic() + 60)
                try:
                    statuses = await control.status()
                finally:
                    await control.close()
                require(all(row.phase == 'paused' and row.quiescent and row.model_version.startswith(self.incarnation + ':')
                            for row in statuses), 'native bootstrap must actually be paused before full policy refit')
                proof = dict(replica_rank=0, logical_model_replicas=1, native_boot=self.boot,
                    native_incarnation=self.incarnation, target_global_step=int(global_step),
                    previous_committed_version=previous_committed_version, nodes=results,
                    manifest_count=72, initial_statuses=_status_rows(statuses),
                    startup_does_not_establish_complete_policy=True, ready_unix=time.time())
                publications = await asyncio.gather(*[owner.publish.remote(manifests, self.topology.to_dict(), proof) for owner in self.owners])
                require(len(publications) == 18 and all(row['published'] for row in publications), 'not all native owners published world manifests')
                proof['native_nccl_publications'] = publications
                _write_json(self.main_audit_root / f'rtp_boot_0_step{global_step}.json', proof)
                self._create_clients()
                self._started = True
                return dict(native_boot=self.boot, native_incarnation=self.incarnation, restarted=self.boot > 0)
            except BaseException:
                self.poisoned = True
                try:
                    await self._close_clients()
                finally:
                    receipts = await self._stop_owners(permanent=True)
                    self._owners_active = any(isinstance(row, BaseException) for row in receipts)
                    self._started = self._generation_ready = False
                    self._incarnation_failed = True
                    self._last_suspend = dict(world_size=72, native_boot=self.boot, native_incarnation=self.incarnation,
                        full_world_was_ready=False, process_tree_terminated=not self._owners_active,
                        all_owned_native_processes_exited=not self._owners_active,
                        remaining_owned_gpu_contexts=[], startup_failed=True, training_admission_safe=False,
                        all_world_pause_acknowledged=False, errors=['native startup failed'],
                        checked_native_ranks=sorted(rank for row in receipts if not isinstance(row, BaseException)
                                                    for rank in row['checked_native_ranks']),
                        nodes=[row for row in receipts if not isinstance(row, BaseException)],
                        cleanup_errors=[repr(row) for row in receipts if isinstance(row, BaseException)])
                raise

    def _create_clients(self):
        _source()
        from transformers import AutoConfig, AutoTokenizer
        from rtp_llm.RL.token_rpc_client import TokenProtocolError, TokenRpcClient
        from rtp_request_journal import JournaledTokenRpcMixin, NativeRpcJournal
        config = AutoConfig.from_pretrained(self.model_path, local_files_only=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        generation = json.loads((Path(self.model_path) / 'generation_config.json').read_text())
        eos = generation.get('eos_token_id', self.tokenizer.eos_token_id)
        eos = eos if isinstance(eos, list) else [eos]
        require(eos and all(type(value) is int for value in eos), 'native tokenizer EOS IDs are missing')

        class StrictClient(JournaledTokenRpcMixin, TokenRpcClient):
            async def _request(self, request_id, ids, config):
                response = await super()._request(request_id, ids, config)
                if self.expected_step is None or response.global_step != self.expected_step or response.model_version != self.expected_version:
                    raise TokenProtocolError('native DP response version or step differs from the fully committed policy')
                if config.return_softmax_probs:
                    infos = response.flatten_output.aux_info
                    if len(infos) != 1 or 'raw_logprobs' not in infos[0].DESCRIPTOR.fields_by_name or not infos[0].HasField('raw_logprobs'):
                        raise TokenProtocolError('native FP32 raw logprobs are required; probability fallback is forbidden')
                return response

        for dp_rank, address in enumerate(self.topology.generation_addresses):
            context = lambda dp=dp_rank: dict(native_boot=self.boot, native_incarnation=self.incarnation,
                                              logical_model_replicas=1, dp_rank=dp, generation_world_rank=dp * 8)
            journal = NativeRpcJournal(self.main_audit_root, dp_rank, context)
            client = StrictClient(address, vocab_size=getattr(config, 'text_config', config).vocab_size,
                eos_token_ids=eos, timeout=float(os.environ.get('RLLM_RTP_TOKEN_TIMEOUT', '600')), request_journal=journal)
            client.expected_step = client.expected_version = None
            client.close_admission('awaiting_complete_world_refit')
            self.clients.append(client)
        self.client = None  # Never silently route the entire world through DP0.

    async def wake_and_probe(self, expected_step, expected_version):
        async with self._lifecycle_lock:
            try:
                return await self._wake_and_probe_locked(expected_step, expected_version)
            except BaseException:
                self._incarnation_failed = True
                self._generation_ready = False
                for client in self.clients:
                    client.close_admission('world_control_or_probe_failed')
                raise

    async def verify_final_policy(self, expected_step=2):
        require(type(expected_step) is int and expected_step == 2
                and self._committed_step == expected_step and self._committed_version,
                'the final gate requires the actually committed policy after exactly two optimizer updates')
        record = await self.wake_and_probe(expected_step, self._committed_version)
        record = dict(record, final_native_GPU_reprobe=True, final_expected_optimizer_updates=2,
                      controller_identity=self.controller_identity)
        _write_json(self.main_audit_root / 'rtp_final_policy2_actual_reprobe.json', record)
        return record

    async def _wake_and_probe_locked(self, expected_step, expected_version):
        require(self._started and not self.poisoned and not self._incarnation_failed
                and expected_step is not None and expected_version,
                'a fully refitted native world is required')
        await self._close_admission('post_refit_probe')
        sync_path = self.main_audit_root / f'rtp_sync_rank0_step{expected_step}.json'
        sync = json.loads(sync_path.read_text())
        hf_path = Path(os.environ['Q38_TRAINING_AUDIT_DIR']) / f'hf-probe-step{expected_step}-rank0.json'
        hf = validate_policy_probe_sources(sync, json.loads(hf_path.read_text()), step=expected_step,
            version=expected_version, boot=self.boot, incarnation=self.incarnation)
        control = await _control(self.topology)
        try:
            statuses = await control.status()
            require(all(row.model_version == expected_version and row.global_step == expected_step and not row.update_id
                        and row.received_weights == 1590 and row.applied_weights > 0 for row in statuses),
                    'all 72 ranks must ACK the complete main-policy version before resume')
            if statuses[0].phase == 'paused':
                if not statuses[0].kv_cache_ready:
                    await control.restore_cache()
                statuses = await control.resume()
            require(all(row.phase == 'running' for row in statuses), 'the world did not resume for the probes')
        finally:
            await control.close()
        prompt = self.tokenizer.encode('The value of 37 * 43 is', add_special_tokens=False)
        target = self.tokenizer.encode(' 1591.', add_special_tokens=False)
        require(hf['prompt_ids'] == prompt and hf['response_ids'] == target,
                'native and actual HF probes must score the identical prompt and target IDs')
        for client in self.clients:
            client.expected_step, client.expected_version = expected_step, expected_version
            client.open_admission()  # Internal probes only; controller admission is still closed.
        try:
            hf_report_only = os.environ.get('Q38_HF_RTP_REPORT_ONLY') == '1'
            probe_request_ids = [uuid.uuid4().int & ((1 << 63) - 1) for _ in self.clients]
            values = await asyncio.gather(*[client.score_tokens(request_id, prompt, target,
                expected_version=expected_version) for client, request_id in zip(self.clients, probe_request_ids)],
                return_exceptions=True)
            # Persist failures too. The accepted probe filename below remains
            # reserved for scores that pass every original numerical gate.
            attempt_time_ns = time.time_ns()
            attempt = _probe_attempt_payload(values, hf, prompt, target, probe_request_ids, dict(
                logical_model_replicas=1, world_size=72, tp_size=8, dp_size=9, ep_size=72,
                native_boot=self.boot, native_incarnation=self.incarnation, global_step=expected_step,
                model_version=expected_version, recorded_unix_ns=attempt_time_ns,
                HF_comparison_mode='report_only' if hf_report_only else 'absolute_error_gate',
                HF_error_used_for_acceptance=not hf_report_only,
                HF_reference_tolerance=None if hf_report_only else .05,
                actual_HF_reference_path=str(hf_path),
                actual_HF_reference_sha256=hashlib.sha256(hf_path.read_bytes()).hexdigest(),
                source_transaction_path=str(sync_path),
                source_transaction_sha256=hashlib.sha256(sync_path.read_bytes()).hexdigest(),
                full_source_sha256=sync['full_source_sha256']))
            _write_json(self.main_audit_root / f'rtp_probe_attempt_step{expected_step}_{attempt_time_ns}.json', attempt)
            require(len(values) == 9 and not any(isinstance(row, BaseException) for row in values)
                    and all(len(row) == len(target) and all(math.isfinite(x) for x in row) for row in values),
                    'all nine actual DP probes must return finite target logprobs')
            max_dp_error = max(abs(left - right) for row in values for left, right in zip(values[0], row))
            dp_report_only = os.environ.get('Q38_LOGPROB_REPORT_ONLY') == '1'
            if not dp_report_only:
                require(max_dp_error <= .05, 'nine DP copies disagree on the committed policy scores')
            hf_errors = [[abs(left - right) for left, right in zip(hf['raw_log_probs'], row)] for row in values]
            max_hf_error = max(error for row in hf_errors for error in row)
            mean_hf_absolute_error = sum(error for row in hf_errors for error in row) / sum(len(row) for row in hf_errors)
            if not hf_report_only:
                require(max_hf_error <= .05, 'actual BF16 HF and nine native DP probes differ beyond 0.05')
            record = dict(replica_rank=0, logical_model_replicas=1, world_size=72, tp_size=8, dp_size=9, ep_size=72,
                native_boot=self.boot, native_incarnation=self.incarnation, global_step=expected_step,
                model_version=expected_version, prompt_ids=prompt, response_ids=target,
                full_source_sha256=sync['full_source_sha256'], statuses=_status_rows(statuses),
                probes=[dict(dp_rank=dp, generation_world_rank=dp * 8, address=self.topology.generation_addresses[dp],
                             raw_log_probs=row, raw_fp32_bits_hex=struct.pack(f'<{len(row)}f', *row).hex(),
                             absolute_HF_errors=hf_errors[dp],
                             mean_HF_absolute_error=sum(hf_errors[dp]) / len(hf_errors[dp]))
                        for dp, row in enumerate(values)], max_dp_error=max_dp_error,
                DP_comparison_mode='report_only' if dp_report_only else 'absolute_error_gate',
                DP_error_used_for_acceptance=not dp_report_only,
                actual_HF_reference_path=str(hf_path), actual_HF_reference_sha256=hashlib.sha256(hf_path.read_bytes()).hexdigest(),
                hf_raw_log_probs=hf['raw_log_probs'], hf_raw_fp32_bits_hex=hf.get('raw_fp32_bits_hex'),
                max_HF_error=max_hf_error, mean_HF_absolute_error=mean_hf_absolute_error,
                HF_comparison_mode='report_only' if hf_report_only else 'absolute_error_gate',
                HF_error_used_for_acceptance=not hf_report_only,
                HF_reference_tolerance=None if hf_report_only else .05,
                source_transaction_path=str(sync_path), source_transaction_sha256=hashlib.sha256(sync_path.read_bytes()).hexdigest(),
                finished_unix=time.time())
            _write_json(self.main_audit_root / f'rtp_probe_rank0_step{expected_step}.json', record)
        except BaseException:
            self._incarnation_failed = True
            for client in self.clients:
                client.close_admission('world_probe_failed')
            raise
        self._committed_version, self._committed_step = expected_version, int(expected_step)
        async with self._admission:
            self._generation_ready = True
        return record

    async def _route(self):
        async with self._admission:
            require(self._started and self._generation_ready and not self.poisoned, 'world generation admission is closed')
            dp = self._next_dp % 9
            self._next_dp += 1
            self._inflight += 1
            return dp, self.clients[dp], self._committed_step, self._committed_version

    async def _route_done(self):
        async with self._admission:
            self._inflight -= 1
            self._admission.notify_all()

    async def generate(self, request_id, *, prompt_ids, sampling_params, **kwargs):
        _source()
        from rtp_llm.RL.token_rpc_client import TokenSamplingParams
        from verl.workers.rollout.replica import TokenOutput
        require(not any(kwargs.get(key) for key in ('image_data', 'video_data', 'audio_data')), 'this policy is text-only')
        params = dict(sampling_params)
        temperature, top_p, top_k = float(params.pop('temperature', 1)), float(params.pop('top_p', 1)), int(params.pop('top_k', 0))
        require(temperature in (0., 1.) and top_p == 1. and top_k in (0, -1), 'strict policy sampling configuration differs')
        budget = int(params.pop('max_tokens', params.pop('max_new_tokens', 2048)))
        require(params.pop('logprobs', 1) in (1, True), 'training requires selected-token logprobs')
        seed = params.pop('seed', params.pop('random_seed', None))
        require(not params, f'unsupported RTP sampling fields {sorted(params)}')
        dp, client, step, version = await self._route()
        try:
            native_id = uuid.uuid4().int & ((1 << 63) - 1)
            result = await client.generate(native_id, prompt_ids,
                TokenSamplingParams(max_new_tokens=budget, temperature=temperature, top_p=1., top_k=0,
                    seed=seed, logprobs=True, logprobs_mode='raw_logprobs', stop_token_ids=tuple(sorted(client.eos_token_ids))),
                expected_version=version)
            require(result.model_version == version and result.global_step == step and result.log_probs,
                    'generated tokens do not bind to the committed native policy')
            receipt = dict(engine='rtp-llm', replica_rank=0, logical_model_replicas=1, request_id=str(request_id),
                native_request_id=native_id, dp_rank=dp, generation_world_rank=dp * 8,
                model_version=version, global_step=step, native_incarnation=self.incarnation, native_boot=self.boot,
                prompt_tokens=len(prompt_ids), response_tokens=len(result.token_ids), logprobs=len(result.log_probs),
                prompt_sha256=hashlib.sha256(json.dumps(prompt_ids).encode()).hexdigest(),
                response_sha256=hashlib.sha256(json.dumps(result.token_ids).encode()).hexdigest(),
                raw_logprobs_fp32_sha256=hashlib.sha256(struct.pack(f'<{len(result.log_probs)}f', *result.log_probs)).hexdigest(),
                logprobs_mode=result.logprobs_mode, finished_at_unix=time.time())
            with (self.main_audit_root / 'rtp_generations_0.jsonl').open('a') as stream:
                stream.write(json.dumps(receipt) + '\n')
            return TokenOutput(token_ids=result.token_ids, log_probs=result.log_probs,
                               stop_reason=result.finish_reason, extra_fields=receipt)
        finally:
            await self._route_done()

    async def score_tokens(self, prompt_ids, response_ids, expected_version=None):
        dp, client, step, version = await self._route()
        try:
            require(expected_version is None or expected_version == version, 'score request references a stale policy')
            values = await client.score_tokens(uuid.uuid4().int & ((1 << 63) - 1), prompt_ids, response_ids, expected_version=version)
            return dict(log_probs=values, model_version=version, global_step=step, dp_rank=dp)
        finally:
            await self._route_done()

    async def suspend_for_training(self):
        async with self._lifecycle_lock:
            if not self._owners_active:
                require(self._last_suspend is not None, 'no completed native incarnation has a cleanup proof')
                return self._last_suspend
            errors = []
            statuses = []
            try:
                await self._close_admission('training_suspend')
                control = await _control(self.topology)
                try:
                    statuses = await control.pause()
                finally:
                    await control.close()
            except BaseException as error:
                errors.append(repr(error))
            try:
                await self._close_clients()
            except BaseException as error:
                errors.append(repr(error))
            receipts = await self._stop_owners()
            failures = [repr(row) for row in receipts if isinstance(row, BaseException)]
            if failures:
                errors.extend(failures)
            clean = [row for row in receipts if not isinstance(row, BaseException)]
            checked = sorted(rank for row in clean for rank in row['checked_native_ranks'])
            all_exited = (len(clean) == 18 and checked == list(range(72))
                          and all(row['process_tree_terminated'] and not row['remaining_owned_gpu_contexts'] for row in clean))
            require(all_exited, 'not all 18 native owner trees and 72 GPU contexts have a verified cleanup')
            inventories = await asyncio.gather(*[owner.inventory.remote() for owner in self.owners])
            remaining = [dict(node_index=index, **row) for index, (receipt, inventory) in enumerate(zip(clean, inventories))
                         for row in inventory['processes'] if row['pid'] in set(receipt['owned_pids'])]
            require(not remaining, 'a delayed owned native context remains after all-owner cleanup')
            complete_policy_verified=(self._committed_step is not None and bool(self._committed_version)
                and not self._incarnation_failed and not self.poisoned and len(statuses)==72
                and all(row.phase=='paused' and row.quiescent and not row.update_id
                    and row.received_weights==1590 and row.applied_weights>0
                    and row.global_step==self._committed_step and row.model_version==self._committed_version
                    and row.model_version.startswith(str(self.incarnation)+':') for row in statuses))
            result = dict(world_size=72, logical_model_replicas=1, tp_size=8, dp_size=9, ep_size=72,
                all_owned_native_processes_exited=True, remaining_owned_gpu_contexts=remaining,
                checked_native_ranks=checked, process_tree_terminated=True, nodes=clean,
                all_world_pause_acknowledged=(len(statuses) == 72), pause_statuses=_status_rows(statuses),
                complete_policy_verified_before_training_handoff=bool(complete_policy_verified),
                training_admission_safe=(not errors and complete_policy_verified),
                native_boot=self.boot, native_incarnation=self.incarnation, errors=errors, finished_unix=time.time())
            self._started = self._owners_active = self._generation_ready = False
            self._last_suspend = result
            _write_json(self.main_audit_root / f'rtp_suspend_0_boot{self.boot}.json', result)
            return result

    async def node_inventories(self):
        return await asyncio.gather(*[owner.inventory.remote() for owner in self.owners])

    async def sleep(self):
        if self.boot==-1 and not self._owners_active and not self._started:
            return dict(world_size=72,logical_model_replicas=1,tp_size=8,dp_size=9,ep_size=72,
                native_never_started=True,native_boot=-1,native_incarnation=None,
                all_owned_native_processes_exited=True,process_tree_terminated=True,
                remaining_owned_gpu_contexts=[],checked_native_ranks=[],nodes=[],
                actor_cleanup_handoff_not_required='initial CPU-only prepare; no native allocation or canonical exporter',
                training_admission_safe=False,all_world_pause_acknowledged=False)
        return await self.suspend_for_training()

    async def shutdown(self):
        if not self._owners_active:
            self.poisoned = True
            return self._last_suspend or dict(world_size=72, native_never_started=True, process_tree_terminated=True,
                                              all_owned_native_processes_exited=True, remaining_owned_gpu_contexts=[], nodes=[])
        try:
            return await self.suspend_for_training()
        finally:
            self.poisoned = True

    async def shutdown_if_owned(self, expected_pid, expected_create_time, expected_audit_dir):
        identity = self.controller_identity
        require((expected_pid, expected_create_time, str(Path(expected_audit_dir).resolve()))
                == (identity['pid'], identity['create_time'], identity['audit_dir']), 'shutdown does not own this world controller')
        # Emergency shutdown must not queue behind an hour-long native startup
        # holding the lifecycle lock. Permanent owner stops also prevent an
        # already queued start from allocating GPUs after the stop receipt.
        self.poisoned = True
        self._generation_ready = False
        for client in self.clients:
            client.close_admission('verified_emergency_shutdown')
        if self._owner_identities:
            await asyncio.gather(*[owner.stop_if_owned.remote(row['pid'], row['create_time'], row['audit_dir'])
                                   for owner, row in zip(self.owners, self._owner_identities)], return_exceptions=True)
        return await self.shutdown()


def launch_world72_server(model_path, topology, node_ids):
    topology = World72Topology.from_dict(topology)
    require(ray.is_initialized() and len(node_ids) == len(set(node_ids)) == 18,
            'one distinct live Ray node per native owner is required')
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
    runtime = dict(env_vars={'RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES': '1',
                            'Q38_LOGPROB_REPORT_ONLY': '1' if os.environ.get('Q38_LOGPROB_REPORT_ONLY') == '1' else '0'})
    owner_class = ray.remote(num_cpus=1, num_gpus=0)(WorldNodeOwner)
    owners = [owner_class.options(name=f'q38_native_world_owner_{index}', max_concurrency=16,
                  scheduling_strategy=NodeAffinitySchedulingStrategy(node_id, soft=False), runtime_env=runtime)
              .remote(model_path, topology.to_dict(), index) for index, node_id in enumerate(node_ids)]
    controller_class = ray.remote(num_cpus=1, num_gpus=0)(World72RTPServer)
    controller = (controller_class.options(name='rtp_server_0_0', max_concurrency=256,
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_ids[0], soft=False), runtime_env=runtime)
        .remote(model_path, topology.to_dict(), owners, node_ids))
    return controller, owners
