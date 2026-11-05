"""Own one native RTP TP world across nodes, preserving node-local CUDA IPC.

Native owners run on their assigned Ray nodes. The stable controller preserves
policy probes and pauses the resident native world around FSDP training. A fresh
checkpoint incarnation never serves RL generations until the full policy has
been committed and its native probe succeeds.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import uuid

import psutil
import ray

from rtp_native_server import (
    RTPServer, _audit_root, _check_native_http_health, _deployment_root,
    _new_control, _source, _write_json,
)
from rtp_topology import ReplicaTopology
from rtp_request_journal import NativeRpcJournal, JournaledTokenRpcMixin
from q397_nccl_scope import scope_config, inspect_pid_nccl


class PinnedNcclScopeError(RuntimeError):
    pass


def canonical_uuid(value):
    return str(uuid.UUID(str(value).removeprefix('GPU-')))


def gpu_inventory():
    import csv
    devices = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,uuid,memory.total,memory.used',
         '--format=csv,noheader,nounits'], text=True)
    processes = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_gpu_memory',
         '--format=csv,noheader,nounits'], text=True)
    return dict(hostname=socket.gethostname(),
        devices=[dict(index=int(a),uuid=b.strip(),total_mib=int(c),used_mib=int(d))
                 for a,b,c,d in csv.reader(devices.splitlines())],
        processes=[dict(pid=int(a),uuid=b.strip(),used_mib=c.strip())
                   for a,b,c in csv.reader(processes.splitlines()) if a.strip().isdigit()])


class NativeNodeOwner(RTPServer):
    """One local process tree, holding a contiguous subset of native TP ranks."""
    def __init__(self, replica_rank, model_path, topology, node_index):
        self.topology = ReplicaTopology.from_dict(topology)
        self.node_index = int(node_index)
        self.node = self.topology.node_groups[self.node_index]
        self.main_audit_root = _audit_root().resolve()
        self.nccl_scope = scope_config(_deployment_root())
        self.boot = -1
        super().__init__(replica_rank, model_path, ','.join(self.node['gpu_ids']), self.topology.tp_size)
        self.owner_identity = dict(pid=os.getpid(),create_time=psutil.Process().create_time(),
            owner_hostname=socket.gethostname(),owner_ip=self.node['host'],node_index=self.node_index,
            audit_dir=str(self.main_audit_root),root=str(_deployment_root()),
            ray_actor=dict(name=f'rtp_native_owner_{replica_rank}_{self.node_index}',
                           namespace=ray.get_runtime_context().namespace))

    async def identity(self):
        return self.owner_identity

    async def stop_if_owned(self,expected_pid,expected_create_time,expected_audit_dir):
        identity=self.owner_identity
        if (int(expected_pid)!=identity['pid'] or float(expected_create_time)!=identity['create_time']
                or str(Path(expected_audit_dir).resolve())!=identity['audit_dir']):
            raise RuntimeError('Shutdown does not own this native node actor')
        return await self.stop_for_training()

    async def start(self, boot, incarnation):
        if self.process is not None and self.process.poll() is None:
            raise RuntimeError('Native owner still holds a running incarnation')
        if self.process is not None and self._cleanup_receipt is None:
            self._terminate()
        self.boot = int(boot)
        self.poisoned = False
        self._cleanup_receipt = None
        self._owned_processes = {}
        self.process = self.process_identity = self.log = None
        own_root = self.main_audit_root/f'boots/boot-{self.boot:03d}/node-{self.node_index:02d}'
        own_root.mkdir(parents=True, exist_ok=True)
        if (own_root/f'rtp_process_{self.replica_rank}.json').exists():
            raise RuntimeError('A native boot requires fresh owner evidence')
        os.environ['RLLM_RTP_ACCEPTANCE_DIR'] = str(own_root)
        _write_json(own_root/f'rtp_topology_{self.replica_rank}.json', self.topology.to_dict())
        addresses = socket.gethostbyname_ex(socket.gethostname())[2]
        if self.node['host'] not in addresses:
            raise RuntimeError(f'Native owner is on {addresses}, expected {self.node["host"]}')
        for port in range(self.topology.start_port-11,
                          self.topology.start_port+len(self.node['ranks'])*self.topology.rank_port_stride+12):
            with socket.socket() as probe:
                probe.bind((self.node['host'], port))
        environment = dict(os.environ)
        for key in ('RANK','WORLD_RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','MASTER_ADDR','MASTER_PORT',
                    'DISTRIBUTE_CONFIG_FILE'):
            environment.pop(key, None)
        environment.update(self.topology.node_environment(self.node_index,incarnation))
        environment.update(RTP_MODEL_PATH=self.model_path,
            RTP_WEIGHT_MANIFEST_DIR=str(own_root/f'rtp_weight_manifest_{self.replica_rank}'))
        self.log = (own_root/f'rtp_replica_{self.replica_rank}.log').open('xb')
        self.process = subprocess.Popen(['/bin/bash',os.environ['RTP_SERVER_LAUNCHER']],
            env=environment, stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)
        leader = psutil.Process(self.process.pid)
        self.process_identity = dict(pid=leader.pid,create_time=leader.create_time(),
            pgid=os.getpgid(leader.pid),replica_rank=self.replica_rank,boot_id=self.boot,
            incarnation=incarnation,node_index=self.node_index,owner_hostname=socket.gethostname(),
            owner_ip=self.node['host'],owned_tp_ranks=self.node['ranks'],tp_size=self.tp_size,
            root=str(_deployment_root()),audit_dir=str(own_root),started=time.time(),
            ray_actor=dict(name=f'rtp_native_owner_{self.replica_rank}_{self.node_index}',
                           namespace=ray.get_runtime_context().namespace))
        self._owned_processes[leader.pid] = leader.create_time()
        _write_json(own_root/f'rtp_process_{self.replica_rank}.json', self.process_identity)
        deadline = time.monotonic()+int(os.environ.get('RLLM_RTP_STARTUP_TIMEOUT','5400'))
        last_error = None
        try:
            while time.monotonic()<deadline:
                self._capture_process_tree()
                if self.process.poll() is not None:
                    raise RuntimeError(f'Native node {self.node_index} exited {self.process.returncode}')
                try:
                    control = await _new_control(self.replica_rank)
                    try:
                        statuses = await control.status()
                    finally:
                        await control.close()
                    health = await asyncio.gather(*[
                        asyncio.to_thread(_check_native_http_health,self.node['host'],
                            self.topology.start_port+local*self.topology.rank_port_stride+5,rank,2)
                        for local,rank in enumerate(self.node['ranks'])])
                    manifests = []
                    for rank in self.node['ranks']:
                        path=own_root/f'rtp_weight_manifest_{self.replica_rank}/rank-{rank:05d}.json'
                        record=json.loads(path.read_text())
                        if record['tp_rank']!=rank or record['tp_size']!=self.tp_size:
                            raise RuntimeError('Native manifest global rank/TP world mismatch')
                        process=psutil.Process(int(record['pid']))
                        if self._owned_processes.get(process.pid)!=process.create_time() or not self._running(process):
                            raise RuntimeError('Native rank is not a live owned process')
                        record.update(owner_hostname=socket.gethostname(),owner_ip=self.node['host'],
                            owner_node_index=self.node_index,boot_id=self.boot,incarnation=incarnation)
                        manifests.append(record)
                    inventory=gpu_inventory()
                    expected={str(row['index']):canonical_uuid(row['uuid']) for row in inventory['devices']}
                    expected=[expected.get(gpu,canonical_uuid(gpu) if gpu.startswith('GPU-') else '')
                              for gpu in self.node['gpu_ids']]
                    if [canonical_uuid(row['gpu_uuid']) for row in manifests]!=expected:
                        raise RuntimeError('Native rank UUIDs differ from this node GPU assignments')
                    native_pids={int(row['pid']) for row in manifests}
                    live=[row for row in inventory['processes'] if row['pid'] in native_pids]
                    if {row['pid'] for row in live}!=native_pids:
                        raise RuntimeError('nvidia-smi does not identify every local native rank')
                    nccl_scope_proofs=[]
                    for rank in manifests:
                        pid=int(rank['pid']);created=self._owned_processes[pid]
                        try:
                            native_proof=inspect_pid_nccl(pid,config=self.nccl_scope)
                            if psutil.Process(pid).create_time()!=created:
                                raise RuntimeError('Native PID identity changed during NCCL inspection')
                        except Exception as error:
                            raise PinnedNcclScopeError('Native rank NCCL mapping failed: '+repr(error)) from error
                        proof_file=own_root/f'nccl-scope/replica-{self.replica_rank}/rank-{rank["tp_rank"]:05d}.json'
                        row=dict(schema='q397-native-rank-nccl-scope-v1',replica_rank=self.replica_rank,
                            tp_rank=rank['tp_rank'],pid=pid,create_time=created,gpu_uuid=rank['gpu_uuid'],
                            owner_ip=self.node['host'],native_incarnation=incarnation,proof=native_proof,
                            proof_file=str(proof_file.relative_to(self.main_audit_root)))
                        _write_json(proof_file,row);nccl_scope_proofs.append(row)
                    proof=dict(passed=True,node_index=self.node_index,boot_id=self.boot,
                        incarnation=incarnation,hostname=socket.gethostname(),node_ip=self.node['host'],
                        local_ranks=self.node['ranks'],rank_manifests=manifests,
                        process_identity=self.process_identity,http_health=health,inventory=inventory,
                        nccl_scope_proofs=nccl_scope_proofs,
                        statuses=[dict(tp_rank=r.tp_rank,tp_size=r.tp_size,model_version=r.model_version,
                            global_step=r.global_step,phase=r.phase) for r in statuses],checked_unix=time.time())
                    _write_json(own_root/f'rtp_startup_readiness_{self.replica_rank}.json',proof)
                    return proof
                except PinnedNcclScopeError:
                    raise
                except Exception as error:
                    last_error=repr(error)
                    await asyncio.sleep(1)
            raise TimeoutError('Native owner startup failed: '+str(last_error))
        except BaseException:
            await asyncio.to_thread(self._terminate)
            raise

    async def publish(self, manifests, topology, boot_proof):
        if (boot_proof['replica_rank']!=self.replica_rank or boot_proof['native_boot']!=self.boot
                or boot_proof['native_incarnation']!=self.process_identity['incarnation']):
            raise RuntimeError('NCCL proof publication belongs to another native boot')
        originals=[]
        for node in boot_proof['nodes']:
            ranks={rank['tp_rank']:rank for rank in node['rank_manifests']}
            for row in node.get('nccl_scope_proofs',[]):
                rank=int(row['tp_rank']);relative=Path(row['proof_file'])
                expected=Path(f'boots/boot-{self.boot:03d}/node-{node["node_index"]:02d}/nccl-scope/replica-{self.replica_rank}/rank-{rank:05d}.json')
                if (relative.is_absolute() or '..' in relative.parts or relative!=expected
                        or row['replica_rank']!=self.replica_rank
                        or row['native_incarnation']!=boot_proof['native_incarnation']
                        or rank not in ranks or row['pid']!=ranks[rank]['pid']
                        or row['gpu_uuid']!=ranks[rank]['gpu_uuid']
                        or row['proof']['pid']!=row['pid']):
                    raise RuntimeError('Native NCCL original proof identity/path differs from boot manifest')
                originals.append((rank,relative,row))
        if sorted(rank for rank,_,_ in originals)!=list(range(self.tp_size)):
            raise RuntimeError('Native NCCL publication requires all TP original proofs exactly once')
        for _,relative,row in originals:
            destination=self.main_audit_root/relative
            if destination.exists() and json.loads(destination.read_text())!=row:
                raise RuntimeError('Conflicting native NCCL original proof during publication')
            _write_json(destination,row)
        directory=self.main_audit_root/f'rtp_weight_manifest_{self.replica_rank}'
        directory.mkdir(parents=True,exist_ok=True)
        for record in manifests:
            _write_json(directory/f'rank-{record["tp_rank"]:05d}.json',record)
        _write_json(self.main_audit_root/f'rtp_topology_{self.replica_rank}.json',topology)
        _write_json(self.main_audit_root/f'rtp_boot_{self.replica_rank}_step{boot_proof["target_global_step"]}.json',boot_proof)
        return dict(published=True,node_index=self.node_index,manifest_count=len(manifests),
                    nccl_scope_proof_count=len(originals))

    async def stop_for_training(self):
        if self.process is not None:
            await asyncio.to_thread(self._terminate)
            if self.log is not None:
                self.log.close()
        result=dict(node_index=self.node_index,owner_hostname=socket.gethostname(),owner_ip=self.node['host'],
            boot_id=self.boot,**(self._cleanup_receipt or dict(process_tree_terminated=True,owned_pids=[])))
        inventory=gpu_inventory()
        owned=set(result['owned_pids'])
        remaining=[]
        for row in inventory['processes']:
            if row['pid'] in owned:
                try:
                    if psutil.Process(row['pid']).create_time()==self._owned_processes.get(row['pid']):
                        remaining.append(row)
                except psutil.NoSuchProcess:
                    pass
        if remaining:
            raise RuntimeError('Native GPU contexts remain after node cleanup')
        result.update(owned_gpu_processes_after=remaining,finished_at_unix=time.time())
        return result

    async def inventory(self):
        return gpu_inventory()


class MultiNodeRTPServer(RTPServer):
    """Stable RPC/gateway actor; native process ownership lives on each host."""
    def __init__(self, replica_rank, model_path, topology, owners):
        self.topology=ReplicaTopology.from_dict(topology)
        self.owners=owners
        self.boot=-1
        self.incarnation=None
        self._started=False
        self._owners_active=False
        self._generation_ready=False
        self._committed_version=None
        self._lifecycle_lock=asyncio.Lock()
        self._last_suspend=None
        self._owner_identities=[]
        super().__init__(replica_rank,model_path,'',self.topology.tp_size)
        self.controller_identity=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),
            root=str(_deployment_root()),audit_dir=str(_audit_root()),replica_rank=replica_rank,
            owner_hostname=socket.gethostname(),
            ray_actor=dict(name=f'rtp_server_{replica_rank}_0',namespace=ray.get_runtime_context().namespace),
            node_owner_count=len(owners),topology=self.topology.to_dict(),node_owners=[])
        _write_json(_audit_root()/f'rtp_cluster_{replica_rank}.json',self.controller_identity)

    def _native_available(self):
        return self._started and self._generation_ready

    def _terminate(self):
        # The driver and explicit shutdown await all owners before ray.shutdown.
        # atexit cannot start a second async transaction while Ray is shutting down.
        return None

    async def launch_server(self):
        await self.ensure_native_running(0,None)
        return self.topology.addresses[0]

    async def _prepare_owners(self):
        if not self._owner_identities:
            identities=await asyncio.gather(*[owner.identity.remote() for owner in self.owners])
            for index,identity in enumerate(identities):
                if (identity['node_index']!=index or identity['owner_ip']!=self.topology.node_groups[index]['host']
                        or identity['audit_dir']!=str(_audit_root())):
                    raise RuntimeError('Native node owner identity is outside this deployment')
            self._owner_identities=identities
            self.controller_identity['node_owners']=identities
            _write_json(_audit_root()/f'rtp_cluster_{self.replica_rank}.json',self.controller_identity)
        return self.controller_identity

    async def prepare(self):
        """Let the driver persist all remote owners before any native GPU start."""
        async with self._lifecycle_lock:
            return await self._prepare_owners()

    async def ensure_native_running(self,global_step,previous_committed_version):
        async with self._lifecycle_lock:
            if self.poisoned:
                raise RuntimeError('Cluster controller was permanently shut down')
            if self._committed_version!=previous_committed_version:
                raise RuntimeError('RTP refit does not reference the last committed policy')
            if self._started:
                return dict(native_boot=self.boot,native_incarnation=self.incarnation,restarted=False)
            if not self._owner_identities:
                await self._prepare_owners()
            self.boot+=1
            self.incarnation=uuid.uuid4().hex
            self._generation_ready=False
            self._owners_active=True
            results=await asyncio.gather(*[owner.start.remote(self.boot,self.incarnation) for owner in self.owners],return_exceptions=True)
            if any(isinstance(value,BaseException) for value in results):
                receipts=await asyncio.gather(*[owner.stop_for_training.remote() for owner in self.owners],return_exceptions=True)
                self._owners_active=any(isinstance(value,BaseException) for value in receipts)
                raise RuntimeError('Native cross-node startup failed: '+str(results))
            try:
                return await self._finish_start(results,global_step,previous_committed_version)
            except BaseException:
                if self.client is not None:
                    await self.client.close()
                    self.client=None
                receipts=await asyncio.gather(*[owner.stop_for_training.remote() for owner in self.owners],return_exceptions=True)
                self._owners_active=any(isinstance(value,BaseException) for value in receipts)
                self._started=self._generation_ready=False
                raise

    async def _finish_start(self,results,global_step,previous_committed_version):
        manifests=sorted([row for proof in results for row in proof['rank_manifests']],key=lambda row:row['tp_rank'])
        if [row['tp_rank'] for row in manifests]!=list(range(self.tp_size)) or len({canonical_uuid(row['gpu_uuid']) for row in manifests})!=self.tp_size:
            raise RuntimeError('Native cluster is not a complete physical TP world')
        proof=dict(passed=True,replica_rank=self.replica_rank,native_boot=self.boot,
            native_incarnation=self.incarnation,target_global_step=int(global_step),
            previous_committed_version=previous_committed_version,nodes=results,
            manifest_count=len(manifests),started_unix=min(row['process_identity']['started'] for row in results),
            ready_unix=time.time())
        await asyncio.gather(*[owner.publish.remote(manifests,self.topology.to_dict(),proof) for owner in self.owners])
        _write_json(_audit_root()/f'rtp_boot_{self.replica_rank}_step{global_step}.json',proof)
        _write_json(_audit_root()/f'rtp_topology_{self.replica_rank}.json',self.topology.to_dict())
        _source()
        from transformers import AutoTokenizer,AutoConfig
        from rtp_llm.RL.token_rpc_client import TokenRpcClient,TokenProtocolError
        class StrictTokenRpcClient(TokenRpcClient):
            async def _request(self,request_id,ids,config):
                response=await super()._request(request_id,ids,config)
                if config.return_softmax_probs:
                    infos=response.flatten_output.aux_info
                    if len(infos)!=1 or 'raw_logprobs' not in infos[0].DESCRIPTOR.fields_by_name or not infos[0].HasField('raw_logprobs'):
                        raise TokenProtocolError('Native FP32 raw logprobs are required')
                return response
        self.tokenizer=AutoTokenizer.from_pretrained(self.model_path,local_files_only=True)
        config=AutoConfig.from_pretrained(self.model_path,local_files_only=True)
        generation=json.loads((Path(self.model_path)/'generation_config.json').read_text())
        eos=generation.get('eos_token_id',self.tokenizer.eos_token_id)
        eos=eos if isinstance(eos,list) else [eos]
        class JournaledStrictTokenRpcClient(JournaledTokenRpcMixin, StrictTokenRpcClient):
            pass
        journal=NativeRpcJournal(_audit_root(),self.replica_rank,
            lambda:dict(native_boot=self.boot,native_incarnation=self.incarnation))
        self.client=JournaledStrictTokenRpcClient(self.topology.addresses[0],request_journal=journal,
            vocab_size=getattr(config,'text_config',config).vocab_size,eos_token_ids=eos,
            timeout=float(os.environ.get('RLLM_RTP_TOKEN_TIMEOUT','600')))
        self._started=True
        return dict(native_boot=self.boot,native_incarnation=self.incarnation,restarted=self.boot>0)

    async def generate(self,*args,**kwargs):
        if not self._generation_ready:
            raise RuntimeError('Fresh native checkpoint is awaiting full policy refit and probe')
        return await super().generate(*args,**kwargs)

    async def probe_synced_policy(self,expected_step,expected_version):
        # A successful native resume precedes this post-refit score probe.
        # Generation remains closed until the unchanged strict probe passes.
        self.client.open_admission()
        try:
            result=await super().probe_synced_policy(expected_step,expected_version)
        except BaseException:
            self.client.close_admission('policy_probe_failed')
            raise
        self._committed_version=expected_version
        self._generation_ready=True
        return result

    async def accept_cold_checkpoint(self):
        if not self._started:
            raise RuntimeError('Cold native world is not running')
        self._generation_ready=True

    async def pause_for_training(self):
        async with self._lifecycle_lock:
            if not self._started:
                raise RuntimeError('Native world must be live before a training pause')
            self._generation_ready=False
            self.client.close_admission('training_pause')
            from rtp_native_server import _sleep
            await _sleep(self.replica_rank)
            control=await _new_control(self.replica_rank)
            try:
                statuses=await control.status()
            finally:
                await control.close()
            if len(statuses)!=self.tp_size or any(value.phase!='paused' for value in statuses):
                raise RuntimeError('Training pause was not acknowledged by every native rank')
            self._generation_ready=False
            proof=dict(paused=True,replica_rank=self.replica_rank,native_boot=self.boot,
                native_incarnation=self.incarnation,previous_committed_version=self._committed_version,
                kv_pool_retained=os.environ.get('RLLM_RTP_KEEP_KV_CACHE','1')=='1',
                native_processes_released=False,statuses=[dict(tp_rank=value.tp_rank,tp_size=value.tp_size,
                    model_version=value.model_version,global_step=value.global_step,phase=value.phase)
                    for value in statuses],finished_at_unix=time.time())
            _write_json(_audit_root()/f'rtp_training_pause_{self.replica_rank}_version{self._committed_version}.json',proof)
            return proof

    async def suspend_for_training(self):
        async with self._lifecycle_lock:
            if not self._owners_active:
                return self._last_suspend or dict(process_tree_terminated=True,nodes=[])
            pause_error=None
            self._generation_ready=False
            if self.client is not None:
                self.client.close_admission('native_suspend')
            if self._started:
                control=None
                try:
                    control=await _new_control(self.replica_rank)
                    await control.pause()
                except Exception as error:
                    pause_error=repr(error)
                finally:
                    if control is not None:
                        await control.close()
            if self.client is not None:
                await self.client.close()
                self.client=None
            receipts=await asyncio.gather(*[owner.stop_for_training.remote() for owner in self.owners],return_exceptions=True)
            if any(isinstance(row,BaseException) for row in receipts):
                raise RuntimeError('Native owner cleanup failed after awaiting every owner: '+str(receipts))
            if not all(row['process_tree_terminated'] and not row['owned_gpu_processes_after'] for row in receipts):
                raise RuntimeError('Both native owners must release all GPU contexts')
            result=dict(process_tree_terminated=True,replica_rank=self.replica_rank,native_boot=self.boot,
                native_incarnation=self.incarnation,previous_committed_version=self._committed_version,
                nodes=receipts,pause_error=pause_error,finished_at_unix=time.time())
            self._started=self._owners_active=self._generation_ready=False
            self._last_suspend=result
            _write_json(_audit_root()/f'rtp_suspend_{self.replica_rank}_boot{self.boot}.json',result)
            return result

    async def sleep(self):
        return await self.pause_for_training()

    async def shutdown(self):
        result=await self.suspend_for_training()
        self.poisoned=True
        _write_json(_audit_root()/f'rtp_cluster_shutdown_{self.replica_rank}.json',result)
        return result

    async def shutdown_if_owned(self,expected_pid,expected_create_time,expected_audit_dir):
        identity=self.controller_identity
        if (expected_pid!=identity['pid'] or expected_create_time!=identity['create_time']
                or str(Path(expected_audit_dir).resolve())!=identity['audit_dir']):
            raise RuntimeError('Shutdown does not own this cluster controller')
        return await self.shutdown()

    async def node_inventories(self):
        return await asyncio.gather(*[owner.inventory.remote() for owner in self.owners])


def launch_multinode_server(model_path,topology,node_ids):
    """Create the stable controller and local owners on an existing Ray cluster."""
    topology=ReplicaTopology.from_dict(topology)
    if not ray.is_initialized() or len(node_ids)!=len(topology.node_groups):
        raise ValueError('A live Ray placement for every native owner is required')
    if len(set(node_ids))!=len(node_ids):
        raise ValueError('Native owners must be placed on distinct Ray nodes')
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
    owner_class=ray.remote(num_cpus=1,num_gpus=0)(NativeNodeOwner)
    owners=[]
    runtime={'env_vars':{'RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES':'1',
                        **scope_config(_deployment_root())['environment']}}
    for index,node_id in enumerate(node_ids):
        owners.append(owner_class.options(name=f'rtp_native_owner_{topology.replica_rank}_{index}',
            max_concurrency=16,scheduling_strategy=NodeAffinitySchedulingStrategy(node_id,soft=False),
            runtime_env=runtime).remote(topology.replica_rank,model_path,topology.to_dict(),index))
    controller_class=ray.remote(num_cpus=1,num_gpus=0)(MultiNodeRTPServer)
    controller=controller_class.options(name=f'rtp_server_{topology.replica_rank}_0',max_concurrency=128,
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_ids[0],soft=False),runtime_env=runtime).remote(
            topology.replica_rank,model_path,topology.to_dict(),owners)
    return controller,owners
