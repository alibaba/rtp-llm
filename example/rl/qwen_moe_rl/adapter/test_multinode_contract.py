"""CPU-only tests for multi-node routing and owned lifecycle; no model acceptance."""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
import uuid

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from rtp_topology import ReplicaTopology,training_rank_layout


def load_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


class AsyncRemote:
    def __init__(self,method): self.method=method
    def remote(self,*args,**kwargs): return self.method(*args,**kwargs)


class FakeBase:
    pass


@contextmanager
def modules(root):
    saved=dict(sys.modules)
    ray=types.ModuleType('ray')
    ray.util=types.SimpleNamespace(get_node_ip_address=lambda:'192.0.2.1')
    torch=types.ModuleType('torch')
    torch.no_grad=lambda:lambda value:value
    torch.cuda=types.SimpleNamespace(current_device=lambda:0,
        get_device_properties=lambda index:types.SimpleNamespace(uuid='GPU-'+str(uuid.UUID(int=1))))
    native=types.ModuleType('rtp_native_server')
    native.RTPServer=FakeBase
    native._audit_root=lambda:root
    native._deployment_root=lambda:root.parent
    native._source=lambda:None
    native._base_port=lambda replica:24000+1000*replica
    def write(path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value))
    native._write_json=write
    for name in ['_check_native_http_health','_new_control','_sleep','_wake']:
        setattr(native,name,None)
    sys.modules.update(psutil=types.ModuleType('psutil'),ray=ray,torch=torch,rtp_native_server=native)
    sys.modules['verl.workers.rollout.base']=types.SimpleNamespace(BaseRollout=FakeBase)
    sys.modules['verl.workers.rollout.replica']=types.SimpleNamespace(RolloutReplica=FakeBase)
    cluster=load_file('rtp_multinode_server',HERE/'rtp_multinode_server.py')
    backend=load_file('rtp_verl_backend',HERE/'rtp_verl_backend.py')
    cleanup=load_file('rtp_cleanup_test',HERE/'rtp_cleanup.py')
    try:
        yield cluster,backend,cleanup,ray,native
    finally:
        for key in list(sys.modules):
            if key not in saved: del sys.modules[key]
        sys.modules.update(saved)


def topology(replica=0):
    return ReplicaTopology(replica_rank=replica,tp_size=16,gpu_ids=tuple(map(str,range(4)))*4,
        rank_hosts=tuple(f'192.0.2.{replica*4+node+1}' for node in range(4) for rank in range(4)),
        start_port=24000+1000*replica)


def controller(module,owners,topo=None):
    obj=module.MultiNodeRTPServer.__new__(module.MultiNodeRTPServer)
    obj.topology=topo or topology()
    obj.tp_size=16
    obj.replica_rank=0
    obj.owners=owners
    obj._lifecycle_lock=asyncio.Lock()
    obj._started=False
    obj._owners_active=False
    obj._generation_ready=False
    obj._committed_version=None
    obj._owner_identities=[{}]*len(owners)
    obj.boot=-1
    obj.incarnation=None
    obj.poisoned=False
    obj.client=None
    return obj


class TopologyTests(unittest.TestCase):
    def test_two_replicas_four_nodes_each(self):
        for replica in range(2):
            topo=topology(replica)
            self.assertEqual(ReplicaTopology.from_dict(topo.to_dict()),topo)
            self.assertEqual(len(topo.node_groups),4)
            for node in range(4):
                env=topo.node_environment(node,'incarnation')
                self.assertEqual(env['LOCAL_WORLD_SIZE'],'4')
                self.assertEqual(env['RTP_WORLD_RANK'],str(node*4))
                self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'0,1,2,3')
                self.assertEqual(env['RTP_TP_SIZE'],'16')
                self.assertEqual(len(env['GANG_CONFIG_STRING'].split(';')),4)
                ip=f'192.0.2.{replica*4+node+1}'
                self.assertEqual(topo.addresses[node*4:(node+1)*4],
                    [f'{ip}:{topo.start_port+local*9+1}' for local in range(4)])
            for rank in range(replica*16,(replica+1)*16):
                layout=training_rank_layout(rank,32,16)
                self.assertEqual(layout['replica_rank'],replica)
                self.assertEqual(layout['tp_rank'],rank%16)
                self.assertEqual(layout['replica_count'],2)

    def test_rejects_duplicate_device_and_interleaved_nodes(self):
        with self.assertRaises(ValueError):
            ReplicaTopology(0,2,('0','0'),24000,rank_hosts=('a','a'))
        with self.assertRaises(ValueError):
            ReplicaTopology(0,4,('0','0','1','1'),24000,rank_hosts=('a','b','a','b'))
        with self.assertRaises(ValueError):
            ReplicaTopology(0,3,('0','1','0'),24000,rank_hosts=('a','a','b'))

    def test_rejects_recorded_endpoint_reordering(self):
        record=topology().to_dict()
        record['control_addresses'][0]='127.0.0.1:24001'
        with self.assertRaises(ValueError): ReplicaTopology.from_dict(record)


class LifecycleTests(unittest.TestCase):
    def test_stale_version_rejected_even_when_running(self):
        with tempfile.TemporaryDirectory() as tmp,modules(Path(tmp)) as (mod,*rest):
            obj=controller(mod,[])
            obj._started=True
            obj._committed_version='committed'
            with self.assertRaisesRegex(RuntimeError,'last committed'):
                asyncio.run(obj.ensure_native_running(1,'stale'))

    def test_partial_start_failure_cleans_every_owner(self):
        with tempfile.TemporaryDirectory() as tmp,modules(Path(tmp)) as (mod,*rest):
            stopped=[]
            class Owner:
                def __init__(self,index):
                    self.index=index
                    self.start=AsyncRemote(self._start)
                    self.stop_for_training=AsyncRemote(self._stop)
                async def _start(self,*args):
                    if self.index==2: raise RuntimeError('injected startup failure')
                    return {}
                async def _stop(self):
                    stopped.append(self.index)
                    return dict(process_tree_terminated=True)
            obj=controller(mod,[Owner(index) for index in range(4)])
            with self.assertRaisesRegex(RuntimeError,'startup failed'):
                asyncio.run(obj.ensure_native_running(0,None))
            self.assertEqual(sorted(stopped),[0,1,2,3])
            self.assertFalse(obj._owners_active)
            self.assertFalse(obj._started)

    def test_manifest_failure_after_start_cleans_every_owner(self):
        with tempfile.TemporaryDirectory() as tmp,modules(Path(tmp)) as (mod,*rest):
            stopped=[]
            class Owner:
                def __init__(self,index):
                    self.index=index
                    self.start=AsyncRemote(self._start)
                    self.stop_for_training=AsyncRemote(self._stop)
                async def _start(self,*args):
                    return dict(rank_manifests=[dict(tp_rank=self.index*4+rank,
                        gpu_uuid=str(uuid.UUID(int=1))) for rank in range(4)])
                async def _stop(self):
                    stopped.append(self.index)
                    return dict(process_tree_terminated=True)
            obj=controller(mod,[Owner(index) for index in range(4)])
            with self.assertRaisesRegex(RuntimeError,'complete physical TP world'):
                asyncio.run(obj.ensure_native_running(0,None))
            self.assertEqual(sorted(stopped),[0,1,2,3])
            self.assertFalse(obj._owners_active)
            self.assertFalse(obj._generation_ready)

    def test_training_pause_retains_native_and_blocks_generation(self):
        with tempfile.TemporaryDirectory() as tmp,modules(Path(tmp)) as (mod,backend,cleanup,ray,native):
            obj=controller(mod,[])
            obj._started=obj._owners_active=obj._generation_ready=True
            obj._committed_version='v1'
            obj.boot=0
            obj.incarnation='inc'
            admission_reasons=[]
            obj.client=types.SimpleNamespace(close_admission=admission_reasons.append)
            class Control:
                async def status(self):
                    return [types.SimpleNamespace(tp_rank=rank,tp_size=16,model_version='v1',
                        global_step=1,phase='paused') for rank in range(16)]
                async def close(self): pass
            async def control(replica): return Control()
            async def sleep(replica): pass
            native._sleep=sleep
            mod._new_control=control
            result=asyncio.run(obj.pause_for_training())
            self.assertTrue(result['paused'])
            self.assertEqual(len(result['statuses']),16)
            self.assertFalse(result['native_processes_released'])
            self.assertTrue(obj._started)
            self.assertTrue(obj._owners_active)
            self.assertFalse(obj._generation_ready)
            self.assertEqual(admission_reasons,['training_pause'])


class IpcAndCleanupTests(unittest.TestCase):
    def test_readiness_failure_still_drains_global_fsdp_export(self):
        with tempfile.TemporaryDirectory() as tmp,modules(Path(tmp)) as (mod,backend,cleanup,ray,native):
            obj=backend.RTPServerAdapter.__new__(backend.RTPServerAdapter)
            obj.is_leader=True;obj.replica_rank=0;obj._last_sync_version=None
            obj._broadcast=lambda value:value
            obj._gather=lambda value:[value]*16
            obj._any_global_failure=lambda error:True
            consumed=[];terminated=[]
            class Server:
                def __init__(self): self.ensure_native_running=AsyncRemote(self.ensure)
                async def ensure(self,*args): raise RuntimeError('injected native readiness failure')
            ray.get_actor=lambda name:Server()
            async def terminate(): terminated.append(True)
            obj._terminate_owned_replica=terminate
            sys.modules['rtp_llm.utils.cuda_ipc']=types.SimpleNamespace(CudaIpcV2=object)
            def export():
                for index in range(4):
                    consumed.append(index)
                    yield str(index),object()
            with self.assertRaisesRegex(RuntimeError,'synchronization failed'):
                asyncio.run(obj.update_weights(export(),global_steps=1))
            self.assertEqual(consumed,[0,1,2,3])
            self.assertEqual(terminated,[True])

    def test_ipc_requires_same_host_and_physical_gpu(self):
        with tempfile.TemporaryDirectory() as tmp,modules(Path(tmp)) as (mod,backend,cleanup,ray,native):
            obj=backend.RTPServerAdapter.__new__(backend.RTPServerAdapter)
            obj.replica_rank=0;obj.tp_rank=0;obj.tp_size=16;obj.global_rank=0
            path=Path(tmp)/'rtp_weight_manifest_0/rank-00000.json'
            path.parent.mkdir()
            record=dict(tp_rank=0,tp_size=16,gpu_uuid=str(uuid.UUID(int=1)),owner_ip='192.0.2.1',
                pid=10,boot_id=0,incarnation='inc')
            path.write_text(json.dumps(record))
            self.assertTrue(obj._local_ipc_binding(0)['passed'])
            record['owner_ip']='192.0.2.2';path.write_text(json.dumps(record))
            with self.assertRaisesRegex(RuntimeError,'not colocated'): obj._local_ipc_binding(0)
            record['owner_ip']='192.0.2.1';record['gpu_uuid']=str(uuid.UUID(int=2))
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(RuntimeError,'not colocated'): obj._local_ipc_binding(0)

    def test_dead_controller_uses_all_identity_checked_node_owners(self):
        with tempfile.TemporaryDirectory() as tmp,modules(Path(tmp)) as (mod,backend,cleanup,ray,native):
            root=Path(tmp).resolve()
            stopped=[]
            class Owner:
                def __init__(self,index):
                    self.stop_if_owned=types.SimpleNamespace(remote=lambda pid,created,audit:
                        self.stop(index,pid,created,audit))
                def stop(self,index,pid,created,audit):
                    self_expected=(100+index,float(index),str(root))
                    if (pid,created,audit)!=self_expected: raise RuntimeError('identity mismatch')
                    stopped.append(index)
                    return dict(node_index=index,process_tree_terminated=True,owned_gpu_processes_after=[])
            def lookup(name,namespace):
                if name=='controller': raise RuntimeError('controller actor died')
                return Owner(int(name.removeprefix('owner')))
            ray.get_actor=lookup
            ray.is_initialized=lambda:True
            ray.get=lambda values,timeout:values
            owners=[dict(pid=100+index,create_time=float(index),audit_dir=str(root),
                ray_actor=dict(name=f'owner{index}',namespace='owned')) for index in range(4)]
            identity=dict(pid=1,create_time=1.0,node_owner_count=4,node_owners=owners,
                ray_actor=dict(name='controller',namespace='owned'))
            result=cleanup._cleanup_multinode(identity,root)
            self.assertEqual(result['method'],'owned_multinode_owner_rpc')
            self.assertEqual(sorted(stopped),[0,1,2,3])
            ray.is_initialized=lambda:False
            with self.assertRaisesRegex(RuntimeError,'live Ray cluster'):
                cleanup._cleanup_multinode(identity,root)


if __name__=='__main__': unittest.main(verbosity=2)
