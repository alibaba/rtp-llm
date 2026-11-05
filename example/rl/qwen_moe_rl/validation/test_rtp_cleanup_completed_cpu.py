"""CPU-only unit coverage for completed native cleanup and fail-closed fallback."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve()
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--project-root', type=Path, default=HERE.parents[1])
options, remaining = parser.parse_known_args()
sys.path.insert(0, str(options.project_root/'adapter'))
try:
    import psutil
except ModuleNotFoundError:
    # Unit-only process facade on local hosts without psutil. ARM integration
    # uses the installed real module and actual /proc/NVML observations.
    facade=types.ModuleType('psutil')
    class NoSuchProcess(Exception): pass
    class ZombieProcess(Exception): pass
    facade.NoSuchProcess=NoSuchProcess
    facade.ZombieProcess=ZombieProcess
    facade.STATUS_ZOMBIE='zombie'; facade.STATUS_DEAD='dead'
    facade.Process=lambda pid:None
    facade.process_iter=lambda *a:[]
    sys.modules['psutil']=facade
class ActorDiedError(Exception): pass
ray_exceptions=types.ModuleType('ray.exceptions')
ray_exceptions.ActorDiedError=ActorDiedError
sys.modules.setdefault('ray.exceptions',ray_exceptions)
completed = importlib.import_module('rtp_cleanup_completed')
cleanup = importlib.import_module('rtp_cleanup')


class CompletedCleanupTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.deployment = Path(self.temporary.name).resolve()
        self.root = self.deployment/'runs'/'closed-run'/'rtp'
        self.root.mkdir(parents=True)
        self.incarnation = 'a'*32
        groups, owners, boot_nodes, stopped_nodes = [], [], [], []
        for index in range(4):
            host, ip = f'host-{index}', f'192.0.2.{index+1}'
            ranks = list(range(index*4, (index+1)*4))
            group = dict(host=ip, ranks=ranks, gpu_ids=list(map(str, range(4))))
            owner = dict(pid=9000+index, create_time=10.+index, owner_hostname=host, owner_ip=ip,
                         node_index=index, audit_dir=str(self.root), root=str(self.deployment),
                         ray_actor=dict(name=f'rtp_native_owner_0_{index}', namespace='original-namespace'))
            own = self.root/'boots'/'boot-000'/f'node-{index:02d}'
            own.mkdir(parents=True)
            pid = 10000 + index*100
            descendants = [dict(pid=pid+local+1, create_time=100.+index+local/100) for local in range(4)]
            process = dict(pid=pid, create_time=50.+index, pgid=pid, replica_rank=0, boot_id=0,
                           incarnation=self.incarnation, node_index=index, owner_ip=ip,
                           owner_hostname=host, root=str(self.deployment), audit_dir=str(own),
                           descendants=descendants)
            manifests, scope = [], []
            for local, rank in enumerate(ranks):
                uuid = f'GPU-{rank:032x}'
                value = dict(tp_rank=rank, tp_size=16, pid=descendants[local]['pid'], gpu_uuid=uuid)
                manifests.append(value)
                scope.append(dict(tp_rank=rank, pid=value['pid'], create_time=descendants[local]['create_time'],
                                  gpu_uuid=uuid, native_incarnation=self.incarnation, owner_ip=ip))
                self.write(own/'rtp_weight_manifest_0'/f'rank-{rank:05d}.json', value)
            stopped = dict(node_index=index, owner_ip=ip, owner_hostname=host, boot_id=0,
                           process_tree_terminated=True, pid=pid, create_time=50.+index, pgid=pid,
                           owned_pids=[pid]+[row['pid'] for row in descendants],
                           finished_at_unix=460.+index, owned_gpu_processes_after=[])
            local_stopped = {key:stopped[key] for key in ('process_tree_terminated','pid','create_time','pgid','owned_pids')}
            local_stopped['finished_at_unix'] = 450.+index
            self.write(own/'rtp_process_0.json', process)
            self.write(own/'rtp_shutdown_0.json', local_stopped)
            groups.append(group); owners.append(owner); stopped_nodes.append(stopped)
            boot_nodes.append(dict(node_index=index, process_identity=process, rank_manifests=manifests,
                                   nccl_scope_proofs=scope))
        self.identity = dict(pid=8000, create_time=1., root=str(self.deployment), audit_dir=str(self.root),
                             replica_rank=0, owner_hostname='host-0', node_owner_count=4, node_owners=owners,
                             ray_actor=dict(name='rtp_server_0_0', namespace='original-namespace'),
                             topology=dict(replica_rank=0, tp_size=16, node_groups=groups))
        self.boot = dict(passed=True, replica_rank=0, native_boot=0, native_incarnation=self.incarnation,
                         manifest_count=16, target_global_step=0, nodes=boot_nodes)
        self.canonical = dict(process_tree_terminated=True, replica_rank=0, native_boot=0,
                              native_incarnation=self.incarnation, previous_committed_version=self.incarnation+':3',
                              pause_error=None, nodes=stopped_nodes, finished_at_unix=500.)
        self.sync = dict(replica_rank=0, global_step=2, model_version=self.incarnation+':3', native_boot=0,
                         native_incarnation=self.incarnation, full_source_sha256='b'*64, hf_tensor_count=1038,
                         native_received_weights=1038, native_applied_weights=1024, finished_unix=400.,
                         rank_status=[dict(tp_rank=rank, model_version=self.incarnation+':3') for rank in range(16)])
        self.write(self.root/'rtp_cluster_0.json', self.identity)
        self.write(self.root/'rtp_boot_0_step0.json', self.boot)
        self.write(self.root/'rtp_cluster_shutdown_0.json', self.canonical)
        self.write(self.root/'rtp_sync_rank0_step2.json', self.sync)
        self.source_sha = hashlib.sha256(Path(completed.__file__).read_bytes()).hexdigest()
        self.inventory = dict(gpu_uuids=[f'GPU-{rank:032x}' for rank in range(4)],
                              compute_processes=[dict(pid=999999, gpu_uuid='GPU-'+f'{0:032x}')])
        from contextlib import ExitStack
        self.stack = ExitStack()
        self.stack.enter_context(patch.object(completed.socket, 'gethostname', return_value='host-0'))
        self.stack.enter_context(patch.object(completed.socket, 'gethostbyname_ex', return_value=('host-0', [], ['192.0.2.1'])))
        self.process_patch = self.stack.enter_context(patch.object(completed.psutil, 'Process', side_effect=completed.psutil.NoSuchProcess(1)))
        self.stack.enter_context(patch.object(completed.psutil, 'process_iter', return_value=[]))
        self.inventory_patch = self.stack.enter_context(patch.object(completed, '_gpu_inventory', return_value=self.inventory))

    def tearDown(self):
        self.stack.close(); self.temporary.cleanup()

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2))

    def inspect(self):
        return completed._inspect_completed_node(self.identity, str(self.root), 0, self.source_sha)

    def test_complete_shutdown_allows_unrelated_training_context(self):
        result = self.inspect()
        self.assertTrue(result['proof']['passed'])
        self.assertEqual(result['proof']['owned_native_gpu_contexts'], [])
        self.assertEqual(len(result['proof']['inventory']['compute_processes']), 1)
        self.assertFalse(result['proof']['signals_sent'])
        self.assertEqual(result['proof']['latest_source_sha256'], 'b'*64)

    def test_missing_canonical_is_not_success(self):
        (self.root/'rtp_cluster_shutdown_0.json').unlink()
        self.assertIsNone(self.inspect()['proof'])

    def test_stale_incarnation_is_rejected(self):
        self.canonical['native_incarnation'] = 'c'*32
        self.write(self.root/'rtp_cluster_shutdown_0.json', self.canonical)
        with self.assertRaisesRegex(RuntimeError, 'stale'): self.inspect()

    def test_latest_boot_is_required(self):
        newer = copy.deepcopy(self.boot)
        newer.update(native_boot=1, target_global_step=3, native_incarnation='c'*32)
        self.write(self.root/'rtp_boot_0_step3.json', newer)
        with self.assertRaisesRegex(RuntimeError, 'stale'): self.inspect()

    def test_latest_committed_sync_is_required(self):
        newer = copy.deepcopy(self.sync)
        newer.update(global_step=3, model_version=self.incarnation+':4')
        self.write(self.root/'rtp_sync_rank0_step3.json', newer)
        with self.assertRaisesRegex(RuntimeError, 'latest committed policy'): self.inspect()

    def test_incomplete_canonical_nodes_are_rejected(self):
        self.canonical['nodes'].pop()
        self.write(self.root/'rtp_cluster_shutdown_0.json', self.canonical)
        self.assertIsNone(self.inspect()['proof'])

    def test_canonical_wrong_pid_is_rejected(self):
        self.canonical['nodes'][0]['pid'] += 1
        self.write(self.root/'rtp_cluster_shutdown_0.json', self.canonical)
        with self.assertRaisesRegex(RuntimeError, 'identity differs'): self.inspect()

    def test_changed_local_ctime_is_rejected(self):
        path = self.root/'boots/boot-000/node-00/rtp_process_0.json'
        value=json.loads(path.read_text()); value['create_time'] += 1
        self.write(path,value)
        with self.assertRaisesRegex(RuntimeError, 'persisted native process identity'): self.inspect()

    def test_owned_native_pid_still_present_is_rejected(self):
        self.process_patch.side_effect = None
        self.process_patch.return_value = types.SimpleNamespace(create_time=lambda:50., status=lambda:'running')
        with self.assertRaisesRegex(RuntimeError, 'still present'): self.inspect()

    def test_unknown_live_member_of_original_pgid_is_rejected_without_signal(self):
        process=types.SimpleNamespace(pid=88888, info={'pid':88888,'status':'running'})
        with patch.object(completed.psutil,'process_iter',return_value=[process]), patch.object(completed.os,'getpgid',return_value=10000):
            with self.assertRaisesRegex(RuntimeError, 'original native PGID'): self.inspect()

    def test_owned_gpu_context_is_rejected_even_if_proc_is_gone(self):
        self.inventory['compute_processes'].append(dict(pid=10001, gpu_uuid='GPU-'+f'{0:032x}'))
        with self.assertRaisesRegex(RuntimeError, 'still has a GPU context'): self.inspect()

    def test_wrong_physical_gpu_uuid_is_rejected(self):
        self.inventory['gpu_uuids']=['GPU-'+f'{999:032x}']
        with self.assertRaisesRegex(RuntimeError, 'physical GPU UUIDs'): self.inspect()

    def test_reused_unrelated_pid_context_is_recorded_without_signal(self):
        self.process_patch.side_effect = None
        self.process_patch.return_value = types.SimpleNamespace(create_time=lambda:900., status=lambda:'running')
        self.inventory['compute_processes'].append(dict(pid=10001, gpu_uuid='GPU-'+f'{0:032x}'))
        result=self.inspect()
        self.assertTrue(result['proof']['passed'])
        self.assertTrue(all(row['state']=='pid_reused' for row in result['proof']['native_identities']))

    def test_wrong_inspector_source_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'source SHA'):
            completed._inspect_completed_node(self.identity,str(self.root),0,'0'*64)

    def test_invalid_latest_source_sha_is_rejected(self):
        self.sync['full_source_sha256']='invalid'
        self.write(self.root/'rtp_sync_rank0_step2.json',self.sync)
        with self.assertRaisesRegex(RuntimeError,'digest'): self.inspect()

    def test_incomplete_rank_ack_is_rejected(self):
        self.sync['rank_status'].pop()
        self.write(self.root/'rtp_sync_rank0_step2.json',self.sync)
        with self.assertRaisesRegex(RuntimeError,'rank ACK'): self.inspect()

    def test_owner_permission_error_is_not_missing(self):
        self.assertFalse(cleanup._missing_named_actor(PermissionError('denied'),'x'))
        self.assertFalse(cleanup._missing_named_actor(TimeoutError('timeout'),'x'))
        self.assertFalse(cleanup._missing_named_actor(ValueError("Failed to look up actor with name 'y'."),'x'))
        self.assertTrue(cleanup._missing_named_actor(ValueError("Failed to look up actor with name 'x'. More detail"),'x'))

    def fake_ray(self, mode):
        def lookup(name, namespace=None):
            if name=='rtp_server_0_0' and mode=='controller_timeout': raise TimeoutError('lookup timeout')
            if name=='rtp_server_0_0' and mode=='controller_permission': raise PermissionError('denied')
            if name=='rtp_server_0_0' and mode in ('live_controller','controller_rpc_died','controller_rpc_timeout'):
                value = (ActorDiedError('controller owner exited') if mode=='controller_rpc_died' else
                         TimeoutError('controller RPC timeout') if mode=='controller_rpc_timeout' else self.canonical)
                return types.SimpleNamespace(shutdown_if_owned=types.SimpleNamespace(remote=lambda *a:value))
            if name=='rtp_native_owner_0_0' and mode in ('live_owner','owner_rpc_timeout','owner_rpc_died','owner_remote_died'):
                value = (TimeoutError('stop RPC timed out') if mode=='owner_rpc_timeout' else ActorDiedError('owner exited') if mode=='owner_rpc_died' else self.canonical['nodes'][0])
                def remote(*arguments):
                    if mode=='owner_remote_died': raise ActorDiedError('owner exited before remote dispatch')
                    return value
                return types.SimpleNamespace(stop_if_owned=types.SimpleNamespace(remote=remote))
            raise ValueError(f'Failed to look up actor with name {name!r}. Recorded lookup failure')
        def get(value, timeout=None):
            if isinstance(value,Exception):raise value
            return value
        return types.SimpleNamespace(is_initialized=lambda:True,get_actor=lookup,get=get)

    def test_live_controller_path_is_unchanged(self):
        with patch.dict(sys.modules,{'ray':self.fake_ray('live_controller')}), patch.object(completed,'verify_completed_multinode_shutdown',return_value=None) as verify:
            result=cleanup._cleanup_multinode(self.identity,self.root)
            self.assertEqual(result['method'],'owned_multinode_controller_rpc');verify.assert_called_once()

    def test_only_missing_actors_allow_verified_fallback(self):
        with patch.dict(sys.modules,{'ray':self.fake_ray('missing')}), patch.object(completed,'verify_completed_multinode_shutdown',side_effect=[None,{'proof':'checked'}]) as verify:
            result=cleanup._cleanup_multinode(self.identity,self.root)
            self.assertEqual(result['method'],'verified_completed_multinode_shutdown');self.assertEqual(verify.call_count,2)

    def test_missing_actors_without_verified_originals_still_fail(self):
        with patch.dict(sys.modules,{'ray':self.fake_ray('missing')}), patch.object(completed,'verify_completed_multinode_shutdown',side_effect=RuntimeError('no canonical')):
            with self.assertRaisesRegex(RuntimeError,'no canonical'):cleanup._cleanup_multinode(self.identity,self.root)

    def test_controller_timeout_does_not_allow_fallback(self):
        with patch.dict(sys.modules,{'ray':self.fake_ray('controller_timeout')}), patch.object(completed,'verify_completed_multinode_shutdown',return_value=None) as verify:
            with self.assertRaisesRegex(RuntimeError,'without an actor-not-found'):cleanup._cleanup_multinode(self.identity,self.root)
            verify.assert_called_once()

    def test_controller_permission_error_does_not_allow_fallback(self):
        with patch.dict(sys.modules,{'ray':self.fake_ray('controller_permission')}), patch.object(completed,'verify_completed_multinode_shutdown',return_value=None) as verify:
            with self.assertRaisesRegex(RuntimeError,'without an actor-not-found'):cleanup._cleanup_multinode(self.identity,self.root)
            verify.assert_called_once()

    def test_live_owner_stop_is_retained_before_verification(self):
        with patch.dict(sys.modules,{'ray':self.fake_ray('live_owner')}), patch.object(completed,'verify_completed_multinode_shutdown',side_effect=[None,{'proof':'checked'}]):
            result=cleanup._cleanup_multinode(self.identity,self.root)
            self.assertEqual(len(result['live_owner_rpc_receipts']),1)

    def test_owner_rpc_timeout_does_not_allow_fallback(self):
        with patch.dict(sys.modules,{'ray':self.fake_ray('owner_rpc_timeout')}), patch.object(completed,'verify_completed_multinode_shutdown',return_value=None) as verify:
            with self.assertRaisesRegex(RuntimeError,'after attempting every owner'):cleanup._cleanup_multinode(self.identity,self.root)
            verify.assert_called_once()


    def test_complete_positive_proof_does_not_look_up_or_stop_actors(self):
        ray=self.fake_ray('live_controller')
        ray.get_actor=unittest.mock.Mock(side_effect=AssertionError('no lookup or stop is needed'))
        with patch.dict(sys.modules,{'ray':ray}), patch.object(completed,'verify_completed_multinode_shutdown',return_value={'positive':'physical proof'}):
            result=cleanup._cleanup_multinode(self.identity,self.root)
            self.assertEqual(result['method'],'verified_completed_multinode_shutdown')
            ray.get_actor.assert_not_called()

    def test_actor_dies_after_controller_lookup_can_only_use_positive_recheck(self):
        with patch.dict(sys.modules,{'ray':self.fake_ray('controller_rpc_died')}), patch.object(completed,'verify_completed_multinode_shutdown',side_effect=[None,{'positive':'physical proof'}]) as verify:
            result=cleanup._cleanup_multinode(self.identity,self.root)
            self.assertEqual(result['method'],'verified_completed_multinode_shutdown')
            self.assertEqual(verify.call_count,2)

    def test_actor_dies_after_owner_lookup_can_only_use_positive_recheck(self):
        with patch.dict(sys.modules,{'ray':self.fake_ray('owner_rpc_died')}), patch.object(completed,'verify_completed_multinode_shutdown',side_effect=[None,{'positive':'physical proof'}]) as verify:
            result=cleanup._cleanup_multinode(self.identity,self.root)
            self.assertEqual(result['method'],'verified_completed_multinode_shutdown')
            self.assertEqual(verify.call_count,2)

    def test_controller_rpc_timeout_is_not_actor_death(self):
        with patch.dict(sys.modules,{'ray':self.fake_ray('controller_rpc_timeout')}), patch.object(completed,'verify_completed_multinode_shutdown',return_value=None) as verify:
            with self.assertRaisesRegex(RuntimeError,'without an actor-not-found/death race'):
                cleanup._cleanup_multinode(self.identity,self.root)
            verify.assert_called_once()

    def test_no_complete_canonical_after_actor_death_is_failure(self):
        with patch.dict(sys.modules,{'ray':self.fake_ray('controller_rpc_died')}), patch.object(completed,'verify_completed_multinode_shutdown',return_value=None):
            with self.assertRaisesRegex(RuntimeError,'no complete canonical'):
                cleanup._cleanup_multinode(self.identity,self.root)

    def test_contradicted_physical_proof_never_reaches_stop_rpc(self):
        ray=self.fake_ray('live_controller')
        ray.get_actor=unittest.mock.Mock(side_effect=AssertionError('identity contradiction must fail closed'))
        with patch.dict(sys.modules,{'ray':ray}), patch.object(completed,'verify_completed_multinode_shutdown',side_effect=RuntimeError('native PID still present')):
            with self.assertRaisesRegex(RuntimeError,'native PID still present'):
                cleanup._cleanup_multinode(self.identity,self.root)
            ray.get_actor.assert_not_called()

    def test_manifest_pid_mismatch_is_rejected(self):
        path=self.root/'boots/boot-000/node-00/rtp_weight_manifest_0/rank-00000.json'
        value=json.loads(path.read_text()); value['pid']+=1; self.write(path,value)
        with self.assertRaisesRegex(RuntimeError,'local native rank manifest differs'):
            self.inspect()


    def test_actor_dies_during_owner_remote_dispatch_can_use_positive_recheck(self):
        with patch.dict(sys.modules,{'ray':self.fake_ray('owner_remote_died')}), patch.object(completed,'verify_completed_multinode_shutdown',side_effect=[None,{'positive':'physical proof'}]) as verify:
            result=cleanup._cleanup_multinode(self.identity,self.root)
            self.assertEqual(result['method'],'verified_completed_multinode_shutdown')
            self.assertEqual(verify.call_count,2)

    def test_unpublished_later_native_boot_is_rejected(self):
        process=copy.deepcopy(self.boot['nodes'][0]['process_identity'])
        process.update(boot_id=1,incarnation='d'*32)
        self.write(self.root/'boots/boot-001/node-00/rtp_process_0.json',process)
        with self.assertRaisesRegex(RuntimeError,'later or different native process incarnation'):
            self.inspect()


    def nonlive_originals(self, status):
        process=self.boot['nodes'][0]['process_identity']
        owned={process['pid']:process['create_time']}
        owned.update({row['pid']:row['create_time'] for row in process['descendants']})
        self.process_patch.side_effect=lambda pid:types.SimpleNamespace(
            create_time=lambda:owned[pid],status=lambda:status)

    def test_nonlive_original_identity_allows_zombie_or_dead_without_context(self):
        for status in (completed.psutil.STATUS_ZOMBIE,completed.psutil.STATUS_DEAD):
            with self.subTest(status=status):
                self.nonlive_originals(status)
                result=self.inspect()
                self.assertTrue(result['proof']['passed'])
                self.assertTrue(all(row['state']=='nonlive' for row in result['proof']['native_identities']))

    def test_nonlive_original_identity_still_rejects_owned_gpu_context(self):
        self.nonlive_originals(completed.psutil.STATUS_ZOMBIE)
        self.inventory['compute_processes'].append(dict(pid=10001,gpu_uuid='GPU-'+f'{0:032x}'))
        with self.assertRaisesRegex(RuntimeError,'still has a GPU context'):
            self.inspect()


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0]]+remaining)
