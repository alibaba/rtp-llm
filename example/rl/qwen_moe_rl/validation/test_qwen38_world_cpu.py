"""CPU controller contracts; Ray/native clients are explicit test doubles.

The process cleanup tests load and exercise the actual existing RTPServer
identity-bound terminator against only their own short-lived local child.
No CUDA, model, real Ray placement or native EP execution is claimed.
"""
import asyncio
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import struct
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import patch
import uuid

HERE = Path(__file__).resolve().parents[1] / "adapter"
sys.path.insert(0, str(HERE))
from qwen38_world_contract import World72Topology, validate_world_manifests, validate_native_library_proof, fixed_layer_source_map
from qwen38_refit_contract import export_descriptor, validate_collective_stage, exact_native_ack, validate_combined_verl_mesh
from qwen38_owned_cleanup import validate_identity as validate_cleanup_identity, validate_shutdown

BASE = Path(os.environ.get('RTP_BASE_ADAPTER', str(HERE)))
NAMES = [f'model.fixture.source_{i:04d}' for i in range(1590)]
NATIVE_SHA256 = 'a' * 64


def fixed_probe_sources(root, *, step=0, version='inc:1', hf_scores=(-1., -2.)):
    (root / f'rtp_sync_rank0_step{step}.json').write_text(json.dumps(dict(
        global_step=step, model_version=version, full_source_sha256='b'*64,
        native_boot=0, native_incarnation='inc', logical_model_replicas=1,
        received_weights=1590, ranks=list(range(72)))))
    training=root/'training'; training.mkdir(exist_ok=True)
    (training / f'hf-probe-step{step}-rank0.json').write_text(json.dumps(dict(
        schema='q38-real-HF-policy-probe-v1', world_size=72, rank=0, global_step=step,
        prompt_ids=[1,2],response_ids=[1,2],raw_log_probs=list(hf_scores),
        raw_fp32_bits_hex=struct.pack(f'<{len(hf_scores)}f',*hf_scores).hex(),
        master_dtype='bfloat16',compute_dtype='bfloat16',score_dtype='float32',
        max_HF_rank_error=0.,all_rank_HF_probe_agreement=True)))


def topology(bound=True):
    return World72Topology(gpu_ids=tuple(str(r % 4) for r in range(72)),
        rank_hosts=tuple(f'192.0.2.{r // 4 + 1}' for r in range(72)), start_port=24000,
        gpu_uuids=tuple(str(uuid.UUID(int=r + 1000)) for r in range(72)) if bound else ())


def manifests(topo, boot=0, incarnation='inc'):
    rows = [dict(**{key: value for key, value in topo.rank(r).items() if key != 'local_rank'},
        owner_node_index=r // 4, pid=10000 + r, gpu_uuid=topo.gpu_uuids[r],
        boot_id=boot, incarnation=incarnation, incarnation_id=incarnation,
        names=NAMES, names_sha256=hashlib.sha256('\n'.join(NAMES).encode()).hexdigest()) for r in range(72)]
    for row in rows:
        row.update(metadata_bootstrap=True, bootstrap_native_library=dict(
            schema='qwen38-actual-native-library-proof-v1', native_initial_pause_version=1,
            native_library_sha256=NATIVE_SHA256, native_library_path='/fixture/libth_transformer.so',
            native_library_inode=1, native_library_device=1, native_library_bytes=1234,
            **{key: row[key] for key in ('pid', 'world_rank', 'world_size', 'tp_size', 'dp_size', 'ep_size')}))
    return rows


class Remote:
    def __init__(self, method): self.method = method
    def remote(self, *args, **kwargs): return self.method(*args, **kwargs)


class Owner:
    def __init__(self, index, topo, root, events, fail_start=False, fail_stop=False):
        self.index, self.topo, self.root, self.events = index, topo, root, events
        self.fail_start, self.fail_stop = fail_start, fail_stop
        self.identity = Remote(self._identity)
        self.bind_world = Remote(self._bind)
        self.start = Remote(self._start)
        self.publish = Remote(self._publish)
        self.stop_for_training = Remote(self._stop)
        self.stop_if_owned = Remote(self._stop_if_owned)
        self.inventory = Remote(self._inventory)
        self.extra_context = False
    async def _identity(self):
        return dict(node_index=self.index, owner_ip=self.topo.node_groups[self.index]['host'],
            owner_hostname=f'cpu-node-{self.index}', ray_node_id=f'node-{self.index}',
            audit_dir=str(self.root.resolve()), pid=20000 + self.index, create_time=10.,
            allocated_gpu_uuids=list(topology().gpu_uuids[self.index * 4:self.index * 4 + 4]))
    async def _bind(self, record):
        self.events.append(('bind', self.index))
        return dict(node_index=self.index, bound=True)
    async def _start(self, boot, incarnation):
        self.events.append(('start', self.index))
        if self.fail_start: raise RuntimeError('injected local startup failure')
        return dict(node_index=self.index, rank_manifests=manifests(topology(), boot, incarnation)[self.index * 4:self.index * 4 + 4])
    async def _publish(self, *args):
        self.events.append(('publish', self.index))
        return dict(node_index=self.index, published=True)
    async def _stop(self):
        self.events.append(('stop', self.index))
        if self.fail_stop: raise RuntimeError('injected cleanup failure')
        return dict(node_index=self.index, owner_ip=self.topo.node_groups[self.index]['host'],
            process_tree_terminated=True, checked_native_ranks=list(range(self.index * 4, self.index * 4 + 4)),
            owned_pids=[10000 + r for r in range(self.index * 4, self.index * 4 + 4)],
            remaining_owned_gpu_contexts=[], owned_gpu_processes_after=[])
    async def _inventory(self):
        return dict(processes=[dict(pid=10000 + self.index * 4)] if self.extra_context else [])
    async def _stop_if_owned(self, pid, created, audit):
        self.events.append(('emergency_stop', self.index))
        return await self._stop()


class Client:
    def __init__(self):
        self.open = False
        self.gate = None
        self.entered = asyncio.Event()
        self.fail = False
        self.expected_step = self.expected_version = None
    def open_admission(self): self.open = True
    def close_admission(self, reason): self.open = False
    async def close(self): self.open = False
    async def score_tokens(self, *args, **kwargs):
        self.entered.set()
        if self.gate is not None: await self.gate.wait()
        if self.fail: raise ValueError('injected probe failure')
        return [-1., -2.]


class Control:
    def __init__(self, topo, events, phase='paused', version='inc:1', step=0):
        self.events = events
        self.replies = [types.SimpleNamespace(**{key: value for key, value in topo.rank(r).items()
                                                  if key not in ('node_index', 'local_rank', 'owner_ip')},
            protocol_version=3, phase=phase, model_version=version, global_step=step,
            update_id='', received_weights=1590, applied_weights=17,
            quiescent=phase == 'paused', kv_cache_ready=True, kv_cache_bytes=1024) for r in range(72)]
    async def status(self): return self.replies
    async def pause(self):
        self.events.append(('pause', 72))
        for row in self.replies: row.phase = 'paused'; row.quiescent = True
        return self.replies
    async def resume(self):
        self.events.append(('resume', 72))
        for row in self.replies: row.phase = 'running'; row.quiescent = False
        return self.replies
    async def restore_cache(self): return self.replies
    async def close(self): pass


@contextmanager
def module(root):
    saved = dict(sys.modules)
    ray = types.ModuleType('ray')
    ray.util = types.SimpleNamespace(get_node_ip_address=lambda: '192.0.2.1')
    ray.get_runtime_context = lambda: types.SimpleNamespace(namespace='explicit-CPU-test', get_node_id=lambda: 'node-0')
    torch = types.ModuleType('torch')
    token_output = type('TokenOutput', (), {})
    sys.modules.update(ray=ray, torch=torch)
    sys.modules['verl.workers.rollout.replica'] = types.SimpleNamespace(TokenOutput=token_output)
    sys.path.insert(0, str(BASE))
    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
        return mod
    with patch.dict(os.environ, RLLM_RTP_ACCEPTANCE_DIR=str(root), RLLM_RTP_DEPLOY_ROOT=str(root),
                    Q38_TRAINING_AUDIT_DIR=str(root/'training')):
        native = load('rtp_native_server', BASE / 'rtp_native_server.py')
        world = load('qwen38_world_server', HERE / 'qwen38_world_server.py')
        world._sources = lambda _path: set(NAMES)
        try:
            yield world, native
        finally:
            sys.path.remove(str(BASE))
            for key in list(sys.modules):
                if key not in saved: del sys.modules[key]
            sys.modules.update(saved)


def controller(mod, root, events, *, bound=True):
    topo = topology(bound)
    owners = [Owner(i, topo, root, events) for i in range(18)]
    obj = mod.World72RTPServer(str(root), topo.to_dict(), owners, [f'node-{i}' for i in range(18)])
    return obj, owners


class ContractTests(unittest.TestCase):
    def test_actual_verl_combines_tp_and_attention_dp_in_infer_tp_72(self):
        for rank in range(72):
            mesh=types.SimpleNamespace(mesh_dim_names=('dp','infer_tp','infer_pp'),
                size=lambda name:{'dp':1,'infer_tp':72,'infer_pp':1}[name],
                get_local_rank=lambda name,r=rank:r if name=='infer_tp' else 0)
            proof=validate_combined_verl_mesh(mesh,rank)
            self.assertEqual((proof['native_tp_rank'],proof['native_dp_rank']),(rank%8,rank//8))
        wrong=types.SimpleNamespace(mesh_dim_names=('dp','infer_tp','infer_pp'),
            size=lambda name:{'dp':9,'infer_tp':8,'infer_pp':1}[name],get_local_rank=lambda name:0)
        with self.assertRaisesRegex(ValueError,'infer_tp=72'):validate_combined_verl_mesh(wrong,0)

    def test_fixed_expert_pool_has_one_logical_policy_and_64_padding_slots(self):
        mapping=fixed_layer_source_map()
        self.assertEqual(len(mapping),92);self.assertTrue(all(row==mapping[0] for row in mapping))
        self.assertEqual(len(mapping[0]),576);self.assertEqual(set(mapping[0]),set(range(512)))
        seen=set();primary=[]
        for rank in range(72):
            before=len(seen);seen.update(mapping[0][rank*8:rank*8+8]);primary.append(len(seen)-before)
        # A padding slot may contain the next rank's logical source. Count the
        # actual route primaries, independently of repeated checkpoint storage.
        self.assertEqual(mapping[0][:64],list(range(64)))
        self.assertEqual(mapping[0][64:72],list(range(64,72)))
        self.assertEqual(mapping[0][-1],0)
        self.assertEqual(sum(7+int(rank<8) for rank in range(72)),512)

    def test_one_world_nine_dp_addresses_and_native_environment(self):
        topo = topology()
        self.assertEqual(World72Topology.from_dict(topo.to_dict()), topo)
        self.assertEqual(len(set(topo.control_addresses)), 72)
        self.assertEqual(topo.generation_addresses, [topo.control_addresses[r] for r in range(0, 72, 8)])
        for node in range(18):
            env = topo.node_environment(node, 'inc')
            self.assertEqual(tuple(env[k] for k in ('WORLD_SIZE', 'WORLD_RANK', 'LOCAL_WORLD_SIZE',
                'TP_SIZE', 'DP_SIZE', 'EP_SIZE', 'FFN_SP_SIZE')), ('72', str(node * 4), '4', '8', '9', '72', '8'))
            self.assertEqual(env['RTP_LLM_RL_START_PAUSED'], '1')
            self.assertEqual(env['RTP_LLM_RL_METADATA_BOOTSTRAP'], '1')
            self.assertEqual(env['RTP_FIXED_PHYSICAL_EP_LAYOUT'], 'balanced_v1')
            self.assertEqual(env['RTP_RL_BORROW_SINGLETON_HF_SOURCES'], '1')
            self.assertEqual(env['RTP_NCCL_EP_REFERENCE_MAX_PADDED_TOKENS'], '73728')
            self.assertEqual(len(env['GANG_CONFIG_STRING'].split(';')), 18)
    def test_unbound_uuid_profile_cannot_allocate_native(self):
        with self.assertRaisesRegex(ValueError, 'physical UUIDs'): topology(False).node_environment(0, 'inc')
    def test_duplicate_uuid_and_reordered_endpoints_fail(self):
        record = topology().to_dict(); record['gpu_uuids'][71] = record['gpu_uuids'][0]
        with self.assertRaises(ValueError): World72Topology.from_dict(record)
        record = topology().to_dict(); record['control_addresses'].reverse()
        with self.assertRaises(ValueError): World72Topology.from_dict(record)
    def test_every_native_world_manifest_has_distinct_rank_uuid_and_incarnation(self):
        topo = topology(); rows = manifests(topo)
        self.assertEqual(len(validate_world_manifests(rows, topo, NAMES, boot=0, incarnation='inc')), 72)
        for field, value in (('world_rank', 0), ('tp_rank', 0), ('dp_rank', 0), ('ep_rank', 0),
                             ('gpu_uuid', rows[0]['gpu_uuid']), ('incarnation_id', 'foreign')):
            with self.subTest(field=field):
                copy = [dict(row) for row in rows]; copy[71][field] = value
                with self.assertRaises(ValueError): validate_world_manifests(copy, topo, NAMES, boot=0, incarnation='inc')
    def test_mtp_or_missing_source_is_never_accepted(self):
        rows = manifests(topology()); rows[71]['names'] = NAMES[:-1] + ['mtp.fixture']
        with self.assertRaisesRegex(ValueError, '1590'): validate_world_manifests(rows, topology(), NAMES, boot=0, incarnation='inc')

    def test_actual_bootstrap_library_proof_binds_fresh_digest_marker_and_rank_pid(self):
        row = manifests(topology())[0]
        proof = row['bootstrap_native_library']
        self.assertEqual(validate_native_library_proof(proof, row, NATIVE_SHA256), proof)
        for field, value in (('pid', row['pid']+1), ('world_rank', 1),
                             ('native_initial_pause_version', 0), ('native_library_sha256', 'b'*64)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_native_library_proof(dict(proof, **{field: value}), row, NATIVE_SHA256)
        rows = manifests(topology()); rows[71]['metadata_bootstrap'] = False
        with self.assertRaisesRegex(ValueError, 'metadata bootstrap'):
            validate_world_manifests(rows, topology(), NAMES, boot=0, incarnation='inc')

    def test_canonical_export_order_and_native_ack_are_whole_world_contracts(self):
        weights = types.SimpleNamespace(order=NAMES, acknowledge=lambda *a, **k: None,
            close=lambda: None, confirm_native_cleanup=lambda proof: None, __iter__=lambda: iter(()))
        descriptor = export_descriptor(weights, NAMES, step=2)
        records = [dict(error=None, **descriptor) for _ in range(72)]
        self.assertEqual(len(validate_collective_stage(records, 'fixture', identity_fields=('order_sha256',))), 72)
        records[71]['order_sha256'] = 'foreign'
        with self.assertRaisesRegex(RuntimeError, 'order_sha256 differs'):
            validate_collective_stage(records, 'fixture', identity_fields=('order_sha256',))
        records[71] = dict(error='single-rank iterator failed')
        with self.assertRaisesRegex(RuntimeError, 'rank-local failure'):
            validate_collective_stage(records, 'fixture')
        weights.order = NAMES + [NAMES[0]]
        with self.assertRaisesRegex(ValueError, 'exactly once'): export_descriptor(weights, NAMES)
        self.assertEqual(exact_native_ack(list(range(72))), list(range(72)))
        for ranks in (list(range(71)), list(range(71))+[0], list(range(72))+[0]):
            with self.assertRaisesRegex(RuntimeError, 'exactly one ACK'): exact_native_ack(ranks)


class ControllerTests(unittest.IsolatedAsyncioTestCase):
    async def test_launch_server_only_prepares_cpu_identity_without_start(self):
        with tempfile.TemporaryDirectory() as temp, module(Path(temp)) as (mod, native):
            events = []; obj, owners = controller(mod, Path(temp), events, bound=False)
            address = await obj.launch_server()
            self.assertEqual(address, topology().generation_addresses[0])
            self.assertEqual(len(obj.topology.gpu_uuids), 72)
            self.assertEqual(len(obj.controller_identity['node_owners']), 18)
            self.assertEqual([kind for kind, i in events], ['bind'] * 18)
            self.assertFalse(obj._started); self.assertFalse(obj._owners_active); self.assertIsNone(obj.process)
            initial_sleep=await obj.sleep()
            self.assertTrue(initial_sleep['native_never_started']);self.assertEqual(initial_sleep['native_boot'],-1)
            self.assertEqual(initial_sleep['checked_native_ranks'],[])
            self.assertNotIn('start',[kind for kind,_ in events])

    async def test_cleanup_discovers_18_owner_identity_and_never_accepts_legacy_four_owner_shape(self):
        with tempfile.TemporaryDirectory() as temp,module(Path(temp)) as (mod,native):
            root=Path(temp);obj,owners=controller(mod,root,[])
            identity=await obj.prepare()
            validate_cleanup_identity(identity,root.resolve(),root.resolve(),'explicit-CPU-test')
            with self.assertRaisesRegex(RuntimeError,'18 owners'):
                validate_cleanup_identity(dict(identity,node_owner_count=4),root.resolve(),root.resolve(),'explicit-CPU-test')
            with self.assertRaisesRegex(RuntimeError,'omits'):
                validate_shutdown(dict(world_size=72,all_owned_native_processes_exited=True,
                    process_tree_terminated=True,remaining_owned_gpu_contexts=[],checked_native_ranks=list(range(71)),nodes=[{}]*18))

    async def test_partial_start_failure_awaits_all_eighteen_owner_cleanups(self):
        with tempfile.TemporaryDirectory() as temp, module(Path(temp)) as (mod, native):
            root = Path(temp); events = []; obj, owners = controller(mod, root, events)
            owners[7].fail_start = True
            with self.assertRaisesRegex(ValueError, 'startup failed'): await obj.ensure_native_running(0, None)
            self.assertEqual(sorted(i for kind, i in events if kind == 'stop'), list(range(18)))
            self.assertFalse(obj._started); self.assertFalse(obj._owners_active); self.assertFalse(obj._generation_ready)

    async def test_initial_running_native_is_rejected_and_every_owner_cleaned(self):
        with tempfile.TemporaryDirectory() as temp, module(Path(temp)) as (mod, native):
            root = Path(temp); events = []; obj, owners = controller(mod, root, events)
            async def control(_topo, **kwargs): return Control(_topo, events, phase='running', version=obj.incarnation + ':0')
            mod._control = control
            with self.assertRaisesRegex(ValueError, 'actually be paused'): await obj.ensure_native_running(0, None)
            self.assertEqual(len([i for kind, i in events if kind == 'stop']), 18)
            self.assertFalse(obj._generation_ready)

    async def test_nine_probes_must_all_finish_before_any_generation_admission(self):
        with tempfile.TemporaryDirectory() as temp, module(Path(temp)) as (mod, native):
            root = Path(temp); events = []; obj, owners = controller(mod, root, events)
            obj._started = obj._owners_active = True; obj.boot = 0; obj.incarnation = 'inc'
            obj.clients = [Client() for _ in range(9)]; obj.clients[8].gate = asyncio.Event()
            obj.tokenizer = types.SimpleNamespace(encode=lambda *args, **kwargs: [1, 2])
            fixed_probe_sources(root)
            async def control(_topo, **kwargs): return Control(_topo, events)
            mod._control = control
            pending = asyncio.create_task(obj.wake_and_probe(0, 'inc:1'))
            await asyncio.wait_for(obj.clients[8].entered.wait(), 2)
            self.assertTrue(all(client.entered.is_set() for client in obj.clients))
            self.assertFalse(pending.done()); self.assertFalse(obj._generation_ready)
            with self.assertRaisesRegex(ValueError, 'admission is closed'): await obj._route()
            obj.clients[8].gate.set(); proof = await pending
            self.assertEqual(len(proof['probes']), 9); self.assertTrue(obj._generation_ready)
            routes = []
            for _ in range(9):
                dp, *_ = await obj._route(); routes.append(dp); await obj._route_done()
            self.assertEqual(routes, list(range(9)))

    async def test_failed_probe_awaits_all_dp_responses_and_poison_blocks_reprobe(self):
        with tempfile.TemporaryDirectory() as temp, module(Path(temp)) as (mod, native):
            root = Path(temp); events = []; obj, owners = controller(mod, root, events)
            obj._started = obj._owners_active = True; obj.boot = 0; obj.incarnation = 'inc'
            obj.clients = [Client() for _ in range(9)]; obj.clients[8].fail = True
            obj.tokenizer = types.SimpleNamespace(encode=lambda *args, **kwargs: [1, 2])
            fixed_probe_sources(root)
            async def control(_topo, **kwargs): return Control(_topo, events)
            mod._control = control
            with self.assertRaisesRegex(ValueError, 'nine actual DP probes'): await obj.wake_and_probe(0, 'inc:1')
            self.assertTrue(all(client.entered.is_set() for client in obj.clients))
            self.assertTrue(all(not client.open for client in obj.clients))
            self.assertFalse(obj._generation_ready); self.assertTrue(obj._incarnation_failed)
            with self.assertRaisesRegex(ValueError, 'fully refitted'): await obj.wake_and_probe(0, 'inc:1')

    async def test_actual_hf_mismatch_keeps_all_nine_dp_admission_closed(self):
        with tempfile.TemporaryDirectory() as temp, module(Path(temp)) as (mod, native):
            root=Path(temp); events=[]; obj, owners=controller(mod,root,events)
            obj._started=obj._owners_active=True; obj.boot=0; obj.incarnation='inc'
            obj.clients=[Client() for _ in range(9)]
            obj.tokenizer=types.SimpleNamespace(encode=lambda *args,**kwargs:[1,2])
            fixed_probe_sources(root,hf_scores=(-1.2,-2.))
            async def control(_topo,**kwargs):return Control(_topo,events)
            mod._control=control
            with self.assertRaisesRegex(ValueError,'BF16 HF and nine native DP'):
                await obj.wake_and_probe(0,'inc:1')
            self.assertFalse(obj._generation_ready);self.assertTrue(obj._incarnation_failed)
            self.assertTrue(all(not client.open for client in obj.clients))

    async def test_missing_actual_hf_reference_prevents_native_resume(self):
        with tempfile.TemporaryDirectory() as temp, module(Path(temp)) as (mod, native):
            root=Path(temp);events=[];obj,owners=controller(mod,root,events)
            obj._started=obj._owners_active=True;obj.boot=0;obj.incarnation='inc'
            obj.clients=[Client() for _ in range(9)]
            fixed_probe_sources(root)
            (root/'training/hf-probe-step0-rank0.json').unlink()
            async def control(_topo,**kwargs):return Control(_topo,events)
            mod._control=control
            with self.assertRaises(FileNotFoundError):await obj.wake_and_probe(0,'inc:1')
            self.assertEqual(events,[]);self.assertFalse(obj._generation_ready)

    async def test_final_gate_reprobes_actual_policy2_and_refuses_policy1(self):
        with tempfile.TemporaryDirectory() as temp,module(Path(temp)) as (mod,native):
            root=Path(temp);events=[];obj,owners=controller(mod,root,events)
            obj._started=obj._owners_active=True;obj.boot=0;obj.incarnation='inc'
            obj._committed_step=1;obj._committed_version='inc:2'
            with self.assertRaisesRegex(ValueError,'exactly two optimizer'):
                await obj.verify_final_policy(2)
            obj._committed_step=2;obj._committed_version='inc:3'
            obj.clients=[Client() for _ in range(9)]
            obj.tokenizer=types.SimpleNamespace(encode=lambda *args,**kwargs:[1,2])
            fixed_probe_sources(root,step=2,version='inc:3')
            async def control(_topo,**kwargs):return Control(_topo,events,step=2,version='inc:3')
            mod._control=control
            result=await obj.verify_final_policy(2)
            self.assertTrue(result['final_native_GPU_reprobe']);self.assertEqual(result['global_step'],2)
            self.assertTrue(all(client.entered.is_set() for client in obj.clients))
            self.assertTrue((root/'rtp_final_policy2_actual_reprobe.json').is_file())

    async def test_suspend_drains_before_global_pause_then_eighteen_true_cleanup_receipts(self):
        with tempfile.TemporaryDirectory() as temp, module(Path(temp)) as (mod, native):
            root = Path(temp); events = []; obj, owners = controller(mod, root, events)
            obj._started = obj._owners_active = obj._generation_ready = True
            obj._committed_step=0;obj._committed_version='inc:1';obj.incarnation='inc'
            obj.clients = [Client() for _ in range(9)]; obj._inflight = 1
            async def control(_topo, **kwargs): return Control(_topo, events, phase='running')
            mod._control = control
            pending = asyncio.create_task(obj.suspend_for_training())
            await asyncio.sleep(.01)
            self.assertFalse(pending.done()); self.assertFalse(obj._generation_ready)
            self.assertEqual(events, [])
            await obj._route_done(); proof = await pending
            self.assertEqual(events[0], ('pause', 72))
            self.assertEqual(sorted(i for kind, i in events if kind == 'stop'), list(range(18)))
            self.assertEqual(proof['checked_native_ranks'], list(range(72)))
            self.assertTrue(proof['all_owned_native_processes_exited'])
            self.assertTrue(proof['complete_policy_verified_before_training_handoff'])
            self.assertTrue(proof['training_admission_safe'])

    async def test_metadata_only_world_never_grants_training_safe_even_after_true72_pause_and_exit(self):
        with tempfile.TemporaryDirectory() as temp,module(Path(temp)) as (mod,native):
            root=Path(temp);events=[];obj,owners=controller(mod,root,events)
            obj._started=obj._owners_active=True;obj.boot=0;obj.incarnation='uncommitted'
            async def control(_topo,**kwargs):
                result=Control(_topo,events,phase='paused',version='uncommitted:0')
                for row in result.replies:row.received_weights=0;row.applied_weights=0
                return result
            mod._control=control
            proof=await obj.suspend_for_training()
            self.assertTrue(proof['all_world_pause_acknowledged'])
            self.assertTrue(proof['all_owned_native_processes_exited'])
            self.assertEqual(proof['checked_native_ranks'],list(range(72)))
            self.assertFalse(proof['complete_policy_verified_before_training_handoff'])
            self.assertFalse(proof['training_admission_safe'])
            self.assertEqual(proof['remaining_owned_gpu_contexts'], [])
            self.assertEqual(await obj.suspend_for_training(), proof)

    async def test_one_owner_cleanup_failure_never_claims_all_contexts_gone(self):
        with tempfile.TemporaryDirectory() as temp, module(Path(temp)) as (mod, native):
            root = Path(temp); events = []; obj, owners = controller(mod, root, events)
            obj._started = obj._owners_active = True; owners[7].fail_stop = True
            async def control(_topo, **kwargs): return Control(_topo, events, phase='running')
            mod._control = control
            with self.assertRaisesRegex(ValueError, 'not all 18'): await obj.suspend_for_training()
            self.assertEqual(len([i for kind, i in events if kind == 'stop']), 18)
            self.assertIsNone(obj._last_suspend)

    async def test_delayed_nvml_context_after_cleanup_is_a_fatal_gate(self):
        with tempfile.TemporaryDirectory() as temp, module(Path(temp)) as (mod, native):
            root = Path(temp); events = []; obj, owners = controller(mod, root, events)
            obj._started = obj._owners_active = True; owners[17].extra_context = True
            async def control(_topo, **kwargs): return Control(_topo, events, phase='running')
            mod._control = control
            with self.assertRaisesRegex(ValueError, 'delayed owned native context'): await obj.suspend_for_training()
            self.assertIsNone(obj._last_suspend)

    async def test_pause_failure_still_returns_truthful_cleanup_only_and_never_training_safe(self):
        with tempfile.TemporaryDirectory() as temp, module(Path(temp)) as (mod, native):
            root = Path(temp); events = []; obj, owners = controller(mod, root, events)
            obj._started = obj._owners_active = True
            async def control(_topo, **kwargs): raise RuntimeError('injected missing world ACK')
            mod._control = control
            proof = await obj.suspend_for_training()
            self.assertTrue(proof['all_owned_native_processes_exited'])
            self.assertEqual(proof['checked_native_ranks'], list(range(72)))
            self.assertFalse(proof['training_admission_safe'])
            self.assertFalse(proof['all_world_pause_acknowledged'])
            self.assertTrue(proof['errors'])
            again = await obj.suspend_for_training()
            self.assertFalse(again['training_admission_safe'])

    async def test_verified_emergency_stop_reaches_all_owners_while_lifecycle_is_locked(self):
        with tempfile.TemporaryDirectory() as temp, module(Path(temp)) as (mod, native):
            root = Path(temp); events = []; obj, owners = controller(mod, root, events)
            await obj.prepare()
            events.clear(); obj._started = obj._owners_active = True
            async def control(_topo, **kwargs): raise RuntimeError('already stopped by emergency')
            mod._control = control
            await obj._lifecycle_lock.acquire()
            identity = obj.controller_identity
            pending = asyncio.create_task(obj.shutdown_if_owned(identity['pid'], identity['create_time'], identity['audit_dir']))
            await asyncio.sleep(.01)
            self.assertEqual(sorted(i for kind, i in events if kind == 'emergency_stop'), list(range(18)))
            self.assertFalse(pending.done())
            obj._lifecycle_lock.release()
            proof = await pending
            self.assertTrue(proof['all_owned_native_processes_exited'])
            self.assertFalse(proof['training_admission_safe'])

    async def test_permanently_stopped_owner_cannot_execute_a_queued_start(self):
        with tempfile.TemporaryDirectory() as temp, module(Path(temp)) as (mod, native):
            obj = mod.WorldNodeOwner.__new__(mod.WorldNodeOwner)
            obj.poisoned = True
            with patch.object(mod, 'gpu_inventory', side_effect=AssertionError('must not inspect/allocate GPUs')):
                with self.assertRaisesRegex(ValueError, 'permanently stopped'): await obj.start(0, 'inc')


class ActualOwnedProcessCleanupTests(unittest.TestCase):
    def test_existing_native_terminator_reaps_only_its_verified_child(self):
        import psutil
        with tempfile.TemporaryDirectory() as temp, module(Path(temp)) as (mod, native):
            process = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(60)'], start_new_session=True)
            obj = mod.WorldNodeOwner.__new__(mod.WorldNodeOwner)
            obj._process_lock = threading.RLock()
            created = psutil.Process(process.pid).create_time()
            obj.process = process; obj.replica_rank = 0; obj._cleanup_receipt = None
            obj._owned_processes = {process.pid: created}
            obj.process_identity = dict(pid=process.pid, create_time=created, pgid=os.getpgid(process.pid))
            try:
                obj._terminate()
                self.assertEqual(process.returncode, -15)
                self.assertEqual(obj._cleanup_receipt['owned_pids'], [process.pid])
                self.assertFalse(psutil.pid_exists(process.pid))
                original = dict(obj._cleanup_receipt)
                obj._terminate()
                self.assertEqual(obj._cleanup_receipt, original)
            finally:
                if process.poll() is None: process.terminate(); process.wait(timeout=5)

    def test_invalid_creation_time_never_signals_the_child(self):
        import psutil
        with tempfile.TemporaryDirectory() as temp, module(Path(temp)) as (mod, native):
            process = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(60)'], start_new_session=True)
            obj = mod.WorldNodeOwner.__new__(mod.WorldNodeOwner)
            obj._process_lock = threading.RLock()
            created = psutil.Process(process.pid).create_time()
            obj.process = process; obj.replica_rank = 0; obj._cleanup_receipt = None
            obj._owned_processes = {process.pid: created}
            obj.process_identity = dict(pid=process.pid, create_time=created - 1., pgid=os.getpgid(process.pid))
            try:
                with self.assertRaisesRegex(RuntimeError, 'PID was reused'): obj._terminate()
                self.assertIsNone(process.poll())
                self.assertIsNone(obj._cleanup_receipt)
            finally:
                if process.poll() is None: process.terminate(); process.wait(timeout=5)


class BackendRefitTests(unittest.IsolatedAsyncioTestCase):
    """Explicit CPU doubles exercise adapter failure sequencing, never CUDA IPC."""

    @contextmanager
    def backend(self, root):
        with module(root):
            torch=sys.modules['torch'];torch.no_grad=lambda:lambda function:function
            torch.bfloat16='bf16';torch.float32='fp32';torch.device=lambda kind,index:f'{kind}:{index}'
            torch.cuda=types.SimpleNamespace(current_device=lambda:0)
            sys.modules['torch.distributed']=types.ModuleType('torch.distributed')
            sys.modules['verl.workers.rollout.base']=types.SimpleNamespace(BaseRollout=type('BaseRollout',(),{}))
            sys.modules['verl.workers.rollout.replica'].RolloutReplica=type('RolloutReplica',(),{})
            sys.modules['rtp_llm.utils.cuda_ipc']=types.SimpleNamespace(
                CudaIpcV2=types.SimpleNamespace(export_tensor=lambda tensor:dict(explicit_cpu_fixture=True)))
            spec=importlib.util.spec_from_file_location('qwen38_verl_backend',HERE/'qwen38_verl_backend.py')
            mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
            obj=mod.World72ServerAdapter.__new__(mod.World72ServerAdapter)
            obj.required_sources=set(NAMES);obj.is_leader=True;obj.rank=0
            obj._last_sync_step=obj._last_sync_version=obj._cleanup_callback=obj._retained_source=None
            obj._broadcast=lambda value:value
            obj._gather=lambda value:[dict(value) for _ in range(72)]
            obj._binding=lambda:dict(world_rank=0,explicit_cpu_fixture=True)
            mod._source=lambda:None
            mod._hash_source=lambda tensor,digest,name:('a'*64,4)
            yield mod,obj

    async def test_replica_sleep_initially_skips_unbound_gpu_callback_then_awaits_all72_handoffs(self):
        with tempfile.TemporaryDirectory() as temp,self.backend(Path(temp)) as (mod,obj):
            replica=mod.World72Replica.__new__(mod.World72Replica)
            replica._world72_bound_topology=topology();events=[]
            initial=dict(native_never_started=True,native_boot=-1,checked_native_ranks=[])
            current=[initial]
            async def sleep():return current[0]
            replica._server_handle=types.SimpleNamespace(sleep=Remote(sleep))
            def make_call(rank):
                async def call(function,proof,expected):
                    events.append(expected)
                    return dict(world_rank=rank,gpu_uuid=topology().gpu_uuids[rank],phase='rollout',
                        adapter_callback_cleared=True,native_contexts_released=proof.get('training_admission_safe'))
                return call
            replica.workers=[types.SimpleNamespace(__ray_call__=Remote(make_call(rank))) for rank in range(72)]
            self.assertEqual(await replica.sleep(),initial);self.assertEqual(events,[])
            proof=dict(world_size=72,native_boot=0,native_incarnation='inc',all_owned_native_processes_exited=True,
                process_tree_terminated=True,remaining_owned_gpu_contexts=[],checked_native_ranks=list(range(72)),
                nodes=[{}]*18,training_admission_safe=True,all_world_pause_acknowledged=True)
            current[0]=proof
            result=await replica.sleep()
            self.assertEqual(sorted(events),list(range(72)))
            self.assertEqual(result['actor_cleanup_handoff_world_size'],72)
            current[0]=dict(proof,training_admission_safe=False,all_world_pause_acknowledged=False)
            with self.assertRaisesRegex(RuntimeError,'cleanup-only proof'):await replica.sleep()

    async def test_worker_handoff_frees_exporter_and_preserves_cleanup_only_training_block(self):
        for safe in (True,False):
            with self.subTest(training_safe=safe),tempfile.TemporaryDirectory() as temp,self.backend(Path(temp)) as (mod,adapter):
                rank=8;events=[]
                mod.dist.is_initialized=lambda:True;mod.dist.get_world_size=lambda:72;mod.dist.get_rank=lambda:rank
                mod.torch.cuda.ipc_collect=lambda:events.append('ipc_collect')
                mod.torch.cuda.empty_cache=lambda:events.append('empty_cache')
                mod.torch.cuda.get_device_properties=lambda device:types.SimpleNamespace(uuid=topology().gpu_uuids[rank])
                engine=types.SimpleNamespace(_q38_phases=types.SimpleNamespace(phase='rollout'),_q38_exporter=object())
                def confirm(proof):
                    events.append('actual_cleanup_callback');engine._q38_exporter=None
                    engine._q38_native_contexts_released=proof['training_admission_safe']
                    engine._q38_last_refit_source_release=dict(all_source_views_released=True,backing_released=True)
                engine.q38_confirm_native_cleanup=confirm;adapter._cleanup_callback=confirm
                worker=types.SimpleNamespace(worker_dict={'actor_rollout':types.SimpleNamespace(
                    _is_actor=True,actor=types.SimpleNamespace(engine=engine),rollout=adapter)})
                proof=dict(world_size=72,native_boot=0,native_incarnation='inc',all_owned_native_processes_exited=True,
                    process_tree_terminated=True,remaining_owned_gpu_contexts=[],checked_native_ranks=list(range(72)),
                    nodes=[{}]*18,training_admission_safe=safe,all_world_pause_acknowledged=safe)
                result=mod._confirm_worker_native_cleanup(worker,proof,rank)
                self.assertIsNone(engine._q38_exporter);self.assertIsNone(adapter._cleanup_callback)
                self.assertEqual(engine._q38_phases.phase,'rollout')
                self.assertIs(result['native_contexts_released'],safe)
                self.assertEqual(events,['actual_cleanup_callback','ipc_collect','empty_cache'])

    async def test_single_rank_order_failure_is_collective_before_native_start(self):
        with tempfile.TemporaryDirectory() as temp,self.backend(Path(temp)) as (mod,obj):
            events=[]
            async def call(*args):events.append(args);raise AssertionError('native startup must not be reached')
            obj._leader_call=call
            def gather(value):
                rows=[dict(value) for _ in range(72)];rows[71]['order_sha256']='foreign';return rows
            obj._gather=gather
            weights=types.SimpleNamespace(order=NAMES,acknowledge=lambda *a,**k:None,
                close=lambda:None,confirm_native_cleanup=lambda proof:None,__iter__=lambda:iter(()))
            with self.assertRaisesRegex(RuntimeError,'order_sha256 differs'):
                await obj.update_weights(weights,global_steps=0)
            self.assertEqual(events,[])

    async def test_incomplete_native_ack_tears_down_before_releasing_the_source(self):
        with tempfile.TemporaryDirectory() as temp,self.backend(Path(temp)) as (mod,obj):
            events=[]
            class Weights:
                order=NAMES
                def __iter__(self):
                    for name in NAMES:
                        events.append(('borrow',name));yield name,types.SimpleNamespace(dtype='bf16',device='cuda:0',shape=(2,))
                def acknowledge(self,*args,**kwargs):
                    events.append(('ack',args));raise AssertionError('partial ACK must not release lease')
                def close(self):raise AssertionError('failed export must not claim normal source close')
                def confirm_native_cleanup(self,proof):
                    if proof.get('all_owned_native_processes_exited') is not True:raise AssertionError('cleanup proof required')
                    events.append(('free_after_cleanup',None))
            async def call(method,*args):
                events.append((method,args))
                if method=='ensure_native_running':return dict(native_boot=0,native_incarnation='inc')
                if method=='suspend_for_training':return dict(all_owned_native_processes_exited=True)
                raise AssertionError(method)
            obj._leader_call=call
            class PartialControl:
                async def status(self):return [types.SimpleNamespace(phase='paused',model_version='inc:0')]
                async def begin_update(self,**kwargs):events.append(('begin',None))
                async def push_weight(self,name,metadata):
                    events.append(('push',name));return [types.SimpleNamespace(world_rank=rank) for rank in range(71)]
                async def close(self):events.append(('control_close',None))
            async def control():return PartialControl()
            mod._control=control
            with self.assertRaisesRegex(RuntimeError,'exactly one ACK'):
                await obj.update_weights(Weights(),global_steps=0)
            kinds=[kind for kind,_ in events]
            self.assertEqual(kinds.count('borrow'),1);self.assertNotIn('ack',kinds)
            self.assertLess(kinds.index('suspend_for_training'),kinds.index('free_after_cleanup'))
            self.assertEqual(kinds[-1],'control_close')


if __name__ == '__main__':
    unittest.main()
