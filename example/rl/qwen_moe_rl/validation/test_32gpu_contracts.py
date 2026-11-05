"""Synthetic CPU regressions; none of these records are GPU acceptance evidence."""
from argparse import Namespace
from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checkpoint_contract import identity
from q397_evidence_contracts import hf_residency, native_boot, native_shutdown, reference_gpu_membership
from validate_rl_run import (Validator, Check, shard_extent, digest_ids, fp32_bytes,
                             OFFICIAL_OBJECT_MANIFEST_SHA256)
from q397_memory_monitor import Peaks


def uuid(index):
    return f'{index+1:08x}-1111-2222-3333-444444444444'


def boot_fixture():
    incarnation = 'a'*32
    statuses = [dict(tp_rank=i, tp_size=16, model_version='cold-v0', global_step=0, phase='running') for i in range(16)]
    nodes = []
    for node_index in range(4):
        ranks = list(range(node_index*4, node_index*4+4))
        hostname, ip = f'fixture-node-{node_index}', f'192.0.2.{node_index+1}'
        leader = 2000+node_index*100
        pids = [leader+1+local for local in range(4)]
        process = dict(pid=leader, pgid=leader, create_time=100., owner_hostname=hostname,
                       owner_ip=ip, node_index=node_index, owned_tp_ranks=ranks,
                       incarnation=incarnation, boot_id=0,
                       descendants=[dict(pid=pid, create_time=101.) for pid in pids])
        manifests = [dict(tp_rank=rank, tp_size=16, world_rank=rank, dp_size=1, ep_size=1,
                          owner_hostname=hostname, owner_ip=ip, owner_node_index=node_index,
                          incarnation=incarnation, boot_id=0, pid=pids[local], gpu_uuid=uuid(rank),
                          names=['fixture.weight']) for local, rank in enumerate(ranks)]
        inventory = dict(hostname=hostname,
                         devices=[dict(index=local, uuid=uuid(rank)) for local, rank in enumerate(ranks)],
                         processes=[dict(pid=pids[local], uuid=uuid(rank), used_mib=100)
                                    for local, rank in enumerate(ranks)])
        nodes.append(dict(node_index=node_index, boot_id=0, incarnation=incarnation,
                          hostname=hostname, node_ip=ip, local_ranks=ranks,
                          process_identity=process, rank_manifests=manifests,
                          inventory=inventory, statuses=deepcopy(statuses)))
    return dict(passed=True, manifest_count=16, native_incarnation=incarnation,
                native_boot=0, nodes=nodes)


class FullModelShardRegression(unittest.TestCase):
    def test_independent_hf_and_cold_worlds_may_be_disjoint_inside_formal_32(self):
        training={uuid(i) for i in range(32)}
        hf={uuid(i) for i in range(4)}
        cold={uuid(i) for i in range(16,32)}
        self.assertEqual(0,reference_gpu_membership(hf,cold,training)['hf_native_overlap_count'])
        for wrong_hf,wrong_cold in [(hf-{uuid(0)}|{uuid(32)},cold),
                                    (hf,cold-{uuid(16)}|{uuid(32)})]:
            with self.assertRaises(ValueError):
                reference_gpu_membership(wrong_hf,wrong_cold,training)

    def test_all_official_objects_are_statically_pinned(self):
        path=Path(__file__).with_name('official-model-objects.json')
        self.assertEqual(OFFICIAL_OBJECT_MANIFEST_SHA256,sha256(path.read_bytes()).hexdigest())
        record=json.loads(path.read_text())
        self.assertEqual('8472618112abcbd45acbcdc58436aff4233c23f7',record['revision'])
        self.assertEqual(105,len(record['objects']))
        self.assertEqual(94,len([row for row in record['objects'] if row['name'].endswith('.safetensors') and row['lfs_sha256']]))
        self.assertEqual(806819469998,sum(row['bytes'] for row in record['objects']))

    def test_first_axis_slices_cover_parameters_including_empty_tail(self):
        for shape in ([1, 4096], [16], [32], [512, 2048, 4096], [248320, 4096]):
            rows = [shard_extent(shape, rank) for rank in range(32)]
            self.assertEqual(0, rows[0][0])
            self.assertEqual(shape[0], rows[-1][1])
            self.assertEqual([row[1] for row in rows[:-1]], [row[0] for row in rows[1:]])
            self.assertEqual(math.prod(shape), sum(math.prod(row[2]) for row in rows))
        self.assertEqual(31, sum(shard_extent([1, 4096], rank)[0] == shard_extent([1, 4096], rank)[1]
                                 for rank in range(32)))

    def test_true_four_gpu_hf_residency_is_required(self):
        checkpoint = SimpleNamespace(text={f'layer{i}.weight': [16, 16] for i in range(4)}, visual={},
                                     text_numel=1024, visual_numel=0)
        placements = {f'layer{i}.weight': f'cuda:{i}' for i in range(4)}
        devices = [dict(index=i, device=f'cuda:{i}', gpu_uuid=uuid(i), cuda_numel=256,
                        allocated_bytes=512, peak_allocated_bytes=512, peak_reserved_bytes=512)
                   for i in range(4)]
        reference = dict(schema='qwen35-397b-hf-cold-reference-v1', **identity(), dtype='bfloat16', mtp=False,
                         device='layer_sharded_cuda', input_device='cuda:0',
                         model_class='Qwen3_5MoeForConditionalGeneration',
                         parameter_placements=placements,
                         parameter_device_sha256=sha256(json.dumps(placements, sort_keys=True).encode()).hexdigest(),
                         devices=devices, cuda_numel=1024, parameter_dtype_numel={'torch.bfloat16':1024},
                         peak_allocated_bytes=2048)
        self.assertEqual(1024, hf_residency(reference, checkpoint, {uuid(i) for i in range(16)})['cuda_numel'])
        for field, value in [('schema','qwen35-122b-hf-cold-reference-v1'), ('cuda_numel',256),
                             ('devices',devices[:1]), ('model_revision','old-checkpoint')]:
            corrupted = dict(reference, **{field:value})
            with self.assertRaises(ValueError):
                hf_residency(corrupted, checkpoint, {uuid(i) for i in range(16)})


class NativeOwnerRegression(unittest.TestCase):
    def test_native_boot_requires_four_real_node_owner_maps(self):
        boot = boot_fixture()
        self.assertEqual(16, len(native_boot(boot)['manifests']))
        corrupted = deepcopy(boot)
        corrupted['nodes'][0]['rank_manifests'][0]['pid'] = 99999
        with self.assertRaises(ValueError):
            native_boot(corrupted)
        corrupted = deepcopy(boot)
        corrupted['nodes'][1]['inventory']['processes'][0]['uuid'] = uuid(0)
        with self.assertRaises(ValueError):
            native_boot(corrupted)
        corrupted = deepcopy(boot)
        corrupted['nodes'] = corrupted['nodes'][:2]
        with self.assertRaises(ValueError):
            native_boot(corrupted)

    def test_cleanup_rejects_pid_reuse_and_missing_native_rank(self):
        boot = boot_fixture()
        nodes = []
        for node in boot['nodes']:
            p = node['process_identity']
            nodes.append(dict(node_index=node['node_index'], owner_hostname=node['hostname'],
                              owner_ip=node['node_ip'], boot_id=0, pid=p['pid'], pgid=p['pgid'], create_time=p['create_time'],
                              process_tree_terminated=True, owned_gpu_processes_after=[],
                              owned_pids=[p['pid']]+[x['pid'] for x in p['descendants']]))
        shutdown = dict(native_boot=0, native_incarnation=boot['native_incarnation'],
                        process_tree_terminated=True, finished_at_unix=1000., nodes=nodes)
        self.assertEqual(16, native_shutdown(boot, shutdown)['released_ranks'])
        reused = deepcopy(shutdown)
        reused['nodes'][0]['create_time'] += 1
        with self.assertRaises(ValueError):
            native_shutdown(boot, reused)
        missing = deepcopy(shutdown)
        missing['nodes'][1]['owned_pids'].pop()
        with self.assertRaises(ValueError):
            native_shutdown(boot, missing)


class IndependentReplicaRegression(unittest.TestCase):
    def setup_validator(self):
        v = Validator(Namespace(run_root=Path('/tmp/synthetic-unexecuted-q397-contract'),
                                minimum_updates=2, world_size=32, recorded_run_root=None))
        v.replica_syncs = {r:{} for r in range(2)}
        v.generation_receipts = []
        v.episodes = []
        for step in range(2):
            prompt, output, values = [1,step+5], [3], [-.5]
            key = dict(prompt_sha256=digest_ids(prompt), response_sha256=digest_ids(output),
                       raw_logprobs_fp32_sha256=sha256(fp32_bytes(values)).hexdigest())
            v.episodes.append((Path(f'episode{step}.json'),
                dict(episode_id=f'episode{step}', trajectories=[dict(steps=[dict(prompt_ids=prompt,
                     response_ids=output, logprobs=values, weight_version=step)])])))
            for replica in range(2):
                sync = dict(model_version=f'replica{replica}-v{step}', native_boot=0,
                            native_incarnation=('a' if replica == 0 else 'b')*32,
                            full_source_sha256=sha256(str(step).encode()).hexdigest())
                v.replica_syncs[replica][step] = (Path(f'sync{replica}-{step}.json'),sync)
                record = dict(engine='rtp-llm',logprobs_mode='raw_logprobs',replica_rank=replica,
                              global_step=step,model_version=sync['model_version'],native_boot=0,
                              native_incarnation=sync['native_incarnation'],native_request_id=1+replica*2+step,
                              prompt_tokens=len(prompt),response_tokens=1,logprobs=1,**key)
                v.generation_receipts.append((Path(f'gen{replica}.jsonl'),record))
        return v

    def test_same_policy_identical_payload_from_two_replicas_is_not_a_fake_version_conflict(self):
        validator = self.setup_validator()
        result = Check()
        validator.generation_logprobs(result)
        self.assertEqual([],result.errors)
        self.assertEqual([(0,0),(0,1),(1,0),(1,1)],result.details['sampled_replica_steps'])

    def test_uncommitted_replica_or_changed_source_policy_is_rejected(self):
        validator = self.setup_validator()
        validator.generation_receipts[0][1]['model_version'] = 'not-committed'
        result = Check()
        validator.generation_logprobs(result)
        self.assertTrue(result.errors)
        validator = self.setup_validator()
        validator.replica_syncs[1][0][1]['full_source_sha256'] = '0'*64
        result = Check()
        validator.generation_logprobs(result)
        self.assertTrue(result.errors)

    def test_actual_native_rpc_identity_accepts_signed_int63_boundaries(self):
        for request_id in (1, 2**63-1):
            with self.subTest(request_id=request_id):
                validator = self.setup_validator()
                validator.generation_receipts[0][1]['native_request_id'] = request_id
                result = Check()
                validator.generation_logprobs(result)
                self.assertEqual([], result.errors)

    def test_actual_native_rpc_identity_rejects_wrong_type_and_out_of_range(self):
        for request_id in (None, True, False, '1', 'rpc0-0', 1.0, 0, -1, 2**63):
            with self.subTest(request_id=request_id):
                validator = self.setup_validator()
                validator.generation_receipts[0][1]['native_request_id'] = request_id
                result = Check()
                validator.generation_logprobs(result)
                self.assertTrue(any('positive signed int63' in error for error in result.errors))


class MemorySamplingRegression(unittest.TestCase):
    def test_four_device_used_and_reserved_peaks_are_independent(self):
        peaks=Peaks()
        for cycle in range(3):
            rows=[dict(index=i,uuid=uuid(i),query_unix_ns=(cycle+1)*1000000000+i,
                       query_monotonic_ns=(cycle+1)*20000000+i,total_bytes=1000,
                       used_bytes=100+cycle*10+i,reserved_bytes=50-cycle*10+i,free_bytes=800)
                  for i in range(4)]
            peaks.add(rows,(cycle+1)*20000000)
        result=peaks.result()
        self.assertEqual(20,result['maximum_cycle_gap_ms'])
        self.assertEqual(20,result['maximum_per_device_query_gap_ms'])
        self.assertEqual(120,result['per_gpu'][0]['peak_used_bytes'])
        self.assertEqual(50,result['per_gpu'][0]['peak_reserved_bytes'])
        self.assertEqual(30,result['per_gpu'][0]['reserved_at_peak_bytes'])
        with self.assertRaises(RuntimeError):
            peaks.add(rows[:3],80000000)


if __name__ == '__main__':
    unittest.main()
