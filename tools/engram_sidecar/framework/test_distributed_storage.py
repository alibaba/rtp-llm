# SPDX-License-Identifier: Apache-2.0
"""CPU control/mapping contracts; not a CUDA/fabric integration substitute."""
import base64
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from uuid import UUID
from types import SimpleNamespace as NS

from rtp_llm.models_py.modules.dsv41.engram_descriptors import build_manifest, validate_documents
from tools.engram_sidecar.framework.sharded_checkpoint_owner import validate_local_budget
from tools.engram_sidecar.framework.distributed_plan import render_plan
from rtp_llm.models_py.modules.dsv41.engram_shard import plan_shards
from rtp_llm.models_py.modules.dsv41.engram_manifest import ExpectedLayout, parse_manifest
from rtp_llm.models_py.modules.dsv41.engram_storage import FabricEngramStorage
from rtp_llm.models_py.modules.dsv41.test.test_engram_shard_storage import FakeOps

GEN = str(UUID(int=1))
HASH = 'sha256:' + 'a' * 64


def descriptors(p=3, dim=64):
    rows = p * 4096 - 7
    plan = plan_shards(rows, dim, p, granularity=4096)
    result = []
    for shard in range(p):
        begin, end = plan.row_range(shard)
        tensors = {}
        for layer in (1, 14):
            for role, columns, size in (('weight', dim, plan.weight_bytes),
                                         ('scale', dim // 32, plan.scale_bytes)):
                aid = str(UUID(int=100 + shard * 100 + layer * 2 + (role == 'scale')))
                tensors[f'layers.{layer}.engram.embed.{role}'] = dict(
                    shape=[rows, columns], row_begin=begin, row_end=end,
                    logical_bytes=(end - begin) * columns, allocation_bytes=size,
                    allocation_granularity=4096, allocation_id=aid,
                    sha256=HASH, handle=base64.b64encode(aid.encode()).decode())
        result.append(dict(schema_version=2, shard=shard, shards=p, generation=GEN,
            model_revision='revision', hash_abi=HASH, host_id=f'host-{shard}',
            owner_id=f'host-{shard}', owner_incarnation=str(UUID(int=10 + shard)), tensors=tensors))
    return result


def build(docs):
    return build_manifest(docs, len(docs), GEN, 'revision', HASH)


def read_shards(paths, shards, generation):
    from rtp_llm.models_py.modules.dsv41.engram_bootstrap import _json, _private_read

    documents = [_json(_private_read(Path(path), 1 << 20)) for path in paths]
    return validate_documents(documents, shards, generation)


class DistributedDescriptorTest(unittest.TestCase):
    def test_non_256_columns_and_multiple_p(self):
        for p in (2, 3, 5, 18):
            for dim in (32, 64, 256):
                docs = descriptors(p, dim)
                document = build(list(reversed(docs)))
                manifest = parse_manifest(json.dumps(document).encode())
                for tensor in manifest.tensors:
                    self.assertEqual(len(tensor.shards), p)
                    self.assertEqual(tensor.shape[1], dim if tensor.name.endswith('weight') else dim // 32)
                    self.assertEqual([s.owner_id for s in tensor.shards], [f'host-{i}' for i in range(p)])

    def test_invalid_or_mixed_descriptors_fail_before_cuda(self):
        mutations = {
            'revision': lambda d: d[1].update(model_revision='other'),
            'hash': lambda d: d[1].update(hash_abi='sha256:' + 'b' * 64),
            'generation': lambda d: d[1].update(generation=str(UUID(int=3))),
            'count': lambda d: d[1].update(shards=2),
            'index': lambda d: d[1].update(shard=0),
            'boolean': lambda d: d[1].update(shard=True),
            'same_host': lambda d: d[1].update(host_id='host-0', owner_id='host-0'),
            'host_owner': lambda d: d[1].update(owner_id='other'),
            'missing_tensor': lambda d: d[1]['tensors'].pop('layers.1.engram.embed.scale'),
            'allocation': lambda d: d[1]['tensors']['layers.1.engram.embed.weight'].update(
                allocation_id=d[0]['tensors']['layers.1.engram.embed.weight']['allocation_id']),
            'capability': lambda d: d[1]['tensors']['layers.1.engram.embed.weight'].update(
                handle=d[0]['tensors']['layers.1.engram.embed.weight']['handle']),
            'shape': lambda d: d[1]['tensors']['layers.1.engram.embed.scale'].update(shape=[12281, 8]),
            'range': lambda d: d[1]['tensors']['layers.1.engram.embed.scale'].update(row_begin=0),
            'granularity': lambda d: d[1]['tensors']['layers.1.engram.embed.scale'].update(allocation_granularity=8192),
            'invalid_base64': lambda d: d[1]['tensors']['layers.1.engram.embed.scale'].update(handle='%%%'),
        }
        for name, mutate in mutations.items():
            docs = descriptors()
            mutate(docs)
            with self.subTest(case=name), self.assertRaises(ValueError):
                build(docs)

    def test_swapped_indices_cannot_silently_reorder_handle_rows(self):
        docs = descriptors()
        docs[0]['shard'], docs[1]['shard'] = 1, 0
        with self.assertRaisesRegex(ValueError, 'geometry'):
            build(docs)

    def test_private_descriptor_reader(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for i, doc in enumerate(descriptors()):
                path = Path(directory) / f'{i}.json'
                path.write_text(json.dumps(doc)); path.chmod(0o600)
                paths.append(path)
            self.assertEqual(len(read_shards(paths, 3, GEN)), 3)
            paths[0].chmod(0o644)
            with self.assertRaises(ValueError):
                read_shards(paths, 3, GEN)

    def test_local_budget_counts_only_local_padded_allocations(self):
        rows, dim, p = 12281, 64, 3
        layer = NS(tensors=[NS(name='layer.' + role, shape=(rows, width))
                           for role, width in (('weight', dim), ('scale', dim // 32))])
        plan = plan_shards(rows, dim, p, granularity=4096)
        expected = plan.weight_bytes + plan.scale_bytes
        for shard in range(p):
            self.assertEqual(validate_local_budget(NS(layers=[layer]), shard, p, 4096,
                             plan.weight_bytes, expected), expected)
            with self.assertRaisesRegex(ValueError, 'host budget'):
                validate_local_budget(NS(layers=[layer]), shard, p, 4096,
                                      plan.weight_bytes, expected - 1)

    def test_single_and_multiple_instances_import_same_backing(self):
        for p in (2, 3, 5):
            docs = descriptors(p)
            document = build(docs)
            raw = json.dumps(document).encode()
            handles = {e['allocation_id']: base64.b64decode(e['handle'])
                       for d in docs for e in d['tensors'].values()}
            layout = ExpectedLayout(layers=((1, p * 4096 - 7), (14, p * 4096 - 7)),
                                    head_dim=64, model_revision='revision', hash_abi=HASH)
            for instance_count in (1, 3):
                consumers, recorders = [], []
                # Two workers per instance, each with its own CUDA address space.
                for _ in range(instance_count * 2):
                    ops = FakeOps(); recorders.append(ops)
                    storage = FabricEngramStorage.attach(raw, expected_layout=layout,
                        dataset_id=document['dataset_id'], generation=GEN, handles=handles,
                        device=0, native=None, shard_ops=ops)
                    storage.verify(lambda views, verified: True)
                    consumers.append(storage)
                imports = [[c[1] for c in ops.calls if c[0] == 'import'] for ops in recorders]
                self.assertTrue(all(values == imports[0] for values in imports))
                self.assertEqual(len(imports[0]), 4 * p)
                consumers[0].close()
                self.assertTrue(all(c.info()['state'] == 'READY' for c in consumers[1:]))
                for consumer in consumers[1:]: consumer.close()
                for ops in recorders:
                    self.assertEqual(sum(c[0] == 'release' for c in ops.calls), 4 * p)


class LaunchPlanTest(unittest.TestCase):
    def config(self, p=3, instances=2):
        return dict(instance_count=instances, storage_host_count=p, generation=GEN, checkpoint='/model',
            model_revision='revision', publication_root='/run/engram',
            storage_hosts=[dict(host_id=f'h{i}', device=0, numa=0) for i in range(p)],
            instances=[dict(instance_id=f'i{i}', ip=f'127.0.0.{i + 1}',
                            hosts=[f'h{j}' for j in range(p) if j % instances == i])
                       for i in range(instances)],
            exchange=dict(coordinator=dict(host='127.0.0.1', port=19780)))

    def test_p_independent_of_instance_worker_count(self):
        for p in (2, 3, 5):
            for instances in (1, 2):
                plan = render_plan(self.config(p, instances))
                self.assertEqual(plan['instance_count'], instances)
                self.assertEqual(plan['membership'], 'static')
                self.assertTrue(all(c['held_shards'] for c in plan['consumers']))
                self.assertEqual(len(plan['holders']), p)
                self.assertEqual(len(plan['consumers']), instances)
                cohorts = {c['env']['RTP_ENGRAM_COHORT'] for c in plan['consumers']}
                self.assertEqual(len(cohorts), 1)
                self.assertEqual(len(cohorts.pop().split(',')), instances)
                self.assertTrue(all(c['env']['ENGRAM_SHARD_COUNT'] == str(p) for c in plan['consumers']))
                self.assertTrue(all('RTP_ENGRAM_EXTERNAL_BUNDLE' not in c['env'] for c in plan['consumers']))

    def test_reject_missing_or_duplicate_host(self):
        for kind in ('duplicate', 'missing', 'p1', 'invalid_device'):
            c = self.config()
            if kind == 'duplicate': c['storage_hosts'][1]['host_id'] = 'h0'
            if kind == 'missing': c['storage_hosts'].pop()
            if kind == 'p1': c['storage_host_count'] = 1
            if kind == 'invalid_device': c['storage_hosts'][0]['device'] = True
            with self.subTest(kind=kind), self.assertRaises(ValueError): render_plan(c)

    def test_configured_n_must_match_exact_membership(self):
        for n in (0, True, 1, 3, '2'):
            config = self.config()
            config['instance_count'] = n
            with self.subTest(n=n), self.assertRaisesRegex(ValueError, 'instance_count'):
                render_plan(config)

    def test_extra_consumer_only_instance_rejected(self):
        config = self.config()
        config['instance_count'] = 3
        config['instances'].append(dict(instance_id='late-reader', hosts=['h99']))
        with self.assertRaisesRegex(ValueError, 'every static instance'):
            render_plan(config)

    def test_storage_outside_instance_gang_is_rejected(self):
        config = self.config()
        config['storage_hosts'][0]['host_id'] = 'outside'
        with self.assertRaisesRegex(ValueError, 'belong'):
            render_plan(config)

    def test_overlap_requires_explicit_instance_assignment(self):
        config = self.config()
        config['instances'][1]['hosts'].append('h0')
        with self.assertRaisesRegex(ValueError, 'exactly one instance'):
            render_plan(config)
        config['storage_hosts'][0]['instance_id'] = 'i0'
        plan = render_plan(config)
        self.assertEqual(plan['holders'][0]['instance_id'], 'i0')
        config['storage_hosts'][0]['instance_id'] = 'unknown'
        with self.assertRaisesRegex(ValueError, 'must contain'):
            render_plan(config)

    def test_rpc_plan_emits_env_only(self):
        config = self.config(4, 2)
        plan = render_plan(config)
        self.assertNotIn('coordinator_argv', plan)
        self.assertNotIn('startup_configs', plan)
        for index, consumer in enumerate(plan['consumers']):
            env = consumer['env']
            self.assertNotIn('RTP_ENGRAM_EXTERNAL_BUNDLE', env)
            self.assertNotIn('RTP_ENGRAM_STATIC_GROUP_CONFIG', env)
            self.assertEqual(env['RTP_ENGRAM_INSTANCE_INDEX'], str(index))
            self.assertEqual(env['RTP_ENGRAM_COHORT'], '127.0.0.1:19780,127.0.0.2:19780')
            self.assertEqual(env['RTP_ENGRAM_MODEL_REVISION'], 'revision')
            self.assertEqual(env['RTP_ENGRAM_CHECKPOINT'], '/model')
            self.assertEqual(len(env['RTP_ENGRAM_LOCAL_DESCRIPTORS'].split(',')),
                             len(consumer['held_shards']))

    def test_rpc_plan_requires_coordinator(self):
        config = self.config()
        config['exchange'] = {}
        with self.assertRaisesRegex(ValueError, 'coordinator'):
            render_plan(config)
