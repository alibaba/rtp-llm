# SPDX-License-Identifier: Apache-2.0
import base64
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest import mock
from uuid import UUID

from rtp_llm.models_py.modules.dsv41 import engram_bootstrap as b
from rtp_llm.models_py.modules.dsv41 import engram_exchange as e
from rtp_llm.models_py.modules.dsv41 import engram_startup as s
from rtp_llm.models_py.modules.dsv41.engram_shard import plan_shards


def fixture(n=2, p=3, ranks=2):
    group = dict(generation=str(UUID(int=123)), model_revision='test-revision', members=[])
    docs = []
    plan = plan_shards(p * 4096 - 7, 64, p, granularity=4096)
    for instance in range(n):
        shards = [dict(index=i, host_id=f'host-{i}', publisher_rank=pos)
                  for pos, i in enumerate(range(instance, p, n))]
        group['members'].append(dict(instance_id=f'i{instance}', world_size=ranks, shards=shards))
    for index in range(p):
        begin, end = plan.row_range(index)
        tensors = {}
        for role, width, size in [('weight', 64, plan.weight_bytes), ('scale', 2, plan.scale_bytes)]:
            aid = str(UUID(int=1000 + 2 * index + (role == 'scale')))
            tensors['layers.1.engram.embed.' + role] = dict(shape=[plan.total_rows, width],
                row_begin=begin, row_end=end, allocation_bytes=size, allocation_granularity=4096,
                logical_bytes=(end - begin) * width, allocation_id=aid, sha256='sha256:' + 'a' * 64,
                handle=base64.b64encode(aid.encode()).decode())
        docs.append(dict(schema_version=2, shard=index, shards=p, generation=group['generation'],
            model_revision=group['model_revision'], hash_abi='sha256:' + 'b' * 64, host_id=f'host-{index}',
            owner_id=f'host-{index}', owner_incarnation=str(UUID(int=200 + index)), tensors=tensors))
    return group, docs


def samples(_, manifest):
    return {t.name: dict(row_ids=sorted({0, t.shape[0] // 2, t.shape[0] - 1} |
                                      {r for part in t.shards for r in (part.row_begin, part.row_end - 1)}),
                         sha256='sha256:' + 'c' * 64) for t in manifest.tensors}


def package(group, docs):
    return s.assemble_package(group, docs, None, samples=samples)


def cohort_env(p, instances=2, index=0, timeout=3):
    return {
        s.COHORT_ENV: ','.join(f'127.0.0.{i + 1}:{29500 + i}' for i in range(instances)),
        s.INDEX_ENV: str(index),
        s.REVISION_ENV: 'test-revision',
        s.GENERATION_ENV: str(UUID(int=123)),
        s.COUNT_ENV: str(p),
        s.TIMEOUT_ENV: str(timeout),
        s.CHECKPOINT_ENV: '/tmp/checkpoint',
    }


def derived(env, world_size):
    """Descriptors whose host identity matches the group derived from env."""
    config = s.build_group(env, world_size)
    group = config['group']
    p = sum(len(m['shards']) for m in group['members'])
    plan = plan_shards(p * 4096 - 7, 64, p, granularity=4096)
    hosts = {sh['index']: sh['host_id'] for m in group['members'] for sh in m['shards']}
    docs = []
    for index in range(p):
        begin, end = plan.row_range(index)
        tensors = {}
        for role, width, size in [('weight', 64, plan.weight_bytes), ('scale', 2, plan.scale_bytes)]:
            aid = str(UUID(int=1000 + 2 * index + (role == 'scale')))
            tensors['layers.1.engram.embed.' + role] = dict(
                shape=[plan.total_rows, width], row_begin=begin, row_end=end,
                allocation_bytes=size, allocation_granularity=4096,
                logical_bytes=(end - begin) * width, allocation_id=aid,
                sha256='sha256:' + 'a' * 64, handle=base64.b64encode(aid.encode()).decode())
        docs.append(dict(schema_version=2, shard=index, shards=p,
            generation=group['generation'], model_revision=group['model_revision'],
            hash_abi='sha256:' + 'b' * 64, host_id=hosts[index], owner_id=hosts[index],
            owner_incarnation=str(UUID(int=200 + index)), tensors=tensors))
    return config, docs


class GroupTest(unittest.TestCase):
    def test_supported_static_shapes(self):
        for n, p, ranks in [(1, 2, 2), (2, 3, 2), (3, 5, 2), (5, 5, 1)]:
            group, docs = fixture(n, p, ranks)
            self.assertEqual(len(e.validate_group(group)), 64)
            s.verify_package(group, package(group, docs))

    def test_invalid_memberships_and_host_assignments(self):
        mutations = [lambda g: g['members'][1].update(instance_id='i0'),
                     lambda g: g['members'][1].update(shards=[]),
                     lambda g: g['members'][0]['shards'][0].update(publisher_rank=True),
                     lambda g: g['members'][0]['shards'][0].update(index=1),
                     lambda g: g['members'][0]['shards'][0].update(host_id='host-1'),
                     lambda g: g['members'][0].update(world_size=0)]
        for mutate in mutations:
            group, _ = fixture(); mutate(group)
            with self.assertRaises((ValueError, e.ExchangeError)): e.validate_group(group)

    def test_cohort_waits_for_all_instances_and_rejects_changes(self):
        group, docs = fixture()
        cohort = e.Cohort(group, lambda d: package(group, d), 2)
        with ThreadPoolExecutor(2) as pool:
            first = pool.submit(cohort.submit, 'i0', cohort.group_digest, [docs[0], docs[2]])
            time.sleep(0.03)
            self.assertFalse(first.done())
            second = pool.submit(cohort.submit, 'i1', cohort.group_digest, [docs[1]])
            result = first.result(); self.assertEqual(result, second.result())
        self.assertEqual(cohort.submit('i0', cohort.group_digest, [docs[0], docs[2]]), result)
        changed = deepcopy(docs[0]); changed['owner_incarnation'] = str(UUID(int=999))
        with self.assertRaisesRegex(e.ExchangeError, 'REPUBLICATION'):
            cohort.submit('i0', cohort.group_digest, [changed, docs[2]])
        with self.assertRaisesRegex(e.ExchangeError, 'GROUP_MISMATCH'):
            cohort.submit('i1', 'wrong', [docs[1]])
        with self.assertRaisesRegex(e.ExchangeError, 'UNKNOWN'):
            cohort.submit('outsider', cohort.group_digest, [])
        cohort.ack('i0', e.digest(result)); cohort.ack('i1', e.digest(result))
        cohort.wait_acknowledged()

    def test_missing_member_and_failed_builder_do_not_hang(self):
        group, docs = fixture()
        cohort = e.Cohort(group, lambda d: {}, 0.03)
        with self.assertRaisesRegex(e.ExchangeError, 'TIMEOUT'):
            cohort.submit('i0', cohort.group_digest, [docs[0], docs[2]])
        def fail(_): raise ValueError('private handle details')
        cohort = e.Cohort(group, fail, 2)
        with ThreadPoolExecutor(2) as pool:
            pending = pool.submit(cohort.submit, 'i0', cohort.group_digest, [docs[0], docs[2]])
            with self.assertRaisesRegex(e.ExchangeError, '^ASSEMBLY_FAILED$'):
                cohort.submit('i1', cohort.group_digest, [docs[1]])
            with self.assertRaisesRegex(e.ExchangeError, '^ASSEMBLY_FAILED$'): pending.result()

    def test_slow_builder_and_missing_ack_are_bounded(self):
        group, docs = fixture(1, 2, 2)
        release = threading.Event()
        def slow(_):
            release.wait(2)
            return {}
        cohort = e.Cohort(group, slow, 0.03)
        try:
            with self.assertRaisesRegex(e.ExchangeError, 'TIMEOUT'):
                cohort.submit('i0', cohort.group_digest, docs)
        finally:
            release.set()
        cohort = e.Cohort(group, lambda d: package(group, d), 0.05)
        payload = cohort.submit('i0', cohort.group_digest, docs)
        with self.assertRaisesRegex(e.ExchangeError, 'ACK_MISMATCH'):
            cohort.ack('i0', 'wrong-payload')
        with self.assertRaisesRegex(e.ExchangeError, 'TIMEOUT'):
            cohort.wait_acknowledged()

    def test_bounded_frames_and_invalid_generation(self):
        with self.assertRaisesRegex(e.ExchangeError, 'MESSAGE_TOO_LARGE'):
            e.decode(b'x' * (e.MAX_MESSAGE + 1))
        with self.assertRaisesRegex(e.ExchangeError, 'MESSAGE_TOO_LARGE'):
            e.encode({'value': 'x' * e.MAX_MESSAGE})
        with self.assertRaises(ValueError):
            e.decode(b'{"instance_id":"a","instance_id":"b"}')
        for value in (None, 1, True, 'not-a-uuid'):
            group, _ = fixture(); group['generation'] = value
            with self.assertRaisesRegex(e.ExchangeError, 'INVALID_GENERATION'):
                e.validate_group(group)

    def test_memory_package_and_full_identity_checks(self):
        env = cohort_env(2)
        config, docs = derived(env, world_size=2)
        payload = package(config['group'], docs)
        memory = s.to_memory_package(config['group'], payload)
        self.assertEqual(set(memory), {'raw', 'dataset_id', 'generation',
                                       'model_revision', 'capabilities', 'samples'})
        self.assertIsInstance(memory['raw'], bytes)
        self.assertTrue(all(isinstance(v, bytes) and v for v in memory['capabilities'].values()))
        changed = deepcopy(payload); changed['capabilities'].pop(next(iter(changed['capabilities'])))
        with self.assertRaisesRegex(e.ExchangeError, 'CAPABILITIES'): s.verify_package(config['group'], changed)
        changed_group = deepcopy(config['group']); changed_group['generation'] = str(UUID(int=999))
        with self.assertRaisesRegex(e.ExchangeError, 'IDENTITY'): s.verify_package(changed_group, payload)


class MemoryStore:
    """Shared test Store with real blocking semantics and bounded waits."""
    def __init__(self): self.data = {}; self.condition = threading.Condition()
    def set(self, key, value):
        with self.condition:
            self.data[key] = value; self.condition.notify_all()
    def get(self, key):
        with self.condition: return self.data[key]
    def wait(self, keys, timeout):
        deadline = time.monotonic() + timeout.total_seconds()
        with self.condition:
            while not all(k in self.data for k in keys):
                remaining = deadline - time.monotonic()
                if remaining <= 0: raise RuntimeError('store timeout')
                self.condition.wait(remaining)
    def delete_key(self, key):
        with self.condition: return self.data.pop(key, None) is not None


class IntraInstanceTest(unittest.TestCase):
    def tearDown(self):
        b.clear_staged_package()

    def _descriptor_env(self, env, docs, directory):
        root = Path(directory)
        paths = []
        for doc in docs:
            path = root / str(doc['shard'])
            path.write_bytes(e.encode(doc))
            path.chmod(0o600)
            paths.append(str(path))
        env[s.DESCRIPTORS_ENV] = ','.join(paths)
        return env

    def test_only_rank_zero_exchanges_and_all_ranks_receive_identical_backing(self):
        env = cohort_env(2)
        config, docs = derived(env, world_size=4)
        store = MemoryStore(); calls = []
        def exchange(cfg, local, deadline):
            calls.append(local)
            return package(cfg['group'], docs)
        with tempfile.TemporaryDirectory() as directory:
            self._descriptor_env(env, docs[:1], directory)
            envs = [dict(env) for _ in range(4)]
            with ThreadPoolExecutor(4) as pool:
                futures = [pool.submit(s.prepare_static_engram, store, rank=i, world_size=4,
                                       environ=envs[i], leader_exchange=exchange) for i in range(4)]
                results = [f.result() for f in futures]
            self.assertEqual(len(calls), 1)
            self.assertTrue(all(r is None for r in results))
            self.assertEqual(store.data, {})
            self.assertTrue(all(item['ENGRAM_SHARD_COUNT'] == '2' for item in envs))
            staged = b.staged_package()
            self.assertIsNotNone(staged)
            self.assertEqual(staged['generation'], config['group']['generation'])
            # Serving must not materialize any bundle files.
            self.assertEqual(list(Path(directory).glob('**/*.json')), [])

    def test_bad_p_fails_before_any_store_side_effect(self):
        store = MemoryStore()
        env = cohort_env(2)
        env[s.COUNT_ENV] = '1'
        with self.assertRaisesRegex(ValueError, 'STORAGE_HOST_COUNT'):
            s.prepare_static_engram(store, rank=0, world_size=2, environ=env)
        self.assertEqual(store.data, {})

    def test_inconsistent_worker_group_never_publishes_package(self):
        env = cohort_env(2, timeout=0.2)
        config, docs = derived(env, world_size=2)
        store = MemoryStore()
        prefix = 'engram/' + config['group']['generation'] + '/i0/'
        store.set(prefix + 'config/1', b'wrong-group')
        with tempfile.TemporaryDirectory() as directory:
            self._descriptor_env(env, docs[:1], directory)
            with self.assertRaisesRegex(e.ExchangeError, 'INSTANCE_GROUP_CONFIG_MISMATCH'):
                s.prepare_static_engram(store, rank=0, world_size=2, environ=env)
        self.assertNotIn(prefix + 'package', store.data)

    def test_self_published_shards_describe_their_own_host(self):
        from rtp_llm.models_py.modules.dsv41 import engram_publish
        from unittest.mock import patch

        env = cohort_env(2)
        config, docs = derived(env, world_size=2)
        store = MemoryStore()
        published = []

        def publish(checkpoint, revision, host_id, generation, shard, shards, **kwargs):
            self.assertEqual(checkpoint, env[s.CHECKPOINT_ENV])
            self.assertEqual(revision, config['group']['model_revision'])
            self.assertEqual(generation, config['group']['generation'])
            self.assertEqual(shards, 2)
            self.assertEqual(host_id, '127.0.0.1')  # the shard's own host, not the member's
            self.assertIsNotNone(kwargs['deadline'])
            self.assertEqual(kwargs['max_allocation_bytes'], 128 << 30)
            self.assertEqual(kwargs['budget_bytes'], 256 << 30)
            published.append((host_id, shard))
            return docs[shard]

        def exchange(cfg, local, deadline):
            return package(cfg['group'], docs)

        with patch.object(engram_publish, 'publish_shard', publish), \
                self.assertLogs('rtp_llm.models_py.modules.dsv41.engram_startup', 'WARNING'):
            with ThreadPoolExecutor(2) as pool:
                results = [pool.submit(s.prepare_static_engram, store, rank=i, world_size=2,
                                       environ=dict(env), leader_exchange=exchange)
                           for i in range(2)]
                [result.result() for result in results]
        self.assertEqual(published, [('127.0.0.1', 0)])
        staged = b.staged_package()
        self.assertIsNotNone(staged)
        self.assertEqual(staged['generation'], config['group']['generation'])
        self.assertEqual(store.data, {})  # descriptors went through Store, not files

    def test_unconfigured_models_do_nothing(self):
        self.assertIsNone(s.prepare_static_engram(None, rank=0, world_size=8, environ={}))


class MutualTLSTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tools.engram_sidecar.framework.make_exchange_test_certs import create
        cls.temp = tempfile.TemporaryDirectory()
        cls.tls = create(Path(cls.temp.name) / 'pki', ['i0', 'i1', 'i2'])
    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def test_two_remote_instances_reuse_exactly_two_connections(self):
        group, docs = fixture(3, 3, 1)
        cohort = e.Cohort(group, lambda d: package(group, d), 10)
        server = e.ExchangeServer(('127.0.0.1', 0), cohort, e.tls_context(self.tls['i0'], True), self.tls['i0']['instance_cert_sha256'])
        thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': 0.02}, daemon=True); thread.start()
        try:
            with ThreadPoolExecutor(3) as pool:
                futures = [pool.submit(e.remote_exchange, server.server_address, e.tls_context(self.tls[f'i{i}']),
                    f'i{i}', cohort.group_digest, [docs[i]], time.monotonic() + 8,
                    lambda p: s.verify_package(group, p), self.tls['i0']['instance_cert_sha256']['i0']) for i in (1, 2)]
                payload = cohort.submit('i0', cohort.group_digest, [docs[0]])
                cohort.ack('i0', e.digest(payload)); cohort.wait_acknowledged()
                results = [f.result() for f in futures]
            self.assertEqual(results[0], results[1])
            self.assertEqual(server.connections, 2)  # exchange + ACK on each connection
        finally:
            server.shutdown(); server.server_close(); thread.join()

    def test_valid_ca_but_wrong_coordinator_certificate_rejected(self):
        group, docs = fixture()
        cohort = e.Cohort(group, lambda d: {}, 3)
        server = e.ExchangeServer(('127.0.0.1', 0), cohort, e.tls_context(self.tls['i0'], True), self.tls['i0']['instance_cert_sha256'])
        thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': 0.02}, daemon=True); thread.start()
        try:
            with self.assertRaisesRegex(e.ExchangeError, 'COORDINATOR_CERTIFICATE_MISMATCH'):
                e.remote_exchange(server.server_address, e.tls_context(self.tls['i1']), 'i1', cohort.group_digest,
                    [docs[1]], time.monotonic() + 2, lambda p: None, self.tls['i0']['instance_cert_sha256']['i2'])
            self.assertEqual(cohort.documents, {})
        finally:
            server.shutdown(); server.server_close(); thread.join()

    def test_certificate_cannot_claim_another_instance(self):
        group, docs = fixture()
        cohort = e.Cohort(group, lambda d: {}, 3)
        server = e.ExchangeServer(('127.0.0.1', 0), cohort, e.tls_context(self.tls['i0'], True), self.tls['i0']['instance_cert_sha256'])
        thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': 0.02}, daemon=True); thread.start()
        try:
            with self.assertRaisesRegex(e.ExchangeError, 'REMOTE_EXCHANGE_FAILED'):
                e.remote_exchange(server.server_address, e.tls_context(self.tls['i2']), 'i1', cohort.group_digest,
                                  [docs[1]], time.monotonic() + 2, lambda p: None)
            self.assertEqual(cohort.documents, {})
        finally:
            server.shutdown(); server.server_close(); thread.join()


class RankSpaceTest(unittest.TestCase):
    """A member is either a whole gang or one host of a gang shared by members."""

    def test_shared_gang_members_are_member_local(self):
        env = cohort_env(p=2, instances=2, index=1)
        env[s.RANKS_ENV] = '4'
        group = s.build_group(env, 8)['group']
        self.assertEqual([m['world_size'] for m in group['members']], [4, 4])
        self.assertEqual([sh['publisher_rank'] for m in group['members'] for sh in m['shards']],
                         [0, 0])

    def test_whole_gang_members_keep_global_ranks(self):
        env = cohort_env(p=2, instances=2, index=1)
        group = s.build_group(env, 4)['group']
        self.assertEqual([m['world_size'] for m in group['members']], [4, 4])
        self.assertEqual([sh['publisher_rank'] for m in group['members'] for sh in m['shards']],
                         [0, 0])

    def test_ranks_per_instance_must_describe_the_gang(self):
        env = cohort_env(p=2, instances=2, index=1)
        env[s.RANKS_ENV] = '3'
        with self.assertRaises(e.ExchangeError):
            s.build_group(env, 8)


class DeploymentIdentityTest(unittest.TestCase):
    def AssertUuid(self, value):
        self.assertEqual(str(UUID(value)), value)

    def test_physical_p_matches_explicit_hosts(self):
        for p in (2, 3, 5, 18):
            env = cohort_env(p, instances=p)
            group = s.build_group(env, 4)['group']
            self.assertEqual([m['shards'][0]['host_id'] for m in group['members']],
                             [f'127.0.0.{i+1}' for i in range(p)])
        with self.assertRaisesRegex(e.ExchangeError, 'MUST_MATCH_COHORT'):
            s.build_group(cohort_env(4, instances=2), 8)

    def test_bare_hosts_take_the_shared_port(self):
        explicit = s.build_group(cohort_env(2), 4)['group']
        env = cohort_env(2)
        env[s.COHORT_ENV] = '127.0.0.1,127.0.0.2'
        env[s.PORT_ENV] = '29500'
        self.assertEqual(s.build_group(env, 4)['group'], explicit)
        del env[s.PORT_ENV]
        defaulted = s.build_group(env, 4)['group']
        self.assertEqual(defaulted['members'][0]['shards'][0]['host_id'], '127.0.0.1')
        self.assertEqual(s.build_group(dict(env, **{s.PORT_ENV: '29500'}), 4)['group'], explicit)

    def test_instance_index_is_discovered_from_the_local_address(self):
        env = cohort_env(2)
        del env[s.INDEX_ENV]
        self.assertEqual(s.build_group(env, 4)['index'], 0)
        with mock.patch.object(s, '_local_addresses', lambda: {'127.0.0.2'}):
            self.assertEqual(s.build_group(env, 4)['index'], 1)
        with mock.patch.object(s, '_local_addresses', lambda: {'10.9.9.9'}):
            with self.assertRaisesRegex(e.ExchangeError, 'INSTANCE_INDEX_REQUIRED'):
                s.build_group(env, 4)
        with mock.patch.object(s, '_local_addresses', lambda: {'127.0.0.1', '127.0.0.2'}):
            with self.assertRaisesRegex(e.ExchangeError, 'AMBIGUOUS_LOCAL_ADDRESS'):
                s.build_group(env, 4)

    def test_generation_defaults_and_duplicate_hosts(self):
        env = cohort_env(2)
        del env[s.GENERATION_ENV]
        derived = s.build_group(env, 4)['group']['generation']
        other = cohort_env(2)
        del other[s.GENERATION_ENV]
        self.assertEqual(derived, s.build_group(other, 4)['group']['generation'])
        self.AssertUuid(derived)
        # The address list is the topology: a different list derives a different id.
        other[s.COHORT_ENV] = '127.0.0.1:10001,127.0.0.2:10002'
        self.assertNotEqual(derived, s.build_group(other, 4)['group']['generation'])
        # An explicit value still wins, and must be a canonical UUID.
        self.assertEqual(s.build_group(cohort_env(2), 4)['group']['generation'], str(UUID(int=123)))
        env = cohort_env(2)
        env[s.GENERATION_ENV] = 'not-a-uuid'
        with self.assertRaisesRegex(e.ExchangeError, 'INVALID_GENERATION'):
            s.build_group(env, 4)
        env = cohort_env(2)
        first = s.build_group(env, 4)['group']
        env[s.GENERATION_ENV] = str(UUID(int=456))
        self.assertNotEqual(e.validate_group(first), e.validate_group(s.build_group(env, 4)['group']))
        env[s.COHORT_ENV] = '127.0.0.1:10001,127.0.0.1:10002'
        with self.assertRaisesRegex(e.ExchangeError, 'DISTINCT_STORAGE_HOST'):
            s.build_group(env, 4)

    def test_wait_uses_bounded_operation_not_blocking_get(self):
        class Store:
            def get(self, key):
                raise AssertionError('unbounded get must not implement wait')
            def wait(self, keys, timeout):
                self.keys, self.timeout = keys, timeout.total_seconds()
                raise RuntimeError('timeout')
        store = Store()
        with self.assertRaisesRegex(e.ExchangeError, 'WAIT_FAILED_OR_TIMEOUT'):
            s._wait(store, ['a', 'b'], time.monotonic() + 0.1)
        self.assertEqual(store.keys, ['a', 'b'])
        self.assertTrue(0 < store.timeout <= 0.1)


class TopologySourceTest(unittest.TestCase):
    """The address list can come from the gang, the platform directory or env."""

    def test_one_gang_derives_its_own_parts(self):
        store = MemoryStore()
        for rank, host in enumerate(['10.0.0.1', '10.0.0.1', '10.0.0.2', '10.0.0.2']):
            store.set(s.TOPOLOGY_PREFIX + 'host/' + str(rank), host.encode())
        with mock.patch.object(s, '_local_addresses', lambda: {'10.0.0.1'}):
            members, per_instance = s._gang_cohort({s.COUNT_ENV: '2'}, store, 0, 4)
        self.assertEqual(members, [('10.0.0.1', s.DEFAULT_COHORT_PORT),
                                   ('10.0.0.2', s.DEFAULT_COHORT_PORT)])
        self.assertEqual(per_instance, 2)
        self.assertEqual(store.get(s.TOPOLOGY_PREFIX + 'host/3'), b'10.0.0.2')

    def test_a_part_is_a_run_of_consecutive_ranks(self):
        # Two ranks per part, four parts: no count beyond N is needed.
        store = MemoryStore()
        hosts = ['10.0.0.1', '10.0.0.1', '10.0.0.2', '10.0.0.2',
                 '10.0.0.3', '10.0.0.3', '10.0.0.4', '10.0.0.4']
        for rank, host in enumerate(hosts):
            store.set(s.TOPOLOGY_PREFIX + 'host/' + str(rank), host.encode())
        with mock.patch.object(s, '_local_addresses', lambda: {'10.0.0.1'}):
            members, per_instance = s._gang_cohort({s.COUNT_ENV: '4'}, store, 0, 8)
        self.assertEqual([host for host, _ in members],
                         ['10.0.0.1', '10.0.0.2', '10.0.0.3', '10.0.0.4'])
        self.assertEqual(per_instance, 2)

    def test_ranks_inside_one_part_must_agree(self):
        store = MemoryStore()
        for rank, host in enumerate(['10.0.0.1', '10.0.0.1', '10.0.0.2', '10.0.0.2']):
            store.set(s.TOPOLOGY_PREFIX + 'host/' + str(rank), host.encode())
        with mock.patch.object(s, '_local_addresses', lambda: {'10.0.0.7'}):
            # The local rank publishes itself into its own block and breaks it.
            with self.assertRaisesRegex(e.ExchangeError, 'UNEVEN_PART_SIZE'):
                s._gang_cohort({s.COUNT_ENV: '2'}, store, 1, 4)

    def test_the_declared_count_must_match_the_gang(self):
        store = MemoryStore()
        for rank, host in enumerate(['10.0.0.1', '10.0.0.1', '10.0.0.2', '10.0.0.2']):
            store.set(s.TOPOLOGY_PREFIX + 'host/' + str(rank), host.encode())
        with mock.patch.object(s, '_local_addresses', lambda: {'10.0.0.1'}):
            with self.assertRaisesRegex(e.ExchangeError, 'MUST_MATCH_COHORT'):
                s._gang_cohort({s.COUNT_ENV: '3'}, store, 0, 4)
            with self.assertRaisesRegex(e.ExchangeError, 'RANKS_PER_INSTANCE_MISMATCH'):
                s._gang_cohort({s.RANKS_ENV: '4'}, store, 0, 4)

    def test_a_single_host_gang_asks_for_the_list(self):
        # Prefill is one gang per host: its four ranks cannot see the other hosts.
        store = MemoryStore()
        for rank in range(4):
            store.set(s.TOPOLOGY_PREFIX + 'host/' + str(rank), b'10.0.0.1')
        with mock.patch.object(s, '_local_addresses', lambda: {'10.0.0.1'}):
            with self.assertRaisesRegex(e.ExchangeError, 'COHORT_REQUIRED'):
                s._gang_cohort({s.RANKS_ENV: '4'}, store, 0, 4)
        self.assertEqual(s._gang_cohort({}, MemoryStore(), 0, 16), (None, None))
        with self.assertRaisesRegex(e.ExchangeError, 'INSTANCE_STORE_REQUIRED'):
            s._gang_cohort({s.COUNT_ENV: '4'}, None, 0, 16)

    def test_platform_directory_supplies_the_list(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'hosts').write_text('# deployment topology\n10.0.0.1\n\n10.0.0.2\n')
            self.assertEqual(s._ganginfo_cohort({s.GANGINFO_ENV: directory}),
                             [('10.0.0.1', s.DEFAULT_COHORT_PORT),
                              ('10.0.0.2', s.DEFAULT_COHORT_PORT)])
            (root / 'hosts').unlink()
            (root / 'part-1').write_text('10.0.0.4\n')
            (root / 'part-0').write_text('10.0.0.3\n')
            (root / 'unrelated').write_text('ignored\n')
            self.assertEqual([host for host, _ in s._ganginfo_cohort({s.GANGINFO_ENV: directory})],
                             ['10.0.0.3', '10.0.0.4'])
            (root / 'part-1').write_text('10.0.0.4:29500\n')
            self.assertEqual(s._ganginfo_cohort({s.GANGINFO_ENV: directory})[1], ('10.0.0.4', 29500))
            self.assertIsNone(s._ganginfo_cohort({s.GANGINFO_ENV: '/nonexistent/ganginfo'}))

    def test_explicit_environment_wins_over_the_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'hosts').write_text('10.0.0.1\n10.0.0.2\n')
            env = cohort_env(2)
            env[s.COHORT_ENV] = '127.0.0.1,127.0.0.2'
            env[s.GANGINFO_ENV] = directory
            group = s.build_group(env, 4)['group']
            self.assertEqual([m['shards'][0]['host_id'] for m in group['members']],
                             ['127.0.0.1', '127.0.0.2'])


class RevisionTest(unittest.TestCase):
    def test_revision_is_derived_from_the_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            index = Path(directory) / 'model.safetensors.index.json'
            index.write_text('{"weight_map": {"a": "a.safetensors"}}')
            env = cohort_env(2)
            del env[s.REVISION_ENV]
            env[s.CHECKPOINT_ENV] = directory
            first = s.build_group(env, 4)['group']['model_revision']
            self.assertTrue(first.startswith('ckpt@'))
            # Identical weights give identical identity, wherever they are mounted.
            other = cohort_env(2)
            del other[s.REVISION_ENV]
            other[s.CHECKPOINT_ENV] = directory
            self.assertEqual(first, s.build_group(other, 4)['group']['model_revision'])
            index.write_text('{"weight_map": {"a": "b.safetensors"}}')
            self.assertNotEqual(first, s.build_group(env, 4)['group']['model_revision'])
            # An explicit revision still wins, and a checkpoint without either
            # index is refused rather than given an accidental identity.
            explicit = dict(env, **{s.REVISION_ENV: 'release-1019'})
            self.assertEqual(s.build_group(explicit, 4)['group']['model_revision'], 'release-1019')
            empty = Path(directory) / 'empty'
            empty.mkdir()
            with self.assertRaisesRegex(e.ExchangeError, 'MODEL_REVISION_REQUIRED'):
                s.build_group(dict(env, **{s.CHECKPOINT_ENV: str(empty)}), 4)


if __name__ == '__main__': unittest.main(verbosity=2)
