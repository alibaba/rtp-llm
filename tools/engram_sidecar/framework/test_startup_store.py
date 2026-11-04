#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Real TCPStore startup integration; fake allocations, no CUDA/latency claim.

python -m unittest tools.engram_sidecar.framework.test_startup_store -v
Each rank gets a separate client connection, as in the serving processes.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
import time
import unittest
from unittest.mock import patch

from rtp_llm.models_py.modules.dsv41 import engram_bootstrap as bootstrap
from rtp_llm.models_py.modules.dsv41 import engram_publish as publish
from rtp_llm.models_py.modules.dsv41 import engram_startup as startup
from rtp_llm.models_py.modules.dsv41.test.test_engram_exchange import cohort_env, derived, samples


class RealStoreTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from torch.distributed import TCPStore
        except ImportError:
            raise unittest.SkipTest('requires the serving Torch environment')
        cls.TCPStore = TCPStore

    def setUp(self):
        self.server = self.TCPStore('127.0.0.1', 0, None, True,
                                   timedelta(seconds=20), wait_for_workers=False)

    def client(self):
        return self.TCPStore('127.0.0.1', self.server.port, None, False,
                            timedelta(seconds=20))

    def tearDown(self):
        bootstrap.clear_staged_package()
        del self.server

    def test_disjoint_waits_are_not_collectives_and_keep_per_call_deadline(self):
        a, b = self.client(), self.client()
        started = time.monotonic()
        with ThreadPoolExecutor(2) as pool:
            done = pool.submit(startup._wait, a, ['ready'], time.monotonic() + 2)
            missing = pool.submit(startup._wait, b, ['never'], time.monotonic() + 0.15)
            self.server.set('ready', b'1')
            done.result(timeout=3)
            with self.assertRaisesRegex(startup.ExchangeError, 'TIMEOUT'):
                missing.result(timeout=3)
        self.assertLess(time.monotonic() - started, 3)

    def test_single_gang_uses_existing_store_and_ignores_previous_attempt(self):
        hosts = ['127.0.0.1'] * 2 + ['127.0.0.2'] * 2
        env = cohort_env(2, timeout=5)
        env.pop(startup.COHORT_ENV)
        env.pop(startup.INDEX_ENV)
        env.update(RTP_ENGRAM_LOCAL_DEVICE='0', RTP_ENGRAM_LOCAL_NUMA='0')
        cfg = startup.build_group(env, 4, gang_hosts=hosts)
        _, docs = derived(cohort_env(2), 2)
        for doc in docs:
            doc['generation'] = cfg['group']['generation']
        old = f'engram/{cfg["group"]["generation"]}/attempt/1/gang/package'
        self.server.set(old, b'invalid old package')
        for rank in range(4):
            self.server.add(f'engram/start_count/{rank}', 1)
        original = startup.assemble_package
        published = []

        def allocate(checkpoint, revision, host, generation, shard, count, **kwargs):
            published.append(shard)
            return docs[shard]

        def no_rpc(*args):
            self.fail('single gang must reuse TCPStore')

        clients = [self.client() for _ in range(4)]
        with patch.object(publish, 'publish_shard', allocate), patch.object(
                startup, 'assemble_package', lambda g, d, c: original(g, d, c, samples=samples)):
            with ThreadPoolExecutor(4) as pool:
                pending = [pool.submit(startup.prepare_static_engram, clients[r], rank=r,
                           world_size=4, gang_hosts=hosts, environ=dict(env),
                           leader_exchange=no_rpc) for r in range(4)]
                [p.result(timeout=8) for p in pending]
        self.assertEqual(sorted(published), [0, 1])
        self.assertIsNotNone(bootstrap.staged_package())
        self.assertEqual(self.server.get(old), b'invalid old package')
        prefix = f'engram/{cfg["group"]["generation"]}/attempt/2/gang/'
        self.assertFalse(self.server.check([prefix + 'package']))
        for rank in range(4):
            self.assertEqual(self.server.get(f'engram/start_count/{rank}'), b'2')

    def test_partial_restart_fails_boundedly_without_consuming_old_package(self):
        env = cohort_env(2, timeout=0.15)
        cfg = startup.build_group(env, 2)
        self.server.add('engram/start_count/1', 1)
        self.server.set(f'engram/{cfg["group"]["generation"]}/attempt/1/i0/package', b'stale')
        start = time.monotonic()
        with self.assertRaisesRegex(startup.ExchangeError, 'TIMEOUT'):
            startup.prepare_static_engram(self.client(), rank=1, world_size=2, environ=env)
        self.assertLess(time.monotonic() - start, 3)
        self.assertIsNone(bootstrap.staged_package())


if __name__ == '__main__':
    unittest.main(verbosity=2)
