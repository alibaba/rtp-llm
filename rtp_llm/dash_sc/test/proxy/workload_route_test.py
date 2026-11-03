"""Backend workload selection, freshness, bounded probes and dispatch affinity."""
import asyncio
import unittest
from unittest.mock import patch

from rtp_llm.dash_sc.proxy.service_route import BackendAddr, IpPortListServiceDiscovery
from rtp_llm.dash_sc.proxy.workload_route import WorkloadRouter, create_workload_router_from_env


def snapshot(**kwargs):
    value = dict(schema_version=1, ready=True, replicas=1, age_ms=0, ingress_requests=0,
                 remaining_prefill_positions=0, waiting_prefill_positions=0,
                 unknown_demand_requests=0, decode_requests=0, kv_growth_pages=0,
                 kv_free_pages=100, kv_used_pages=0, waiting_admission=0)
    value.update(kwargs)
    return value


class WorkloadRouteTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.discovery = IpPortListServiceDiscovery('127.0.0.1:8088;127.0.0.2:8088')
        self.a, self.b = self.discovery.candidates()
        self.values = {self.a: snapshot(), self.b: snapshot()}
        self.calls = []
        self.now = 10.0

        async def fetch(addr):
            self.calls.append(addr)
            return self.values[addr]

        self.router = WorkloadRouter(fetch=fetch, clock=lambda: self.now)
        self.addAsyncCleanup(self.router.close)

    async def test_smaller_prefill_backlog_wins_at_same_request_count(self):
        self.values[self.a] = snapshot(ingress_requests=2, remaining_prefill_positions=800000)
        self.values[self.b] = snapshot(ingress_requests=2, remaining_prefill_positions=100000)
        lease = await self.router.acquire(self.discovery)
        self.assertEqual(lease.addr, self.b)

    async def test_kv_growth_and_unknown_prompts_are_pressure(self):
        self.values[self.a] = snapshot(remaining_prefill_positions=1000, kv_growth_pages=99)
        self.values[self.b] = snapshot(remaining_prefill_positions=2000)
        lease = await self.router.acquire(self.discovery)
        self.assertEqual(lease.addr, self.b)
        self.router.release(lease)
        self.now += 2
        self.values[self.a] = snapshot(unknown_demand_requests=1)
        self.assertEqual((await self.router.acquire(self.discovery)).addr, self.b)

    async def test_concurrent_dispatch_penalty_and_refresh_coalescing(self):
        leases = await asyncio.gather(*(self.router.acquire(self.discovery) for _ in range(20)))
        self.assertEqual(sum(lease.addr == self.a for lease in leases), 10)
        self.assertEqual(sum(lease.addr == self.b for lease in leases), 10)
        self.assertEqual(len(self.calls), 2)
        for lease in leases:
            self.router.release(lease)
        self.assertEqual(self.router._pending, {})

    async def test_release_on_failure_and_grace_expiry(self):
        lease = await self.router.acquire(self.discovery)
        other = self.b if lease.addr == self.a else self.a
        self.assertEqual((await self.router.acquire(self.discovery)).addr, other)
        self.router.release(lease)
        self.assertEqual((await self.router.acquire(self.discovery)).addr, lease.addr)
        self.now += 4
        await self.router.acquire(self.discovery)
        self.assertEqual(len(self.router._pending), 1)

    async def test_stale_missing_and_malformed_metrics_fail_closed(self):
        for bad in [None, snapshot(age_ms=3000), snapshot(ready=False),
                    snapshot(kv_free_pages=-1), snapshot(kv_used_pages=True),
                    snapshot(waiting_prefill_positions=10)]:
            self.values[self.a] = self.values[self.b] = bad
            self.now += 2
            self.assertIsNone(await self.router.acquire(self.discovery))

    async def test_cached_age_expires_before_refresh(self):
        self.values[self.a] = self.values[self.b] = snapshot(age_ms=2800)
        await self.router.acquire(self.discovery)
        self.now += .25
        self.assertIsNone(await self.router.acquire(self.discovery))
        self.assertEqual(len(self.calls), 2)

    async def test_membership_change_does_not_reuse_removed_backend(self):
        lease = await self.router.acquire(self.discovery)
        remaining = self.b if lease.addr == self.a else self.a
        self.discovery._addrs = (remaining,)
        self.assertEqual((await self.router.acquire(self.discovery)).addr, remaining)
        self.assertNotIn(lease.addr, self.router._cache)
        # A dispatched request stays pinned even if discovery removes its host.
        self.assertNotEqual(lease.addr, remaining)
        self.router.release(lease)

    async def test_failed_probe_does_not_fall_back_to_random(self):
        async def fail(addr):
            raise OSError('unavailable')
        self.router._fetch = fail
        self.assertIsNone(await self.router.acquire(self.discovery))

    async def test_cancellation_leaves_no_dispatch_reservation(self):
        entered = asyncio.Event()
        blocked = asyncio.Event()
        async def fetch(addr):
            entered.set()
            await blocked.wait()
        self.router._fetch = fetch
        task = asyncio.create_task(self.router.acquire(self.discovery))
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertFalse(self.router._pending)

    def test_opt_in_and_invalid_configuration(self):
        with patch.dict('os.environ', {}, clear=True):
            self.assertIsNone(create_workload_router_from_env())
        with patch.dict('os.environ', {'DASH_SC_GRPC_PROXY_LOAD_BALANCE': 'workload'}):
            self.assertIsInstance(create_workload_router_from_env(), WorkloadRouter)
        with patch.dict('os.environ', {'DASH_SC_GRPC_PROXY_LOAD_BALANCE': 'typo'}):
            with self.assertRaises(ValueError):
                create_workload_router_from_env()

    async def test_real_http_probe_and_size_bound(self):
        from aiohttp import web

        oversized = False
        async def handle(request):
            if oversized:
                return web.Response(text=' ' * 17000, content_type='application/json')
            return web.json_response(snapshot(remaining_prefill_positions=12345))
        app = web.Application()
        app.router.add_get('/scheduling', handle)
        runner = web.AppRunner(app)
        await runner.setup()
        self.addAsyncCleanup(runner.cleanup)
        site = web.TCPSite(runner, '127.0.0.1', 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        discovery = IpPortListServiceDiscovery(f'127.0.0.1:{port}')
        router = WorkloadRouter(ttl=0)
        self.addAsyncCleanup(router.close)
        lease = await router.acquire(discovery)
        self.assertEqual(lease.addr.http_port, port)
        router.release(lease)
        oversized = True
        self.assertIsNone(await router.acquire(discovery))


if __name__ == '__main__':
    unittest.main()
