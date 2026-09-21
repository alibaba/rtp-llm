import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import grpc

from rtp_llm.utils.grpc_host_channel_pool import GrpcHostChannelPool


class GrpcHostChannelPoolTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.factory = patch(
            "rtp_llm.utils.grpc_host_channel_pool.aio.insecure_channel",
            side_effect=self.make_channel,
        )
        self.factory.start()
        self.addCleanup(self.factory.stop)
        self.pool = GrpcHostChannelPool(idle_ttl=10, max_channels=2)
        self.addAsyncCleanup(self.pool.close)

    @staticmethod
    def make_channel(*args, **kwargs):
        channel = MagicMock()
        channel.get_state.return_value = grpc.ChannelConnectivity.READY
        channel.close = AsyncMock()
        return channel

    async def flush_closes(self):
        tasks = list(self.pool._close_tasks)
        if tasks:
            await asyncio.gather(*tasks)

    async def test_shared_leases_and_last_release_starts_idle_timer(self):
        async with self.pool.acquire("a") as first:
            entry = self.pool._channels["a"]
            async with self.pool.acquire("a") as second:
                self.assertIs(first, second)
                self.assertEqual(entry.active_calls, 2)
            self.assertEqual(entry.active_calls, 1)
            with patch(
                "rtp_llm.utils.grpc_host_channel_pool.time.monotonic", return_value=100
            ):
                await self.pool._cleanup_closed()
                first.close.assert_not_awaited()
        self.assertEqual(entry.active_calls, 0)
        self.assertGreater(entry.idle_since, 100)

    async def test_long_stream_survives_ttl_and_expires_after_release(self):
        async with self.pool.acquire("a") as channel:
            self.pool._channels["a"].idle_since = 0
            await self.pool._cleanup_closed()
            self.assertIn("a", self.pool._channels)
            channel.close.assert_not_awaited()
        await self.pool._cleanup_closed()
        self.assertIn("a", self.pool._channels)
        self.pool._channels["a"].idle_since -= 11
        await self.pool._cleanup_closed()
        await self.flush_closes()
        self.assertNotIn("a", self.pool._channels)
        channel.close.assert_awaited_once()

    async def test_transient_failure_recovers_without_replacement_or_close(self):
        async with self.pool.acquire("a") as channel:
            channel.get_state.return_value = grpc.ChannelConnectivity.TRANSIENT_FAILURE
            async with self.pool.acquire("a") as same:
                self.assertIs(channel, same)
        await self.pool._cleanup_closed()
        channel.close.assert_not_awaited()
        channel.get_state.return_value = grpc.ChannelConnectivity.READY
        await self.pool._cleanup_closed()
        async with self.pool.acquire("a") as same:
            self.assertIs(channel, same)
        channel.close.assert_not_awaited()

    async def test_transient_failure_idle_channel_expires(self):
        async with self.pool.acquire("a") as channel:
            channel.get_state.return_value = grpc.ChannelConnectivity.TRANSIENT_FAILURE
        self.pool._channels["a"].idle_since -= 11
        await self.pool._cleanup_closed()
        await self.flush_closes()
        channel.close.assert_awaited_once()

    async def test_state_query_error_does_not_evict_active_channel(self):
        async with self.pool.acquire("a") as channel:
            channel.get_state.side_effect = RuntimeError("unavailable")
            async with self.pool.acquire("a") as same:
                self.assertIs(channel, same)
            await self.pool._cleanup_closed()
        channel.close.assert_not_awaited()

    async def test_shutdown_replaced_after_last_lease(self):
        async with self.pool.acquire("a") as old:
            old.get_state.return_value = grpc.ChannelConnectivity.SHUTDOWN
            with self.assertRaises(asyncio.TimeoutError):
                async with self.pool.acquire("a", timeout=0.01):
                    self.fail("shutdown channel must not accept a new lease")
            old.close.assert_not_awaited()
        async with self.pool.acquire("a") as new:
            self.assertIsNot(new, old)
        await self.flush_closes()
        old.close.assert_awaited_once()

    async def test_capacity_evicts_oldest_idle_only(self):
        async with self.pool.acquire("a") as a:
            pass
        async with self.pool.acquire("b") as b:
            async with self.pool.acquire("c"):
                self.assertEqual(set(self.pool._channels), {"b", "c"})
                b.close.assert_not_awaited()
        await self.flush_closes()
        a.close.assert_awaited_once()

    async def test_capacity_wait_wakes_on_release(self):
        self.pool._max_channels = 1
        started = asyncio.Event()

        async def waiter():
            started.set()
            async with self.pool.acquire("b"):
                return "acquired"

        async with self.pool.acquire("a"):
            task = asyncio.create_task(waiter())
            await started.wait()
            await asyncio.sleep(0)
            self.assertFalse(task.done())
        self.assertEqual(await asyncio.wait_for(task, 1), "acquired")

    async def test_capacity_uses_last_release_order(self):
        async with self.pool.acquire("a"):
            pass
        async with self.pool.acquire("b") as b:
            pass
        async with self.pool.acquire("a") as a:
            pass
        async with self.pool.acquire("c"):
            self.assertEqual(set(self.pool._channels), {"a", "c"})
        await self.flush_closes()
        a.close.assert_not_awaited()
        b.close.assert_awaited_once()

    async def test_cancellation_racing_completed_acquire_releases_lease(self):
        original_take = self.pool._take
        caller = asyncio.current_task()

        async def take_and_cancel(target):
            entry = await original_take(target)
            caller.cancel()
            return entry

        with patch.object(self.pool, "_take", side_effect=take_and_cancel):
            with self.assertRaises(asyncio.CancelledError):
                async with self.pool.acquire("a"):
                    self.fail("cancelled acquisition")
        self.assertEqual(self.pool._channels["a"].active_calls, 0)

    async def test_capacity_timeout_does_not_close_active_channel(self):
        self.pool._max_channels = 1
        async with self.pool.acquire("a") as channel:
            with self.assertRaises(asyncio.TimeoutError):
                async with self.pool.acquire("b", timeout=0.01):
                    self.fail("capacity exceeded")
            self.assertEqual(self.pool._channels["a"].active_calls, 1)
            channel.close.assert_not_awaited()

    async def test_exception_and_cancellation_release_leases(self):
        with self.assertRaisesRegex(ValueError, "RPC failed"):
            async with self.pool.acquire("a"):
                raise ValueError("RPC failed")
        self.assertEqual(self.pool._channels["a"].active_calls, 0)
        entered = asyncio.Event()

        async def rpc():
            async with self.pool.acquire("a"):
                entered.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(rpc())
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(self.pool._channels["a"].active_calls, 0)

    async def test_cancelled_capacity_waiter_does_not_leak(self):
        self.pool._max_channels = 1
        async with self.pool.acquire("a"):

            async def wait():
                async with self.pool.acquire("b"):
                    self.fail("capacity exceeded")

            task = asyncio.create_task(wait())
            await asyncio.sleep(0)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertEqual(set(self.pool._channels), {"a"})
        self.assertEqual(self.pool._channels["a"].active_calls, 0)

    async def test_close_wakes_waiters_and_is_idempotent(self):
        self.pool._max_channels = 1
        async with self.pool.acquire("a") as channel:

            async def wait():
                async with self.pool.acquire("b"):
                    self.fail("closed pool")

            task = asyncio.create_task(wait())
            await asyncio.sleep(0)
            await self.pool.close()
            with self.assertRaisesRegex(RuntimeError, "closed"):
                await task
            channel.close.assert_awaited_once()
        await self.pool.close()
        channel.close.assert_awaited_once()

    async def test_concurrent_acquire_and_cleanup_protects_leases(self):
        async def rpc():
            async with self.pool.acquire("a") as channel:
                self.pool._channels["a"].idle_since = 0
                await self.pool._cleanup_closed()
                await asyncio.sleep(0)
                channel.close.assert_not_awaited()
                return channel

        channels = await asyncio.gather(*(rpc() for _ in range(20)))
        self.assertTrue(all(c is channels[0] for c in channels))
        self.assertEqual(self.pool._channels["a"].active_calls, 0)

    async def test_cleanup_does_not_connect_idle_channels(self):
        async with self.pool.acquire("a") as channel:
            channel.get_state.return_value = grpc.ChannelConnectivity.IDLE
        channel.get_state.reset_mock()
        await self.pool._cleanup_closed()
        channel.get_state.assert_called_once_with()

    async def test_close_failure_does_not_break_pool(self):
        async with self.pool.acquire("a") as channel:
            channel.close.side_effect = RuntimeError("close failed")
        self.pool._channels["a"].idle_since -= 11
        await self.pool._cleanup_closed()
        await self.flush_closes()
        async with self.pool.acquire("a") as new:
            self.assertIsNot(new, channel)

    def test_invalid_limits(self):
        for kwargs in (
            {"idle_ttl": 0},
            {"max_channels": 0},
            {"acquire_timeout": -1},
            {"cleanup_interval": 0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                GrpcHostChannelPool(**kwargs)


if __name__ == "__main__":
    unittest.main()
