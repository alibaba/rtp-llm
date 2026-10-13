import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import grpc

from rtp_llm.utils.grpc_host_channel_pool import GrpcHostChannel, GrpcHostChannelPool


class GrpcHostChannelPoolTest(unittest.IsolatedAsyncioTestCase):
    """Test cases for GrpcHostChannelPool"""

    async def asyncSetUp(self):
        """Setup test environment"""
        self.test_host = "localhost:50051"
        self.test_options = [("grpc.max_receive_message_length", 1000000)]
        self.pool = GrpcHostChannelPool(
            options=self.test_options, cleanup_interval=3600
        )

    async def asyncTearDown(self):
        await self.pool.close()

    async def test_pool_starts_lazily_and_close_stops_task(self):
        """The cleanup task starts on first use and is stopped by close()."""
        await self.pool.get(self.test_host)
        self.assertIsNotNone(self.pool._cleanup_task)
        self.assertFalse(self.pool._cleanup_task.done())

        cleanup_task = self.pool._cleanup_task
        await self.pool.close()

        self.assertTrue(self.pool._stopped)
        self.assertTrue(cleanup_task.done())
        self.assertEqual(self.pool._channels, {})
        self.assertEqual(self.pool._closed_channels, [])

    async def test_close_closes_active_channels_and_is_idempotent(self):
        mock_channel = MagicMock()
        mock_channel.get_state.return_value = grpc.ChannelConnectivity.READY
        mock_channel.close = AsyncMock()

        with patch(
            "rtp_llm.utils.grpc_host_channel_pool.aio.insecure_channel",
            return_value=mock_channel,
        ):
            await self.pool.get(self.test_host)

        await self.pool.close()
        await self.pool.close()

        mock_channel.close.assert_awaited_once()
        with self.assertRaisesRegex(RuntimeError, "is closed"):
            await self.pool.get(self.test_host)

    async def test_close_takes_over_channel_from_cancelled_cleanup(self):
        """A channel selected by cleanup remains owned until close succeeds."""
        cleanup_close_started = asyncio.Event()
        never_finish_first_close = asyncio.Event()
        close_attempts = 0

        async def close_channel():
            nonlocal close_attempts
            close_attempts += 1
            if close_attempts == 1:
                cleanup_close_started.set()
                await never_finish_first_close.wait()

        mock_channel = MagicMock()
        mock_channel.get_state.return_value = grpc.ChannelConnectivity.SHUTDOWN
        mock_channel.close = AsyncMock(side_effect=close_channel)

        self.pool._cleanup_interval = 0
        with patch(
            "rtp_llm.utils.grpc_host_channel_pool.aio.insecure_channel",
            return_value=mock_channel,
        ):
            await self.pool.get(self.test_host)

        await asyncio.wait_for(cleanup_close_started.wait(), timeout=1)
        await self.pool.close()

        self.assertEqual(close_attempts, 2)
        self.assertEqual(self.pool._pending_close_channels, {})
        mock_channel.close.assert_awaited()

    async def test_get_channel(self):
        """Test getting a channel from the pool"""
        # Get a channel for the first time
        channel = await self.pool.get(self.test_host)
        self.assertIsNotNone(channel)

        # Verify channel is cached
        self.assertIn(self.test_host, self.pool._channels)
        entry = self.pool._channels[self.test_host]
        self.assertEqual(entry.host, self.test_host)
        self.assertEqual(entry.channel, channel)

    async def test_get_channel_reuse(self):
        """Test that getting the same host returns the same channel"""
        # Get channel twice
        channel1 = await self.pool.get(self.test_host)
        channel2 = await self.pool.get(self.test_host)

        # Should be the same channel
        self.assertIs(channel1, channel2)

        # Should only have one entry in the pool
        self.assertEqual(len(self.pool._channels), 1)
        self.assertIn(self.test_host, self.pool._channels)

    async def test_get_reuses_transient_failure_channel(self):
        """A temporary outage must not cause one new channel per request."""
        mock_channel = MagicMock()
        mock_channel.get_state.return_value = (
            grpc.ChannelConnectivity.TRANSIENT_FAILURE
        )
        mock_channel.close = AsyncMock()

        with patch(
            "rtp_llm.utils.grpc_host_channel_pool.aio.insecure_channel",
            return_value=mock_channel,
        ) as create_channel:
            channel1 = await self.pool.get(self.test_host)
            channel2 = await self.pool.get(self.test_host)

        self.assertIs(channel1, channel2)
        create_channel.assert_called_once_with(
            self.test_host, options=self.test_options
        )
        self.assertEqual(self.pool._closed_channels, [])

    async def test_get_multiple_hosts(self):
        """Test getting channels for multiple hosts"""
        hosts = [f"localhost:{50051 + i}" for i in range(3)]
        channels = []

        # Get channels for different hosts
        for host in hosts:
            channel = await self.pool.get(host)
            channels.append(channel)

        # All channels should be different
        for i in range(len(channels)):
            for j in range(i + 1, len(channels)):
                self.assertIsNot(channels[i], channels[j])

        # Should have entries for all hosts
        self.assertEqual(len(self.pool._channels), 3)
        for host in hosts:
            self.assertIn(host, self.pool._channels)

    async def test_channel_closed_recreation(self):
        """Test that closed channels are recreated"""
        # Get a channel
        channel = await self.pool.get(self.test_host)
        original_entry = self.pool._channels[self.test_host]

        await channel.close()

        # Get channel again, should create new one
        new_channel = await self.pool.get(self.test_host)
        new_entry = self.pool._channels[self.test_host]

        # Should have new entry
        self.assertIsNot(original_entry, new_entry)
        self.assertIsNot(channel, new_channel)

    def test_is_channel_closed(self):
        """Test _is_channel_closed method"""
        # Create a mock entry with closed channel
        mock_channel = MagicMock()
        mock_channel.get_state.return_value = grpc.ChannelConnectivity.SHUTDOWN
        entry = GrpcHostChannel(self.test_host, mock_channel)

        # Should detect closed channel
        is_closed = self.pool._is_channel_closed(entry)
        self.assertTrue(is_closed)

        # Test with active channel
        mock_channel.get_state.return_value = grpc.ChannelConnectivity.READY
        is_closed = self.pool._is_channel_closed(entry)
        self.assertFalse(is_closed)

        # Test exception handling
        mock_channel.get_state.side_effect = Exception("Test error")
        is_closed = self.pool._is_channel_closed(entry)
        self.assertTrue(is_closed)  # Should return True on exception

    def test_transient_failure_is_not_closed(self):
        """TRANSIENT_FAILURE channels must be retained for automatic recovery."""
        mock_channel = MagicMock()
        mock_channel.get_state.return_value = (
            grpc.ChannelConnectivity.TRANSIENT_FAILURE
        )
        entry = GrpcHostChannel(self.test_host, mock_channel)

        self.assertFalse(self.pool._is_channel_closed(entry))

    async def test_cleanup_evicts_persistent_transient_failure(self):
        """A permanently unavailable peer is evicted after the grace period."""
        mock_channel = MagicMock()
        mock_channel.get_state.return_value = (
            grpc.ChannelConnectivity.TRANSIENT_FAILURE
        )
        mock_channel.close = AsyncMock()

        with patch(
            "rtp_llm.utils.grpc_host_channel_pool.aio.insecure_channel",
            return_value=mock_channel,
        ):
            await self.pool.get(self.test_host)

        await self.pool._cleanup_closed()
        self.assertIn(self.test_host, self.pool._channels)

        self.pool._transient_failure_grace_period = 0
        await self.pool._cleanup_closed()
        self.assertNotIn(self.test_host, self.pool._channels)
        mock_channel.close.assert_awaited_once()

    async def test_cleanup_evicts_unused_idle_channel(self):
        """An unused IDLE peer is evicted without forcing a connection attempt."""
        mock_channel = MagicMock()
        mock_channel.get_state.return_value = grpc.ChannelConnectivity.IDLE
        mock_channel.close = AsyncMock()

        with patch(
            "rtp_llm.utils.grpc_host_channel_pool.aio.insecure_channel",
            return_value=mock_channel,
        ):
            await self.pool.get(self.test_host)

        await self.pool._cleanup_closed()
        self.assertIn(self.test_host, self.pool._channels)

        self.pool._idle_channel_ttl = 0
        await self.pool._cleanup_closed()

        self.assertNotIn(self.test_host, self.pool._channels)
        mock_channel.get_state.assert_called_with()
        mock_channel.close.assert_awaited_once()

    async def test_cleanup_keeps_ready_channel_past_idle_ttl(self):
        """The idle TTL must not close a channel that may serve an active RPC."""
        mock_channel = MagicMock()
        mock_channel.get_state.return_value = grpc.ChannelConnectivity.READY
        mock_channel.close = AsyncMock()

        with patch(
            "rtp_llm.utils.grpc_host_channel_pool.aio.insecure_channel",
            return_value=mock_channel,
        ):
            await self.pool.get(self.test_host)

        self.pool._idle_channel_ttl = 0
        await self.pool._cleanup_closed()

        self.assertIn(self.test_host, self.pool._channels)
        mock_channel.close.assert_not_awaited()

    async def test_cleanup_closed_channels(self):
        """Test cleanup of closed channels"""
        # Create multiple channels
        hosts = [f"localhost:{50051 + i}" for i in range(3)]

        for host in hosts:
            await self.pool.get(host)

        # Close one channel
        entry = self.pool._channels[hosts[1]]
        await entry.channel.close()

        # Run cleanup
        await self.pool._cleanup_closed()

        # Should have only 2 channels remaining
        self.assertEqual(len(self.pool._channels), 2)
        self.assertNotIn(hosts[1], self.pool._channels)
        self.assertIn(hosts[0], self.pool._channels)
        self.assertIn(hosts[2], self.pool._channels)

    async def test_cleanup_loop(self):
        """The background loop invokes cleanup without relying on sleep timing."""
        cleanup_called = asyncio.Event()
        keep_cleanup_pending = asyncio.Event()

        async def cleanup_once():
            cleanup_called.set()
            await keep_cleanup_pending.wait()

        self.pool._cleanup_interval = 0
        self.pool._cleanup_closed = AsyncMock(side_effect=cleanup_once)
        await self.pool.get(self.test_host)

        await asyncio.wait_for(cleanup_called.wait(), timeout=1)
        await self.pool.close()

        self.pool._cleanup_closed.assert_awaited_once()

    def test_destructor(self):
        """Test __del__ method"""
        pool = GrpcHostChannelPool(options=self.test_options)
        pool._channels[self.test_host] = GrpcHostChannel(self.test_host, MagicMock())

        # Call __del__ manually
        pool.__del__()

        # Should be marked as stopped
        self.assertTrue(pool._stopped)
        # Channels should be cleared
        self.assertEqual(len(pool._channels), 0)

    async def test_concurrent_get(self):
        """Test concurrent access to get channels"""

        async def get_channel(host):
            return await self.pool.get(host)

        # Create multiple concurrent tasks
        tasks = []
        for i in range(10):
            tasks.append(get_channel(self.test_host))

        # Wait for all tasks
        channels = await asyncio.gather(*tasks)

        # All should get the same channel
        for channel in channels[1:]:
            self.assertIs(channels[0], channel)

        # Should only have one entry
        self.assertEqual(len(self.pool._channels), 1)

    async def test_channel_close_timeout(self):
        """Test handling of channel close timeout"""
        # Create a mock channel that times out on close
        mock_channel = MagicMock()
        mock_channel.close = AsyncMock(side_effect=asyncio.TimeoutError())

        # Add to pool
        self.pool._channels[self.test_host] = GrpcHostChannel(
            self.test_host, mock_channel
        )

        # Mark as closed
        mock_channel.get_state.return_value = grpc.ChannelConnectivity.SHUTDOWN

        # Cleanup should handle timeout gracefully
        await self.pool._cleanup_closed()

        # Channel should be removed from pool despite close timeout
        self.assertNotIn(self.test_host, self.pool._channels)
        self.assertEqual(len(self.pool._pending_close_channels), 1)

        mock_channel.close.side_effect = None
        await self.pool.close()
        self.assertEqual(mock_channel.close.await_count, 2)
        self.assertEqual(self.pool._pending_close_channels, {})

    async def test_channel_close_exception(self):
        """Test handling of channel close exception"""
        # Create a mock channel that raises exception on close
        mock_channel = MagicMock()
        mock_channel.close = AsyncMock(side_effect=Exception("Close failed"))

        # Add to pool
        self.pool._channels[self.test_host] = GrpcHostChannel(
            self.test_host, mock_channel
        )

        # Mark as closed
        mock_channel.get_state.return_value = grpc.ChannelConnectivity.SHUTDOWN

        # Cleanup should handle exception gracefully
        await self.pool._cleanup_closed()

        # Channel should be removed from pool despite close exception
        self.assertNotIn(self.test_host, self.pool._channels)
        self.assertEqual(len(self.pool._pending_close_channels), 1)

        mock_channel.close.side_effect = None
        await self.pool.close()
        self.assertEqual(mock_channel.close.await_count, 2)
        self.assertEqual(self.pool._pending_close_channels, {})


if __name__ == "__main__":
    unittest.main()
