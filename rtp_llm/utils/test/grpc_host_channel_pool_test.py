import asyncio
import contextlib
import unittest
from unittest import TestCase
from unittest.mock import AsyncMock, MagicMock

import grpc

from rtp_llm.utils.grpc_host_channel_pool import GrpcHostChannel, GrpcHostChannelPool


class GrpcHostChannelPoolTest(TestCase):
    """Test cases for GrpcHostChannelPool"""

    def setUp(self):
        """Setup test environment"""
        self.test_host = "localhost:50051"
        self.test_options = [("grpc.max_receive_message_length", 1000000)]
        self.pool = GrpcHostChannelPool(options=self.test_options, cleanup_interval=1)

    def tearDown(self):
        """Cleanup after tests"""
        # Stop the cleanup task
        if hasattr(self.pool, "_stopped"):
            self.pool._stopped = True
        if hasattr(self.pool, "_cleanup_task") and self.pool._cleanup_task:
            self.pool._cleanup_task.cancel()
        # Clear channels
        if hasattr(self.pool, "_channels"):
            self.pool._channels.clear()

    async def test_pool_start_stop(self):
        """Test starting and stopping the pool cleanup task"""
        # Test start
        await self.pool.start()
        self.assertIsNotNone(self.pool._cleanup_task)
        self.assertFalse(self.pool._cleanup_task.done())

        # Test stop
        self.pool._stopped = True
        if self.pool._cleanup_task:
            self.pool._cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.pool._cleanup_task

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

        # Simulate closed channel
        original_entry.channel._state = grpc.ChannelConnectivity.SHUTDOWN

        # Get channel again, should create new one
        new_channel = await self.pool.get(self.test_host)
        new_entry = self.pool._channels[self.test_host]

        # Should have new entry
        self.assertIsNot(original_entry, new_entry)
        self.assertIsNot(channel, new_channel)

    async def test_is_channel_closed(self):
        """Test _is_channel_closed method"""
        # Create a mock entry with closed channel
        mock_channel = MagicMock()
        mock_channel.get_state.return_value = grpc.ChannelConnectivity.SHUTDOWN
        entry = GrpcHostChannel(self.test_host, mock_channel)

        # Should detect closed channel
        is_closed = await self.pool._is_channel_closed(entry)
        self.assertTrue(is_closed)

        # Test with active channel
        mock_channel.get_state.return_value = grpc.ChannelConnectivity.READY
        is_closed = await self.pool._is_channel_closed(entry)
        self.assertFalse(is_closed)

        # Test exception handling
        mock_channel.get_state.side_effect = Exception("Test error")
        is_closed = await self.pool._is_channel_closed(entry)
        self.assertTrue(is_closed)  # Should return True on exception

    async def test_cleanup_closed_channels(self):
        """Test cleanup of closed channels"""
        # Create multiple channels
        hosts = [f"localhost:{50051 + i}" for i in range(3)]

        for host in hosts:
            await self.pool.get(host)

        # Close one channel
        entry = self.pool._channels[hosts[1]]
        entry.channel._state = grpc.ChannelConnectivity.SHUTDOWN

        # Run cleanup
        await self.pool._cleanup_closed()

        # Should have only 2 channels remaining
        self.assertEqual(len(self.pool._channels), 2)
        self.assertNotIn(hosts[1], self.pool._channels)
        self.assertIn(hosts[0], self.pool._channels)
        self.assertIn(hosts[2], self.pool._channels)

    async def test_cleanup_loop(self):
        """Test the cleanup loop task"""
        # Start cleanup with short interval
        await self.pool.start()

        # Add a channel and mark it as closed
        await self.pool.get(self.test_host)
        entry = self.pool._channels[self.test_host]
        entry.channel._state = grpc.ChannelConnectivity.SHUTDOWN

        # Wait for cleanup to run (interval is 1 second in setUp)
        await asyncio.sleep(1.5)

        # Channel should be cleaned up
        self.assertNotIn(self.test_host, self.pool._channels)

        # Stop cleanup
        self.pool._stopped = True
        if self.pool._cleanup_task:
            self.pool._cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.pool._cleanup_task

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


if __name__ == "__main__":
    unittest.main()
