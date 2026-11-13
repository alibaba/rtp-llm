import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import grpc

import rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 as pb2
from rtp_llm.utils.grpc_client_wrapper import GrpcClientWrapper


def _response(*world_ranks: int) -> pb2.TorchAllocatorDumpResponsePB:
    response = pb2.TorchAllocatorDumpResponsePB()
    for world_rank in world_ranks:
        result = response.results.add()
        result.world_rank = world_rank
        result.dp_rank = world_rank // 2
        result.tp_rank = world_rank % 2
        result.local_rank = world_rank
        result.pid = 1000 + world_rank
        result.success = True
        result.file_path = f"/logs/oom_allocator_rank_{world_rank}.log"
    return response


class _Stub:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = 0

    async def DumpTorchAllocator(self, request, timeout):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.response


class GrpcClientWrapperTest(unittest.IsolatedAsyncioTestCase):
    async def test_dump_torch_allocator_fans_out_all_dp_roots(self):
        client = GrpcClientWrapper(
            server_port=10000,
            dp_addresses=["dp0:10000", "dp1:10000"],
        )
        stubs = {
            "dp0:10000": _Stub(_response(0, 1)),
            "dp1:10000": _Stub(_response(2, 3)),
        }

        async def ensure_connection(address):
            client._control_rpc.stubs[address] = stubs[address]

        client._control_rpc.ensure = ensure_connection
        result = await client.dump_torch_allocator()

        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(result["backends"]), 4)
        self.assertEqual(result["errors"], [])
        self.assertEqual(
            {backend["world_rank"] for backend in result["backends"]},
            {0, 1, 2, 3},
        )
        self.assertEqual(stubs["dp0:10000"].calls, 1)
        self.assertEqual(stubs["dp1:10000"].calls, 1)

    async def test_dump_torch_allocator_preserves_partial_results(self):
        client = GrpcClientWrapper(
            server_port=10000,
            dp_addresses=["dp0:10000", "dp1:10000"],
        )
        stubs = {
            "dp0:10000": _Stub(_response(0, 1)),
            "dp1:10000": _Stub(error=RuntimeError("backend unavailable")),
        }

        async def ensure_connection(address):
            client._control_rpc.stubs[address] = stubs[address]

        client._control_rpc.ensure = ensure_connection
        result = await client.dump_torch_allocator()

        self.assertEqual(result["status"], "error")
        self.assertEqual(len(result["backends"]), 2)
        self.assertEqual(len(result["errors"]), 1)
        self.assertIn("dp1:10000", result["errors"][0])

    async def test_dump_torch_allocator_reports_backend_dump_failure(self):
        response = _response(0)
        response.results[0].success = False
        response.results[0].error = "snapshot failed"
        client = GrpcClientWrapper(server_port=10000, dp_addresses=["dp0:10000"])
        stub = _Stub(response)

        async def ensure_connection(address):
            client._control_rpc.stubs[address] = stub

        client._control_rpc.ensure = ensure_connection
        result = await client.dump_torch_allocator()

        self.assertEqual(result["status"], "error")
        self.assertEqual(len(result["backends"]), 1)
        self.assertEqual(result["errors"], ["dp0:10000/world_rank=0: snapshot failed"])

    async def test_health_check_failure_preserves_lifecycle_channels(self):
        # Regression: a routine health probe timing out during a sleep/wake
        # drain must NOT tear down the addressed control channels. Closing a
        # channel under a genuinely in-flight SleepServing/WakeUpServing call
        # raises asyncio.CancelledError into that RPC (a BaseException that
        # bypasses every ``except Exception``), cancelling the operation and
        # returning HTTP 500 while the backend keeps transitioning -- a
        # control-plane split brain. health_check may only reset its own
        # channel.
        addresses = ["127.0.0.1:10001", "127.0.0.1:10009"]
        wrapper = GrpcClientWrapper(server_port=12345, control_addresses=addresses)
        for address in addresses:
            wrapper._control_rpc.channels[address] = MagicMock()
            wrapper._control_rpc.stubs[address] = MagicMock()
        wrapper.channel = MagicMock()
        wrapper.channel.close = AsyncMock()
        wrapper.stub = MagicMock()
        wrapper.stub.CheckHealth = AsyncMock(
            side_effect=grpc.aio.AioRpcError(
                grpc.StatusCode.DEADLINE_EXCEEDED,
                grpc.aio.Metadata(),
                grpc.aio.Metadata(),
                "backend draining",
            )
        )
        dp_channels_before = dict(wrapper._control_rpc.channels)
        dp_stubs_before = dict(wrapper._control_rpc.stubs)

        result = await wrapper.health_check()

        self.assertEqual(result["status"], "error")
        # Only the health channel is reset; lifecycle channels stay intact.
        self.assertIsNone(wrapper.channel)
        self.assertIsNone(wrapper.stub)
        self.assertEqual(wrapper._control_rpc.channels, dp_channels_before)
        self.assertEqual(wrapper._control_rpc.stubs, dp_stubs_before)
        for address in addresses:
            self.assertFalse(wrapper._control_rpc.channels[address].close.called)

    async def test_public_lifecycle_routes_delegate_to_controller(self):
        client = GrpcClientWrapper(server_port=10000)
        for uri, method in (
            ("sleep", "sleep_serving"),
            ("wake_up", "wake_up_serving"),
            ("sleep_status", "get_sleep_status"),
            ("is_sleeping", "is_sleeping"),
        ):
            with self.subTest(uri=uri):
                request = {"reason": "boundary-test"}
                response = {"marker": uri}
                with patch.object(
                    client._lifecycle, method, AsyncMock(return_value=response)
                ) as action:
                    self.assertIs(await client.post_request(uri, request), response)
                    action.assert_awaited_once_with(request)

    async def test_controller_owns_coordination_and_uses_all_ranks(self):
        client = GrpcClientWrapper(
            server_port=10000,
            dp_addresses=["rank-0"],
            control_addresses=["rank-0", "rank-1"],
            expected_control_address_count=2,
        )
        self.assertEqual(client.dp_addresses, ["rank-0"])
        self.assertEqual(client._lifecycle.control_addresses, ["rank-0", "rank-1"])
        self.assertIs(client._lifecycle._rpc, client._control_rpc)
        # No duplicate orchestration state or obsolete compatibility aliases.
        for attribute in (
            "_lifecycle_lock",
            "_lifecycle_lease",
            "control_addresses",
            "_dp_channels",
            "_dp_stubs",
            "_converge_commit",
        ):
            self.assertNotIn(attribute, vars(client))
        self.assertIsInstance(client._lifecycle._lifecycle_lock, asyncio.Lock)

    async def test_close_releases_shared_transport_once(self):
        client = GrpcClientWrapper(server_port=10000)
        client.channel = MagicMock(close=AsyncMock())
        health_channel = client.channel
        client._control_rpc.close = AsyncMock()
        await client.close()
        health_channel.close.assert_awaited_once()
        client._control_rpc.close.assert_awaited_once()
        self.assertIsNone(client.channel)
        self.assertIsNone(client.stub)

    async def test_normal_status_rpcs_keep_their_metrics_and_main_channel(self):
        client = GrpcClientWrapper(server_port=10000)
        client.channel = MagicMock()
        client.stub = MagicMock(
            GetCacheStatus=AsyncMock(return_value=pb2.EmptyPB()),
            GetWorkerStatus=AsyncMock(return_value=pb2.EmptyPB()),
        )
        with patch("rtp_llm.utils.grpc_client_wrapper.kmonitor.report") as report:
            self.assertEqual(await client.get_cache_status({}), {})
            self.assertEqual(await client.get_worker_status({}), {})
            self.assertEqual(report.call_count, 4)
        self.assertEqual(client._control_rpc.channels, {})


if __name__ == "__main__":
    unittest.main()
