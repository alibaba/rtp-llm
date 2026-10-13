import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import grpc

import rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 as pb2
import rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc as pb2_grpc
from rtp_llm.utils.grpc_client_wrapper import GrpcClientWrapper

_REQUEST = {"auth_token": "test-secret", "dump_id": "dump-correlation-1"}


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
        result.dump_id = _REQUEST["dump_id"]
        result.file_path = f"/logs/oom_allocator_rank_{world_rank}.log"
    return response


class _Stub:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = 0
        self.requests = []

    async def DumpTorchAllocator(self, request, timeout):
        self.calls += 1
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return self.response


class GrpcClientWrapperTest(unittest.IsolatedAsyncioTestCase):
    async def test_health_failure_preserves_all_shared_channels(self):
        client = GrpcClientWrapper(server_port=10000)
        health_channel = MagicMock(close=AsyncMock())
        client.channel = health_channel
        client.stub = MagicMock(
            CheckHealth=AsyncMock(
                side_effect=grpc.aio.AioRpcError(
                    grpc.StatusCode.DEADLINE_EXCEEDED,
                    grpc.aio.Metadata(),
                    grpc.aio.Metadata(),
                    "backend draining",
                )
            )
        )
        health_stub = client.stub
        pools = (
            (client._dp_channels, client._dp_stubs),
            (client._control_rpc.channels, client._control_rpc.stubs),
        )
        for channels, stubs in pools:
            channels["rank-0"] = MagicMock(close=AsyncMock())
            stubs["rank-0"] = MagicMock()
        snapshots = [(dict(channels), dict(stubs)) for channels, stubs in pools]

        self.assertEqual((await client.health_check())["status"], "error")

        health_channel.close.assert_not_awaited()
        self.assertIs(client.channel, health_channel)
        self.assertIs(client.stub, health_stub)
        for (channels, stubs), (old_channels, old_stubs) in zip(pools, snapshots):
            self.assertEqual(channels, old_channels)
            self.assertEqual(stubs, old_stubs)
            channels["rank-0"].close.assert_not_awaited()
        await client.close()
        health_channel.close.assert_awaited_once()
        for old_channels, _ in snapshots:
            old_channels["rank-0"].close.assert_awaited_once()
        for channels, stubs in pools:
            self.assertEqual(channels, {})
            self.assertEqual(stubs, {})

    async def test_failed_health_does_not_cancel_peer_rpc_on_real_channel(self):
        first_started = asyncio.Event()
        peer_started = asyncio.Event()
        release_peer = asyncio.Event()

        class HealthService(pb2_grpc.RpcServiceServicer):
            calls = 0

            async def CheckHealth(self, request, context):
                self.calls += 1
                if self.calls == 1:
                    first_started.set()
                    await peer_started.wait()
                    await context.abort(grpc.StatusCode.UNAVAILABLE, "backend draining")
                elif self.calls == 2:
                    peer_started.set()
                    await release_peer.wait()
                return pb2.EmptyPB()

        server = grpc.aio.server()
        pb2_grpc.add_RpcServiceServicer_to_server(HealthService(), server)
        port = server.add_insecure_port("127.0.0.1:0")
        self.assertNotEqual(port, 0)
        await server.start()
        client = GrpcClientWrapper(server_port=port)
        client.address = f"127.0.0.1:{port}"
        tasks = []
        try:
            first = asyncio.create_task(client.health_check())
            tasks.append(first)
            await asyncio.wait_for(first_started.wait(), timeout=2)
            channel = client.channel
            peer = asyncio.create_task(client.health_check())
            tasks.append(peer)
            self.assertEqual(
                (await asyncio.wait_for(first, timeout=2))["status"], "error"
            )
            release_peer.set()
            self.assertEqual(await asyncio.wait_for(peer, timeout=2), {"status": "ok"})
            self.assertIs(client.channel, channel)
            self.assertEqual(await client.health_check(), {"status": "ok"})
        finally:
            release_peer.set()
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await client.close()
            await server.stop(0)

    async def test_health_preserves_caller_cancellation(self):
        started = asyncio.Event()

        async def pending(*args, **kwargs):
            started.set()
            await asyncio.Event().wait()

        client = GrpcClientWrapper(server_port=10000)
        channel = MagicMock(close=AsyncMock())
        client.channel = channel
        client.stub = MagicMock(CheckHealth=pending)
        task = asyncio.create_task(client.health_check())
        try:
            await asyncio.wait_for(started.wait(), timeout=2)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            channel.close.assert_not_awaited()
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await client.close()

    async def test_public_lifecycle_routes_delegate_to_controller(self):
        client = GrpcClientWrapper(server_port=10000)
        for uri, method in (
            ("sleep", "sleep_serving"),
            ("wake_up", "wake_up_serving"),
            ("sleep_status", "get_sleep_status"),
            ("is_sleeping", "is_sleeping"),
        ):
            with self.subTest(uri=uri):
                request = {"reason": "migration-test"}
                response = {"marker": uri}
                with patch.object(
                    client._lifecycle, method, AsyncMock(return_value=response)
                ) as action:
                    self.assertIs(await client.post_request(uri, request), response)
                    action.assert_awaited_once_with(request)

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
            client._dp_stubs[address] = stubs[address]

        client._ensure_dp_connection = ensure_connection
        result = await client.dump_torch_allocator(_REQUEST)

        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(result["backends"]), 4)
        self.assertEqual(result["errors"], [])
        self.assertEqual(
            {backend["world_rank"] for backend in result["backends"]},
            {0, 1, 2, 3},
        )
        self.assertEqual(stubs["dp0:10000"].calls, 1)
        self.assertEqual(stubs["dp1:10000"].calls, 1)
        for stub in stubs.values():
            self.assertEqual(stub.requests[0].auth_token, _REQUEST["auth_token"])
            self.assertEqual(stub.requests[0].dump_id, _REQUEST["dump_id"])
        self.assertEqual(
            {backend["dump_id"] for backend in result["backends"]},
            {_REQUEST["dump_id"]},
        )

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
            client._dp_stubs[address] = stubs[address]

        client._ensure_dp_connection = ensure_connection
        result = await client.dump_torch_allocator(_REQUEST)

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
            client._dp_stubs[address] = stub

        client._ensure_dp_connection = ensure_connection
        result = await client.dump_torch_allocator(_REQUEST)

        self.assertEqual(result["status"], "error")
        self.assertEqual(len(result["backends"]), 1)
        self.assertEqual(result["errors"], ["dp0:10000/world_rank=0: snapshot failed"])

    async def test_dump_torch_allocator_rejects_mismatched_dump_id(self):
        response = _response(0, 1)
        response.results[1].dump_id = "different-request"
        client = GrpcClientWrapper(server_port=10000, dp_addresses=["dp0:10000"])
        stub = _Stub(response)

        async def ensure_connection(address):
            client._dp_stubs[address] = stub

        client._ensure_dp_connection = ensure_connection
        result = await client.dump_torch_allocator(_REQUEST)

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["backends"], [])
        self.assertEqual(
            result["errors"],
            ["dp0:10000/world_rank=1: allocator dump response id mismatch"],
        )


if __name__ == "__main__":
    unittest.main()
