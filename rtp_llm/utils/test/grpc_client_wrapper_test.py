import unittest

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
            client._dp_stubs[address] = stubs[address]

        client._ensure_dp_connection = ensure_connection
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
            client._dp_stubs[address] = stubs[address]

        client._ensure_dp_connection = ensure_connection
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
            client._dp_stubs[address] = stub

        client._ensure_dp_connection = ensure_connection
        result = await client.dump_torch_allocator()

        self.assertEqual(result["status"], "error")
        self.assertEqual(len(result["backends"]), 1)
        self.assertEqual(result["errors"], ["dp0:10000/world_rank=0: snapshot failed"])


if __name__ == "__main__":
    unittest.main()
