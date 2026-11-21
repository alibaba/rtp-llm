import unittest

import grpc

from rtp_llm.config.server_config import ServerConfig
from rtp_llm.cpp.model_rpc.proto import model_rpc_service_pb2 as pb
from rtp_llm.cpp.model_rpc.proto import model_rpc_service_pb2_grpc as rpc
from rtp_llm.utils.test.smoke_weight_update_rpc import (
    paused_weight_update,
    rank_addresses,
)


class Rank(rpc.RpcServiceServicer):
    def __init__(self, state, rank, fail):
        self.state, self.rank, self.fail = state, rank, fail

    async def call(self, phase, request, ctx):
        self.state[self.rank].append(phase)
        if self.rank == 1 and phase == self.fail:
            await ctx.abort(grpc.StatusCode.FAILED_PRECONDITION, phase)
        if phase == "update" and (
            not all("pause" in calls for calls in self.state)
            or (request.name, request.desc, request.method)
            != ("norm", "abcd", "cuda_ipc")
        ):
            await ctx.abort(
                grpc.StatusCode.FAILED_PRECONDITION, "incomplete pause/update"
            )
        if phase == "restart" and not all("update" in calls for calls in self.state):
            await ctx.abort(grpc.StatusCode.FAILED_PRECONDITION, "incomplete update")
        return pb.EmptyPB()

    async def SetPause(self, request, context):
        return await self.call("pause", request, context)

    async def UpdateWeights(self, request, context):
        return await self.call("update", request, context)

    async def SetRestart(self, request, context):
        return await self.call("restart", request, context)


class UpdateBarrierTest(unittest.IsolatedAsyncioTestCase):
    def test_server_port_layout(self):
        config = ServerConfig()
        config.worker_info_port_num = 13
        self.assertEqual(
            rank_addresses(16000, config, 2),
            ["127.0.0.1:16001", "127.0.0.1:16014"],
        )

    async def test_all_ranks_and_failure_barriers(self):
        for failure, phases in (
            (None, ["pause", "update", "restart"]),
            ("pause", ["pause"]),
            ("update", ["pause", "update"]),
            ("restart", ["pause", "update", "restart"]),
        ):
            with self.subTest(failure=failure):
                state = [[], []]
                servers, members = [], []
                try:
                    for rank in range(2):
                        server = grpc.aio.server()
                        rpc.add_RpcServiceServicer_to_server(
                            Rank(state, rank, failure), server
                        )
                        port = server.add_insecure_port("127.0.0.1:0")
                        await server.start()
                        servers.append(server)
                        members.append(f"127.0.0.1:{port}")
                    op = paused_weight_update(members, "norm", "abcd", 10)
                    if failure is None:
                        await op
                    else:
                        with self.assertRaisesRegex(RuntimeError, failure):
                            await op
                    self.assertEqual(state, [phases, phases])
                finally:
                    for server in servers:
                        await server.stop(0)


if __name__ == "__main__":
    unittest.main()
