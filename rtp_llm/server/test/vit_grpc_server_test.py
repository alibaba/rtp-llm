import threading
from unittest import TestCase, main

import grpc

from rtp_llm.server.vit_grpc_server import create_vit_grpc_server, stop_vit_grpc_server


class VitGrpcServerTest(TestCase):
    def test_release_runs_while_embedding_executor_is_saturated(self):
        entered = threading.Event()
        unblock = threading.Event()

        def embedding(request, context):
            entered.set()
            unblock.wait(5)
            return b"embedding"

        server = create_vit_grpc_server(max_workers=1)
        self.addCleanup(stop_vit_grpc_server, server)
        server.add_generic_rpc_handlers(
            (
                grpc.method_handlers_generic_handler(
                    "MultimodalRpcService",
                    {
                        "RemoteMultimodalEmbedding": grpc.unary_unary_rpc_method_handler(
                            embedding
                        ),
                        "ReleaseRdmaLease": grpc.unary_unary_rpc_method_handler(
                            lambda request, context: b"released"
                        ),
                    },
                ),
            )
        )
        port = server.add_insecure_port("127.0.0.1:0")
        server.start()
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            call = channel.unary_unary(
                "/MultimodalRpcService/RemoteMultimodalEmbedding"
            ).future(b"", timeout=5)
            try:
                self.assertTrue(entered.wait(2))
                self.assertEqual(
                    channel.unary_unary("/MultimodalRpcService/ReleaseRdmaLease")(
                        b"", timeout=1
                    ),
                    b"released",
                )
                self.assertFalse(call.done())
            finally:
                unblock.set()
            self.assertEqual(call.result(), b"embedding")
        executors = server._vit_executors
        stop_vit_grpc_server(server)
        for executor in executors:
            with self.assertRaises(RuntimeError):
                executor.submit(lambda: None)


if __name__ == "__main__":
    main()
