import pathlib
import tempfile
import unittest
from unittest import mock

from kimi_k3_rdma_readiness import initialized_rdma_ranks, main


class RdmaReadinessTest(unittest.TestCase):
    def line(self, rank):
        return (
            f"[INFO] [RANK {rank}][host][RdmaMessager.cpp:59] "
            "rdma messager init success, server port 27190, io thread count 2, "
            "worker thread count 32, rdma server port 27192, rdma io thread count 4\n"
        )

    def test_rank_zero_health_and_duplicate_messages_do_not_prove_tp8_ready(self):
        log = self.line(0) * 8 + "[RANK 7] rdma listen port is [27255] rdma_mode is [1]"
        self.assertEqual(initialized_rdma_ranks(log), {0})

    def test_requires_successful_transport_initialization_for_each_rank(self):
        for ranks in (1, 2, 4, 8):
            with self.subTest(ranks=ranks):
                log = "".join(self.line(rank) for rank in range(ranks - 1))
                self.assertNotEqual(initialized_rdma_ranks(log), set(range(ranks)))
                self.assertEqual(
                    initialized_rdma_ranks(log + self.line(ranks - 1)),
                    set(range(ranks)),
                )

    def test_rdma_ready_does_not_hide_a_late_decode_rpc_listener(self):
        def rpc_line(rank):
            return (
                f"[RANK {rank}][RtpLLMOp.cc:398] Server listening on "
                f"0.0.0.0:{32489 + rank * 9}\n"
            )

        log = "".join(self.line(rank) for rank in range(8))
        log += "".join(rpc_line(rank) for rank in range(8) if rank != 3)
        # A configured address and the HTTP listener do not prove gRPC ready.
        log += (
            "[RANK 3][RemoteRpcServer.cc:56] worker grpc address is 11.163.39.111:32516,\n"
            "[RANK 3][RtpLLMOp.cc:413] normal HTTP Server listening on tcp:0.0.0.0:32520\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "engine.log"
            with mock.patch("sys.argv", ["readiness", str(path), "--ranks", "8"]):
                path.write_text(log)
                self.assertEqual(main(), 1)
                path.write_text(log + rpc_line(3))
                self.assertEqual(main(), 0)

if __name__ == "__main__":
    unittest.main()
