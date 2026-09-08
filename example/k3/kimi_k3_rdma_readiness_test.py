import unittest

from kimi_k3_rdma_readiness import initialized_rdma_ranks


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



if __name__ == "__main__":
    unittest.main()
