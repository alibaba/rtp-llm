import threading
from unittest import TestCase, main
from unittest.mock import MagicMock, patch

from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    MMRdmaDescPB,
    MultimodalInputsPB,
    MultimodalOutputPB,
    ReleaseEmbeddingPB,
)
from rtp_llm.server.vit_proxy_server import (
    LoadBalancer,
    VitProxyRpcServer,
    WorkerConnectionPool,
)


class LoadBalancerRoundRobinTest(TestCase):
    def test_cycles_through_workers(self):
        lb = LoadBalancer(["a", "b", "c"], strategy="round_robin")
        self.assertEqual(
            [lb.get_worker() for _ in range(6)], ["a", "b", "c", "a", "b", "c"]
        )

    def test_single_worker(self):
        lb = LoadBalancer(["only"], strategy="round_robin")
        for _ in range(5):
            self.assertEqual(lb.get_worker(), "only")

    def test_empty_workers_raises(self):
        lb = LoadBalancer([], strategy="round_robin")
        with self.assertRaises(RuntimeError):
            lb.get_worker()


class LoadBalancerLeastConnectionsTest(TestCase):
    def test_picks_worker_with_min_connections(self):
        lb = LoadBalancer(["a", "b", "c"], strategy="least_connections")
        lb.increment_connections("a")
        lb.increment_connections("a")
        lb.increment_connections("b")
        # c has 0, should be picked
        self.assertEqual(lb.get_worker(), "c")

    def test_tie_rotates_among_equals(self):
        lb = LoadBalancer(["a", "b", "c"], strategy="least_connections")
        # all zero — tie broken by rotating index
        first = lb.get_worker()
        second = lb.get_worker()
        third = lb.get_worker()
        self.assertEqual({first, second, third}, {"a", "b", "c"})

    def test_respects_updated_counts(self):
        lb = LoadBalancer(["a", "b"], strategy="least_connections")
        lb.increment_connections("a")
        # b is less loaded
        self.assertEqual(lb.get_worker(), "b")
        lb.increment_connections("b")
        lb.increment_connections("b")
        # now a (1) is less than b (2)
        self.assertEqual(lb.get_worker(), "a")


class LoadBalancerCountersTest(TestCase):
    def test_increment_decrement(self):
        lb = LoadBalancer(["a"])
        lb.increment_connections("a")
        lb.increment_connections("a")
        self.assertEqual(lb.connection_counts["a"], 2)
        lb.decrement_connections("a")
        self.assertEqual(lb.connection_counts["a"], 1)

    def test_decrement_clamped_at_zero(self):
        lb = LoadBalancer(["a"])
        lb.decrement_connections("a")
        lb.decrement_connections("a")
        self.assertEqual(lb.connection_counts["a"], 0)

    def test_decrement_unknown_address_is_noop(self):
        lb = LoadBalancer(["a"])
        lb.decrement_connections("never_seen")
        self.assertNotIn("never_seen", lb.connection_counts)


class LoadBalancerInvalidStrategyTest(TestCase):
    def test_unknown_strategy_raises(self):
        lb = LoadBalancer(["a"], strategy="bogus")
        with self.assertRaises(ValueError):
            lb.get_worker()


class LoadBalancerThreadSafetyTest(TestCase):
    def test_concurrent_round_robin_covers_all_workers(self):
        workers = [f"w{i}" for i in range(4)]
        lb = LoadBalancer(workers, strategy="round_robin")
        results = []
        lock = threading.Lock()

        def pull():
            for _ in range(100):
                w = lb.get_worker()
                with lock:
                    results.append(w)

        threads = [threading.Thread(target=pull) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(results), 800)
        # round_robin over 4 workers with 800 calls — each gets exactly 200
        for w in workers:
            self.assertEqual(results.count(w), 200)

    def test_concurrent_increment_decrement(self):
        lb = LoadBalancer(["a"])

        def inc_dec():
            for _ in range(1000):
                lb.increment_connections("a")
                lb.decrement_connections("a")

        threads = [threading.Thread(target=inc_dec) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(lb.connection_counts["a"], 0)


class WorkerConnectionPoolTest(TestCase):
    @patch("rtp_llm.server.vit_proxy_server.grpc.insecure_channel")
    @patch("rtp_llm.server.vit_proxy_server.MultimodalRpcServiceStub")
    def test_stub_created_once_per_address(self, mock_stub_cls, mock_channel):
        mock_channel.return_value = MagicMock()
        mock_stub_cls.side_effect = lambda ch: MagicMock(name="stub")

        pool = WorkerConnectionPool(["a:1", "b:2"])
        stub_a_1 = pool.get_stub("a:1")
        stub_a_2 = pool.get_stub("a:1")
        stub_b = pool.get_stub("b:2")

        self.assertIs(stub_a_1, stub_a_2)
        self.assertIsNot(stub_a_1, stub_b)
        self.assertEqual(mock_channel.call_count, 2)

    @patch("rtp_llm.server.vit_proxy_server.grpc.insecure_channel")
    @patch("rtp_llm.server.vit_proxy_server.MultimodalRpcServiceStub")
    def test_close_all_closes_each_channel(self, mock_stub_cls, mock_channel):
        channels = [MagicMock(name=f"ch{i}") for i in range(2)]
        mock_channel.side_effect = channels
        mock_stub_cls.side_effect = lambda ch: MagicMock()

        pool = WorkerConnectionPool(["a", "b"])
        pool.get_stub("a")
        pool.get_stub("b")
        pool.close_all()

        for ch in channels:
            ch.close.assert_called_once()
        self.assertEqual(pool.channels, {})
        self.assertEqual(pool.stubs, {})

    @patch("rtp_llm.server.vit_proxy_server.grpc.insecure_channel")
    @patch("rtp_llm.server.vit_proxy_server.MultimodalRpcServiceStub")
    def test_close_all_tolerates_close_errors(self, mock_stub_cls, mock_channel):
        ch = MagicMock()
        ch.close.side_effect = RuntimeError("boom")
        mock_channel.return_value = ch
        mock_stub_cls.return_value = MagicMock()

        pool = WorkerConnectionPool(["a"])
        pool.get_stub("a")
        # should not propagate
        pool.close_all()
        self.assertEqual(pool.channels, {})


class VitProxyRdmaReleaseTest(TestCase):
    @patch("rtp_llm.server.vit_proxy_server.kmonitor.init")
    @patch("rtp_llm.server.vit_proxy_server.kmonitor.report")
    def test_release_is_forwarded_to_workers_that_created_handles(
        self, _mock_report, _mock_init
    ):
        load_balancer = MagicMock()
        load_balancer.get_worker.side_effect = ["worker-a", "worker-b"]
        connection_pool = MagicMock()
        stub_a = MagicMock()
        stub_b = MagicMock()
        connection_pool.get_stub.side_effect = lambda address: {
            "worker-a": stub_a,
            "worker-b": stub_b,
        }[address]
        stub_a.RemoteMultimodalEmbedding.return_value = MultimodalOutputPB(
            output_rdma=MMRdmaDescPB(handle="handle-a")
        )
        response_b = MultimodalOutputPB()
        response_b.output_rdma_chunks.add(handle="handle-b-1")
        response_b.output_rdma_chunks.add(handle="handle-b-2")
        stub_b.RemoteMultimodalEmbedding.return_value = response_b

        servicer = VitProxyRpcServer(load_balancer, connection_pool)
        servicer.RemoteMultimodalEmbedding(MultimodalInputsPB(), MagicMock())
        servicer.RemoteMultimodalEmbedding(MultimodalInputsPB(), MagicMock())
        servicer.ReleaseMultimodalEmbedding(
            ReleaseEmbeddingPB(handle=["handle-a", "handle-b-1", "handle-b-2"]),
            MagicMock(),
        )

        request_a = stub_a.ReleaseMultimodalEmbedding.call_args.args[0]
        request_b = stub_b.ReleaseMultimodalEmbedding.call_args.args[0]
        self.assertEqual(list(request_a.handle), ["handle-a"])
        self.assertEqual(list(request_b.handle), ["handle-b-1", "handle-b-2"])
        self.assertEqual(stub_a.ReleaseMultimodalEmbedding.call_args.kwargs["timeout"], 1.0)
        self.assertEqual(stub_b.ReleaseMultimodalEmbedding.call_args.kwargs["timeout"], 1.0)


if __name__ == "__main__":
    main()
