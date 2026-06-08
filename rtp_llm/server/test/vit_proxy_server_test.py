import threading
from concurrent import futures
from unittest import TestCase, main
from unittest.mock import MagicMock, patch

import grpc

from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    CacheStatusPB,
    CacheVersionPB,
    StatusVersionPB,
    WorkerStatusPB,
)
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc import (
    MultimodalRpcServiceServicer,
    add_MultimodalRpcServiceServicer_to_server,
)
from rtp_llm.server.vit_proxy_server import (
    LoadBalancer,
    VitProxyRpcServer,
    WorkerConnectionPool,
)


class FakeContext:
    def __init__(self):
        self.code = None
        self.details = None

    def set_code(self, code):
        self.code = code

    def set_details(self, details):
        self.details = details


class StatusOnlyVitWorker(MultimodalRpcServiceServicer):
    def __init__(self, status):
        self.status = status
        self.requests = []

    def GetWorkerStatus(self, request, context):
        self.requests.append(request)
        return self.status


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


class VitProxyRpcServerStatusTest(TestCase):
    def _make_server(self, stubs):
        load_balancer = LoadBalancer(list(stubs.keys()))
        connection_pool = MagicMock()
        connection_pool.get_stub.side_effect = lambda addr: stubs[addr]
        return VitProxyRpcServer(load_balancer, connection_pool)

    def test_worker_status_is_alive_when_any_worker_is_alive(self):
        dead_stub = MagicMock()
        dead_stub.GetWorkerStatus.return_value = WorkerStatusPB(
            role="VIT", alive=False, status_version=1
        )
        live_stub = MagicMock()
        live_stub.GetWorkerStatus.return_value = WorkerStatusPB(
            role="VIT", alive=True, status_version=2
        )
        server = self._make_server({"dead": dead_stub, "live": live_stub})
        context = FakeContext()

        response = server.GetWorkerStatus(StatusVersionPB(), context)

        self.assertEqual(response.role, "VIT")
        self.assertTrue(response.alive)
        self.assertEqual(context.code, None)

    def test_worker_status_is_unavailable_when_no_worker_is_alive(self):
        stub = MagicMock()
        stub.GetWorkerStatus.return_value = WorkerStatusPB(
            role="VIT", alive=False, status_version=1
        )
        server = self._make_server({"dead": stub})
        context = FakeContext()

        response = server.GetWorkerStatus(StatusVersionPB(), context)

        self.assertFalse(response.alive)
        self.assertEqual(context.code, grpc.StatusCode.UNAVAILABLE)

    def test_cache_status_is_ok_when_any_worker_is_alive(self):
        stub = MagicMock()
        stub.GetWorkerStatus.return_value = WorkerStatusPB(
            role="VIT", alive=True, status_version=1
        )
        server = self._make_server({"live": stub})
        context = FakeContext()

        response = server.GetCacheStatus(CacheVersionPB(), context)

        self.assertIsInstance(response, CacheStatusPB)
        self.assertEqual(context.code, None)

    def test_cache_status_is_unavailable_when_no_worker_is_alive(self):
        stub = MagicMock()
        stub.GetWorkerStatus.return_value = WorkerStatusPB(
            role="VIT", alive=False, status_version=1
        )
        server = self._make_server({"dead": stub})
        context = FakeContext()

        response = server.GetCacheStatus(CacheVersionPB(), context)

        self.assertIsInstance(response, CacheStatusPB)
        self.assertEqual(context.code, grpc.StatusCode.UNAVAILABLE)

    def test_status_check_works_with_generated_grpc_stub_timeout(self):
        worker = StatusOnlyVitWorker(
            WorkerStatusPB(role="", alive=True, status_version=3)
        )
        grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
        add_MultimodalRpcServiceServicer_to_server(worker, grpc_server)
        port = grpc_server.add_insecure_port("127.0.0.1:0")
        grpc_server.start()
        worker_address = f"127.0.0.1:{port}"
        connection_pool = WorkerConnectionPool([worker_address])
        server = VitProxyRpcServer(LoadBalancer([worker_address]), connection_pool)
        context = FakeContext()

        try:
            response = server.GetWorkerStatus(
                StatusVersionPB(latest_cache_version=11), context
            )
        finally:
            connection_pool.close_all()
            grpc_server.stop(0)

        self.assertTrue(response.alive)
        self.assertEqual(response.role, "VIT")
        self.assertEqual(context.code, None)
        self.assertEqual(worker.requests[0].latest_cache_version, 11)


if __name__ == "__main__":
    main()
