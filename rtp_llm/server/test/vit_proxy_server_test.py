import threading
import time
from concurrent import futures
from unittest import TestCase, main
from unittest.mock import MagicMock, patch

import grpc

from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    CacheStatusPB,
    CacheVersionPB,
    MultimodalInputsPB,
    MultimodalOutputPB,
    StatusVersionPB,
    WorkerStatusPB,
)
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc import (
    MultimodalRpcServiceServicer,
    MultimodalRpcServiceStub,
    add_MultimodalRpcServiceServicer_to_server,
)
from rtp_llm.server.vit_proxy_server import (
    STATUS_CHECK_TIMEOUT_SEC,
    LoadBalancer,
    VitProxyRpcServer,
    WorkerConnectionPool,
)


class FakeContext:
    def __init__(self, time_remaining_values=None):
        self.code = None
        self.details = None
        self.time_remaining_values = list(time_remaining_values or [])

    def set_code(self, code):
        self.code = code

    def set_details(self, details):
        self.details = details

    def time_remaining(self):
        if self.time_remaining_values:
            return self.time_remaining_values.pop(0)
        return None


class RpcDeadlineExceeded(grpc.RpcError):
    def code(self):
        return grpc.StatusCode.DEADLINE_EXCEEDED

    def details(self):
        return "deadline exceeded"


class RpcUnavailable(grpc.RpcError):
    def code(self):
        return grpc.StatusCode.UNAVAILABLE

    def details(self):
        return "unavailable"


class FakeGrpcFuture:
    def __init__(self, response=None, error=None, done=True):
        self.response = response
        self.error = error
        self._done = done
        self.cancelled = False
        self.callbacks = []

    def add_done_callback(self, callback):
        self.callbacks.append(callback)
        if self._done:
            callback(self)

    def result(self, timeout=None):
        if self.error:
            raise self.error
        if not self._done:
            raise grpc.FutureTimeoutError()
        return self.response

    def done(self):
        return self._done

    def cancel(self):
        if self._done:
            return False
        self.cancelled = True
        self._done = True
        for callback in self.callbacks:
            callback(self)
        return True


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

    def test_skips_unhealthy_worker(self):
        lb = LoadBalancer(["a", "b"], strategy="round_robin")
        lb.set_worker_alive("a", False)

        self.assertEqual(lb.get_worker(), "b")

    def test_round_robin_balances_over_healthy_subset(self):
        lb = LoadBalancer(["a", "b", "c"], strategy="round_robin")
        lb.set_worker_alive("a", False)

        self.assertEqual(
            [lb.get_worker() for _ in range(6)], ["b", "c", "b", "c", "b", "c"]
        )

    def test_all_unhealthy_workers_raise(self):
        lb = LoadBalancer(["a", "b"], strategy="round_robin")
        lb.set_worker_alive("a", False)
        lb.set_worker_alive("b", False)

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
    def _make_status_stub(self, worker_status=None, error=None, done=True):
        stub = MagicMock()
        stub.GetWorkerStatus.future.return_value = FakeGrpcFuture(
            response=worker_status, error=error, done=done
        )
        return stub

    def _make_server(self, stubs):
        load_balancer = LoadBalancer(list(stubs.keys()))
        connection_pool = MagicMock()
        connection_pool.get_stub.side_effect = lambda addr: stubs[addr]
        return VitProxyRpcServer(load_balancer, connection_pool)

    def test_worker_status_is_alive_when_any_worker_is_alive(self):
        dead_stub = self._make_status_stub(
            WorkerStatusPB(role="VIT", alive=False, status_version=1)
        )
        live_stub = self._make_status_stub(
            WorkerStatusPB(role="VIT", alive=True, status_version=2)
        )
        server = self._make_server({"dead": dead_stub, "live": live_stub})
        context = FakeContext()

        response = server.GetWorkerStatus(StatusVersionPB(), context)

        self.assertEqual(response.role, "VIT")
        self.assertTrue(response.alive)
        self.assertEqual(context.code, None)

    def test_worker_status_continues_after_timeout_with_parallel_probe(self):
        timeout_stub = self._make_status_stub(error=RpcDeadlineExceeded())
        live_stub = self._make_status_stub(
            WorkerStatusPB(role="", alive=True, status_version=2)
        )
        server = self._make_server({"timeout": timeout_stub, "live": live_stub})
        context = FakeContext(time_remaining_values=[1.0])

        response = server.GetWorkerStatus(StatusVersionPB(), context)

        self.assertTrue(response.alive)
        self.assertEqual(response.role, "VIT")
        self.assertEqual(context.code, None)
        first_timeout = timeout_stub.GetWorkerStatus.future.call_args.kwargs["timeout"]
        second_timeout = live_stub.GetWorkerStatus.future.call_args.kwargs["timeout"]
        self.assertGreater(first_timeout, 0)
        self.assertLessEqual(first_timeout, STATUS_CHECK_TIMEOUT_SEC)
        self.assertGreater(second_timeout, 0)
        self.assertLessEqual(second_timeout, STATUS_CHECK_TIMEOUT_SEC)

    @patch("rtp_llm.server.vit_proxy_server.STATUS_CHECK_TIMEOUT_SEC", 0.01)
    def test_worker_status_parallel_probe_reaches_late_live_worker(self):
        stubs = {}
        pending_futures = []
        for i in range(8):
            pending_future = FakeGrpcFuture(done=False)
            pending_futures.append(pending_future)
            stub = MagicMock()
            stub.GetWorkerStatus.future.return_value = pending_future
            stubs[f"slow-{i}"] = stub
        live_stub = self._make_status_stub(
            WorkerStatusPB(role="", alive=True, status_version=3)
        )
        stubs["live"] = live_stub
        server = self._make_server(stubs)
        context = FakeContext(time_remaining_values=[1.0])

        response = server.GetWorkerStatus(StatusVersionPB(), context)

        self.assertTrue(response.alive)
        self.assertEqual(response.role, "VIT")
        self.assertEqual(context.code, None)
        live_stub.GetWorkerStatus.future.assert_called_once()
        for pending_future in pending_futures:
            self.assertTrue(pending_future.cancelled)
        self.assertEqual(server.load_balancer.get_alive_worker_addresses(), ["live"])

    @patch("rtp_llm.server.vit_proxy_server.STATUS_CHECK_TIMEOUT_SEC", 0.01)
    def test_worker_status_marks_slow_workers_unhealthy_after_full_probe_deadline(
        self,
    ):
        slow_future = FakeGrpcFuture(done=False)
        slow_stub = MagicMock()
        slow_stub.GetWorkerStatus.future.return_value = slow_future
        live_stub = self._make_status_stub(
            WorkerStatusPB(role="", alive=True, status_version=3)
        )
        server = self._make_server({"slow": slow_stub, "live": live_stub})
        context = FakeContext(time_remaining_values=[10.0])

        start_time = time.monotonic()
        response = server.GetWorkerStatus(StatusVersionPB(), context)
        elapsed_s = time.monotonic() - start_time

        self.assertTrue(response.alive)
        self.assertLess(elapsed_s, 0.2)
        self.assertTrue(slow_future.cancelled)
        self.assertEqual(context.code, None)
        self.assertEqual(server.load_balancer.get_alive_worker_addresses(), ["live"])

    def test_worker_status_checks_workers_marked_unhealthy(self):
        dead_stub = self._make_status_stub(
            WorkerStatusPB(role="VIT", alive=False, status_version=1)
        )
        live_stub = self._make_status_stub(
            WorkerStatusPB(role="VIT", alive=True, status_version=2)
        )
        server = self._make_server({"dead": dead_stub, "live": live_stub})
        server.load_balancer.set_worker_alive("dead", False)
        server.load_balancer.set_worker_alive("live", False)

        response = server.GetWorkerStatus(StatusVersionPB(), FakeContext())

        self.assertTrue(response.alive)
        dead_stub.GetWorkerStatus.future.assert_called_once()
        live_stub.GetWorkerStatus.future.assert_called_once()
        self.assertEqual(server.load_balancer.get_alive_worker_addresses(), ["live"])

    def test_worker_status_updates_all_completed_workers_before_returning(self):
        first_stub = self._make_status_stub(
            WorkerStatusPB(role="VIT", alive=True, status_version=1)
        )
        recovered_stub = self._make_status_stub(
            WorkerStatusPB(role="VIT", alive=True, status_version=2)
        )
        server = self._make_server({"first": first_stub, "recovered": recovered_stub})
        server.load_balancer.set_worker_alive("first", False)
        server.load_balancer.set_worker_alive("recovered", False)

        response = server.GetWorkerStatus(StatusVersionPB(), FakeContext())

        self.assertTrue(response.alive)
        first_stub.GetWorkerStatus.future.assert_called_once()
        recovered_stub.GetWorkerStatus.future.assert_called_once()
        self.assertEqual(
            server.load_balancer.get_alive_worker_addresses(), ["first", "recovered"]
        )

    def test_worker_status_is_unavailable_when_no_worker_is_alive(self):
        stub = self._make_status_stub(
            WorkerStatusPB(role="VIT", alive=False, status_version=1)
        )
        server = self._make_server({"dead": stub})
        context = FakeContext()

        response = server.GetWorkerStatus(StatusVersionPB(), context)

        self.assertFalse(response.alive)
        self.assertEqual(context.code, grpc.StatusCode.UNAVAILABLE)

    def test_cache_status_is_ok_when_any_worker_is_alive(self):
        stub = self._make_status_stub(
            WorkerStatusPB(role="VIT", alive=True, status_version=1)
        )
        server = self._make_server({"live": stub})
        context = FakeContext()

        response = server.GetCacheStatus(CacheVersionPB(), context)

        self.assertIsInstance(response, CacheStatusPB)
        self.assertEqual(context.code, None)

    def test_cache_status_is_unavailable_when_no_worker_is_alive(self):
        stub = self._make_status_stub(
            WorkerStatusPB(role="VIT", alive=False, status_version=1)
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


class VitProxyRpcServerForwardingTest(TestCase):
    def _make_status_stub(self, worker_status=None, error=None, done=True):
        stub = MagicMock()
        stub.GetWorkerStatus.future.return_value = FakeGrpcFuture(
            response=worker_status, error=error, done=done
        )
        return stub

    def _make_server(self, stubs):
        load_balancer = LoadBalancer(list(stubs.keys()))
        connection_pool = MagicMock()
        connection_pool.get_stub.side_effect = lambda addr: stubs[addr]
        return VitProxyRpcServer(load_balancer, connection_pool)

    def test_status_health_is_used_by_forwarding(self):
        dead_stub = self._make_status_stub(
            WorkerStatusPB(role="VIT", alive=False, status_version=1)
        )
        live_stub = self._make_status_stub(
            WorkerStatusPB(role="VIT", alive=True, status_version=2)
        )
        dead_stub.RemoteMultimodalEmbedding.side_effect = AssertionError(
            "unhealthy worker should not receive forwarded requests"
        )
        live_stub.RemoteMultimodalEmbedding.return_value = MultimodalOutputPB()
        server = self._make_server({"dead": dead_stub, "live": live_stub})

        status = server.GetWorkerStatus(StatusVersionPB(), FakeContext())
        response = server.RemoteMultimodalEmbedding(MultimodalInputsPB(), FakeContext())

        self.assertTrue(status.alive)
        self.assertIsInstance(response, MultimodalOutputPB)
        dead_stub.RemoteMultimodalEmbedding.assert_not_called()
        live_stub.RemoteMultimodalEmbedding.assert_called_once()

    def test_forwarding_retry_marks_failed_worker_unhealthy(self):
        failed_stub = MagicMock()
        failed_stub.RemoteMultimodalEmbedding.side_effect = RpcUnavailable()
        live_stub = MagicMock()
        live_stub.RemoteMultimodalEmbedding.return_value = MultimodalOutputPB()
        server = self._make_server({"failed": failed_stub, "live": live_stub})

        response = server.RemoteMultimodalEmbedding(MultimodalInputsPB(), FakeContext())

        self.assertIsInstance(response, MultimodalOutputPB)
        failed_stub.RemoteMultimodalEmbedding.assert_called_once()
        live_stub.RemoteMultimodalEmbedding.assert_called_once()
        self.assertEqual(server.load_balancer.get_alive_worker_addresses(), ["live"])

    def test_forwarding_uses_latest_status_probe_result(self):
        failed_stub = self._make_status_stub(
            WorkerStatusPB(role="VIT", alive=False, status_version=1)
        )
        recovered_stub = self._make_status_stub(
            WorkerStatusPB(role="VIT", alive=True, status_version=2)
        )
        recovered_stub.RemoteMultimodalEmbedding.return_value = MultimodalOutputPB()
        server = self._make_server({"failed": failed_stub, "recovered": recovered_stub})
        server.load_balancer.set_worker_alive("failed", False)
        server.load_balancer.set_worker_alive("recovered", False)

        context = FakeContext()
        with self.assertRaises(RuntimeError):
            server.RemoteMultimodalEmbedding(MultimodalInputsPB(), context)
        self.assertEqual(context.code, grpc.StatusCode.UNAVAILABLE)
        self.assertIn("No healthy VIT worker behind proxy", context.details)
        failed_stub.RemoteMultimodalEmbedding.assert_not_called()
        recovered_stub.RemoteMultimodalEmbedding.assert_not_called()

        status = server.GetWorkerStatus(StatusVersionPB(), FakeContext())
        response = server.RemoteMultimodalEmbedding(MultimodalInputsPB(), FakeContext())

        self.assertTrue(status.alive)
        self.assertIsInstance(response, MultimodalOutputPB)
        recovered_stub.RemoteMultimodalEmbedding.assert_called_once()
        self.assertEqual(
            server.load_balancer.get_alive_worker_addresses(), ["recovered"]
        )

    def test_forwarding_without_healthy_worker_returns_unavailable_grpc_status(self):
        worker_address = "dead"
        load_balancer = LoadBalancer([worker_address])
        load_balancer.set_worker_alive(worker_address, False)
        connection_pool = MagicMock()
        server = VitProxyRpcServer(load_balancer, connection_pool)
        grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
        add_MultimodalRpcServiceServicer_to_server(server, grpc_server)
        port = grpc_server.add_insecure_port("127.0.0.1:0")
        grpc_server.start()
        channel = grpc.insecure_channel(f"127.0.0.1:{port}")
        stub = MultimodalRpcServiceStub(channel)

        try:
            with self.assertRaises(grpc.RpcError) as error:
                stub.RemoteMultimodalEmbedding(MultimodalInputsPB(), timeout=1)
        finally:
            channel.close()
            grpc_server.stop(0)

        self.assertEqual(error.exception.code(), grpc.StatusCode.UNAVAILABLE)
        self.assertIn("No healthy VIT worker behind proxy", error.exception.details())
        connection_pool.get_stub.assert_not_called()

    def test_forwarding_non_rpc_exception_does_not_retry_or_mark_unhealthy(self):
        failed_stub = MagicMock()
        failed_stub.RemoteMultimodalEmbedding.side_effect = ValueError("bad request")
        live_stub = MagicMock()
        live_stub.RemoteMultimodalEmbedding.return_value = MultimodalOutputPB()
        server = self._make_server({"failed": failed_stub, "live": live_stub})

        with self.assertRaises(ValueError):
            server.RemoteMultimodalEmbedding(MultimodalInputsPB(), FakeContext())

        failed_stub.RemoteMultimodalEmbedding.assert_called_once()
        live_stub.RemoteMultimodalEmbedding.assert_not_called()
        self.assertEqual(
            server.load_balancer.get_alive_worker_addresses(), ["failed", "live"]
        )

    def test_forwarding_deadline_does_not_retry_or_mark_unhealthy(self):
        timeout_stub = MagicMock()
        timeout_stub.RemoteMultimodalEmbedding.side_effect = RpcDeadlineExceeded()
        live_stub = MagicMock()
        live_stub.RemoteMultimodalEmbedding.return_value = MultimodalOutputPB()
        server = self._make_server({"timeout": timeout_stub, "live": live_stub})

        with self.assertRaises(RpcDeadlineExceeded):
            server.RemoteMultimodalEmbedding(MultimodalInputsPB(), FakeContext())

        timeout_stub.RemoteMultimodalEmbedding.assert_called_once()
        live_stub.RemoteMultimodalEmbedding.assert_not_called()
        self.assertEqual(
            server.load_balancer.get_alive_worker_addresses(), ["timeout", "live"]
        )


if __name__ == "__main__":
    main()
