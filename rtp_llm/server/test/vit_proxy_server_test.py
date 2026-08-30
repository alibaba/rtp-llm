import threading
import time
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import ANY, MagicMock, patch

import grpc

from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    MMRdmaDescPB,
    MultimodalInputsPB,
    MultimodalOutputPB,
    ReleaseEmbeddingPB,
)
from rtp_llm.metrics.kmonitor_metric_reporter import AccMetrics
from rtp_llm.server.vit_proxy_server import (
    LoadBalancer,
    VitProxyRpcServer,
    WorkerConnectionPool,
    _resolve_rpc_timeout_seconds,
)
from rtp_llm.server.vit_rpc_server import MultimodalRpcServer


class VitErrorQpsTest(TestCase):
    @patch("rtp_llm.server.vit_proxy_server.kmonitor.report")
    def test_wait_greennet_worker_rpc_error_reports_error_qps(self, report):
        class WorkerRpcError(grpc.RpcError):
            def code(self):
                return grpc.StatusCode.UNAVAILABLE

            def details(self):
                return "worker unavailable"

            def trailing_metadata(self):
                return ()

        load_balancer = MagicMock()
        load_balancer.get_worker.return_value = "worker-a"
        load_balancer.decrement_connections.side_effect = RuntimeError("counter failed")
        connection_pool = MagicMock()
        stub = MagicMock()
        connection_pool.get_stub.return_value = stub
        worker_call = MagicMock()
        worker_call.result.side_effect = WorkerRpcError()
        stub.WaitGreenNetVerdict.future.return_value = worker_call
        context = MagicMock()
        context.add_callback.return_value = True
        context.abort.side_effect = RuntimeError("proxy aborted")

        servicer = VitProxyRpcServer(load_balancer, connection_pool)
        with self.assertRaises(RuntimeError):
            servicer.WaitGreenNetVerdict(MultimodalInputsPB(), context)

        error_reports = [
            call
            for call in report.call_args_list
            if call.args and call.args[0] == AccMetrics.VIT_ERROR_QPS_METRIC
        ]
        self.assertEqual(len(error_reports), 1)
        worker_call.cancel.assert_not_called()

    @patch("rtp_llm.server.vit_rpc_server.trans_mm_input", return_value=[])
    def test_worker_rpc_failed_verdict_is_reported(self, _trans_mm_input):
        engine = MagicMock()
        verdict = SimpleNamespace(passed=False, code=2, message="unsafe")
        engine.wait_greennet_verdict.return_value = verdict
        servicer = MultimodalRpcServer.__new__(MultimodalRpcServer)
        servicer.engine = engine
        servicer._rdma = None
        context = MagicMock()
        context.add_callback.return_value = True

        servicer.WaitGreenNetVerdict(MultimodalInputsPB(), context)

        engine.report_vit_error.assert_called_once_with(verdict)
        self.assertEqual(
            context.set_code.call_args.args[0], grpc.StatusCode.PERMISSION_DENIED
        )

    @patch("rtp_llm.server.vit_proxy_server.kmonitor.report")
    def test_rdma_release_worker_error_reports_error_qps(self, report):
        load_balancer = MagicMock()
        connection_pool = MagicMock()
        connection_pool.get_stub.side_effect = RuntimeError("release failed")
        servicer = VitProxyRpcServer(load_balancer, connection_pool)
        servicer._rdma_handle_routes["h"] = ("worker-a", time.monotonic())

        servicer.ReleaseMultimodalEmbedding(
            ReleaseEmbeddingPB(handle=["h"]), MagicMock()
        )

        error_reports = [
            call
            for call in report.call_args_list
            if call.args and call.args[0] == AccMetrics.VIT_ERROR_QPS_METRIC
        ]
        self.assertEqual(len(error_reports), 1)

    @patch("rtp_llm.server.vit_proxy_server.kmonitor.report")
    def test_connection_cleanup_error_reports_error_qps(self, report):
        load_balancer = MagicMock()
        load_balancer.get_worker.return_value = "worker-a"
        load_balancer.decrement_connections.side_effect = RuntimeError("counter failed")
        connection_pool = MagicMock()
        stub = MagicMock()
        worker_call = MagicMock()
        worker_call.result.return_value = MultimodalOutputPB()
        stub.RemoteMultimodalEmbedding.future.return_value = worker_call
        connection_pool.get_stub.return_value = stub
        context = MagicMock()
        context.add_callback.return_value = True

        servicer = VitProxyRpcServer(load_balancer, connection_pool)
        self.assertIsInstance(
            servicer.RemoteMultimodalEmbedding(MultimodalInputsPB(), context),
            MultimodalOutputPB,
        )

        error_reports = [
            call
            for call in report.call_args_list
            if call.args and call.args[0] == AccMetrics.VIT_ERROR_QPS_METRIC
        ]
        self.assertEqual(len(error_reports), 1)


class VitWorkerRequestIdTest(TestCase):
    @patch("rtp_llm.server.vit_rpc_server.trans_mm_input")
    def test_request_id_is_forwarded_to_all_engine_entrypoints(self, trans_mm_input):
        converted = [MagicMock(name="mm_input")]
        trans_mm_input.return_value = converted
        engine = MagicMock()
        engine.wait_greennet_verdict.return_value = SimpleNamespace(passed=True)
        engine.get_embedding_result.return_value = []
        servicer = MultimodalRpcServer.__new__(MultimodalRpcServer)
        servicer.engine = engine
        servicer._rdma = None
        request = MultimodalInputsPB(request_id=987654321)
        async_context = MagicMock()
        wait_context = MagicMock()
        remote_context = MagicMock()
        wait_context.add_callback.return_value = True
        remote_context.add_callback.return_value = True

        servicer.AsyncSubmitEmbedding(request, async_context)
        servicer.WaitGreenNetVerdict(request, wait_context)
        servicer.RemoteMultimodalEmbedding(request, remote_context)

        engine.async_submit.assert_called_once_with(converted, 987654321)
        engine.wait_greennet_verdict.assert_called_once_with(
            converted, request_id=987654321, cancellation_event=ANY
        )
        engine.get_embedding_result.assert_called_once_with(
            converted, request_id=987654321, cancellation_event=ANY
        )

        wait_context.add_callback.call_args.args[0]()
        remote_context.add_callback.call_args.args[0]()
        self.assertEqual(engine.cancel_queued_request.call_count, 2)
        for cancel_call in engine.cancel_queued_request.call_args_list:
            self.assertEqual(cancel_call.args, (987654321,))


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


class RpcTimeoutTest(TestCase):
    def test_uses_configured_default_when_request_timeout_is_unset(self):
        request = MultimodalInputsPB()
        request.multimodal_inputs.add().mm_preprocess_config.mm_timeout_ms = -1

        self.assertEqual(_resolve_rpc_timeout_seconds(request, 123.0), 123.0)

    def test_uses_largest_positive_request_timeout(self):
        request = MultimodalInputsPB()
        request.multimodal_inputs.add().mm_preprocess_config.mm_timeout_ms = 2000
        request.multimodal_inputs.add().mm_preprocess_config.mm_timeout_ms = 3500

        self.assertEqual(_resolve_rpc_timeout_seconds(request, 123.0), 3.5)


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
        call_a = MagicMock()
        call_a.result.return_value = MultimodalOutputPB(
            output_rdma=MMRdmaDescPB(handle="handle-a")
        )
        stub_a.RemoteMultimodalEmbedding.future.return_value = call_a
        response_b = MultimodalOutputPB()
        response_b.output_rdma_chunks.add(handle="handle-b-1")
        response_b.output_rdma_chunks.add(handle="handle-b-2")
        call_b = MagicMock()
        call_b.result.return_value = response_b
        stub_b.RemoteMultimodalEmbedding.future.return_value = call_b

        servicer = VitProxyRpcServer(load_balancer, connection_pool)
        context_a = MagicMock()
        context_a.add_callback.return_value = True
        context_b = MagicMock()
        context_b.add_callback.return_value = True
        servicer.RemoteMultimodalEmbedding(MultimodalInputsPB(), context_a)
        servicer.RemoteMultimodalEmbedding(MultimodalInputsPB(), context_b)
        servicer.ReleaseMultimodalEmbedding(
            ReleaseEmbeddingPB(handle=["handle-a", "handle-b-1", "handle-b-2"]),
            MagicMock(),
        )

        request_a = stub_a.ReleaseMultimodalEmbedding.call_args.args[0]
        request_b = stub_b.ReleaseMultimodalEmbedding.call_args.args[0]
        self.assertEqual(list(request_a.handle), ["handle-a"])
        self.assertEqual(list(request_b.handle), ["handle-b-1", "handle-b-2"])
        self.assertEqual(
            stub_a.ReleaseMultimodalEmbedding.call_args.kwargs["timeout"], 1.0
        )
        self.assertEqual(
            stub_b.ReleaseMultimodalEmbedding.call_args.kwargs["timeout"], 1.0
        )


class VitProxyCancellationTest(TestCase):
    @patch("rtp_llm.server.vit_proxy_server.kmonitor.init")
    @patch("rtp_llm.server.vit_proxy_server.kmonitor.report")
    def test_parent_rpc_cancellation_cancels_worker_rpc(self, _mock_report, _mock_init):
        load_balancer = MagicMock()
        load_balancer.get_worker.return_value = "worker-a"
        connection_pool = MagicMock()
        stub = MagicMock()
        connection_pool.get_stub.return_value = stub
        worker_call = MagicMock()
        worker_call.result.return_value = MultimodalOutputPB()
        stub.RemoteMultimodalEmbedding.future.return_value = worker_call
        context = MagicMock()
        context.add_callback.return_value = True

        servicer = VitProxyRpcServer(load_balancer, connection_pool)
        servicer.RemoteMultimodalEmbedding(MultimodalInputsPB(), context)

        cancel_callback = context.add_callback.call_args.args[0]
        cancel_callback()
        worker_call.cancel.assert_called_once_with()


if __name__ == "__main__":
    main()
