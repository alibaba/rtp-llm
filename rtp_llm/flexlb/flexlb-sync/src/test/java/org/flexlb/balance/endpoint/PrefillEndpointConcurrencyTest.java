package org.flexlb.balance.endpoint;

import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.scheduler.RequestContext;
import org.flexlb.balance.scheduler.RequestRoute;
import org.flexlb.balance.scheduler.SchedulingTestConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/** Real resource locks and ledgers; only the predictor and notification sinks are substituted. */
@Timeout(20)
class PrefillEndpointConcurrencyTest {
    enum Change {
        REMOVE_ONE,
        REMOVE_ALL,
        REPLACE_REQUEST_ID,
        RETIRE
    }

    @ParameterizedTest
    @EnumSource(Change.class)
    void statusCommitRechecksConcurrentCleanupAndNeverRestoresOldOwnership(Change change)
            throws Exception {
        var config = SchedulingTestConfig.newConfig();
        var runtime = EndpointTestSupport.requestRuntime();
        var endpoint =
                EndpointTestSupport.prefill(
                        EndpointTestSupport.workerStatus(RoleType.PREFILL, "127.0.0.1", 8080, 8090),
                        config,
                        EndpointTestSupport.routeStrategy(runtime),
                        org.flexlb.balance.scheduler.SchedulerTestSupport.repository(
                                runtime.events()),
                        mock(DeliveryMetricsReporter.class));
        var first = route(endpoint, config, 1L);
        var terminal = route(endpoint, config, 2L);
        var removed = route(endpoint, config, 3L);
        commit(endpoint, 1L, 100L, List.of(first, terminal, removed));
        var predictionEntered = new CountDownLatch(1);
        var resumePrediction = new CountDownLatch(1);
        var calls = new AtomicInteger();
        var evaluator = mock(PrefillTimePredictor.Evaluator.class);
        when(evaluator.predictBatchMs(any()))
                .thenAnswer(
                        call -> {
                            assertFalse(endpoint.resourceLock().isHeldByCurrentThread());
                            int attempt = calls.incrementAndGet();
                            assertTrue(attempt <= 2, "membership stabilization must terminate");
                            if (attempt == 1) {
                                predictionEntered.countDown();
                                assertTrue(resumePrediction.await(5, TimeUnit.SECONDS));
                                return 99.0;
                            }
                            return 55.0;
                        });
        var predictor = mock(PrefillTimePredictor.class);
        when(predictor.evaluator()).thenReturn(evaluator);
        ReflectionTestUtils.setField(endpoint, "predictor", predictor);
        // Daemon workers and bounded Future.get make a lock-order regression fail, not hang Maven.
        var workers = Executors.newFixedThreadPool(3, Thread.ofPlatform().daemon().factory());
        try {
            var reduction =
                    workers.submit(
                            () -> {
                                var task = new TaskInfo();
                                task.setRequestId(String.valueOf(2L));
                                task.setBatchId(1L);
                                task.setErrorCode(500);
                                task.setExecutionTimeMs(0L);
                                var response = new WorkerStatusResponse();
                                response.setFinishedTaskInfo(Map.of("2", task));
                                response.setRunningTaskInfo(Map.of());
                                EndpointTestSupport.applyStatus(endpoint, response).run();
                            });
            assertTrue(predictionEntered.await(5, TimeUnit.SECONDS));
            var mutation =
                    workers.submit(
                            () -> {
                                // Opposite actor holds Context while requesting Endpoint.
                                // Prediction must not
                                // keep Endpoint locked or acquire Context from inside that lock.
                                synchronized (removed.ctx()) {
                                    if (change == Change.RETIRE) {
                                        endpoint.close();
                                    } else {
                                        assertTrue(endpoint.releaseRequest(removed));
                                        if (change != Change.REMOVE_ONE)
                                            assertTrue(endpoint.releaseRequest(first));
                                        if (change == Change.REPLACE_REQUEST_ID) {
                                            commit(
                                                    endpoint,
                                                    2L,
                                                    200L,
                                                    List.of(route(endpoint, config, 1L)));
                                        }
                                    }
                                }
                            });
            mutation.get(5, TimeUnit.SECONDS);
            var read = workers.submit(endpoint::stats);
            assertTrue(read.get(5, TimeUnit.SECONDS).locallyOwnedRequests() >= 0);
            assertFalse(reduction.isDone(), "status reduction remains paused in the predictor");
            resumePrediction.countDown();
            reduction.get(5, TimeUnit.SECONDS);
            var work = endpoint.captureRouteProjectionInputs().work();
            long expected =
                    switch (change) {
                        case REMOVE_ONE -> 55L;
                        case REPLACE_REQUEST_ID -> 200L;
                        default -> 0L;
                    };
            assertEquals(expected, work.totalRemainingWorkMs().orElseThrow());
            assertEquals(change == Change.REMOVE_ONE ? 2 : 1, calls.get());
            assertFalse(work.containsRequest("2"));
            assertFalse(work.containsRequest("3"));
            assertEquals(expected != 0, work.containsRequest("1"));
            assertFalse(endpoint.releaseRequest(removed), "late old cleanup must be harmless");
            if (change == Change.REPLACE_REQUEST_ID) {
                assertFalse(
                        endpoint.releaseRequest(first), "same request id is not the same owner");
                assertEquals(
                        200L,
                        endpoint.captureRouteProjectionInputs()
                                .work()
                                .totalRemainingWorkMs()
                                .orElseThrow());
            }
            if (change == Change.RETIRE) assertTrue(endpoint.isRetired());
        } finally {
            resumePrediction.countDown();
            workers.shutdownNow();
            // Never enter endpoint.close if a failed test left a worker holding its lock.
            if (workers.awaitTermination(5, TimeUnit.SECONDS)) endpoint.close();
            else fail("concurrent endpoint actors did not terminate");
        }
        assertEquals(0, endpoint.stats().locallyOwnedRequests());
        assertEquals(0, endpoint.stats().batchCount());
    }

    private static RequestRoute route(PrefillEndpoint endpoint, FlexlbConfig config, long id) {
        var ctx = new RequestContext(config);
        ctx.data().setRequest(Request.builder().requestId(String.valueOf(id)).seqLen(100L).build());
        var server = new ServerStatus();
        server.setRole(RoleType.PREFILL);
        server.setServerIp("127.0.0.1");
        server.setHttpPort(8080);
        server.setGrpcPort(8090);
        return SchedulingTestConfig.createRoute(
                ctx, null, server, null, endpoint, null, null, System.currentTimeMillis());
    }

    private static void commit(
            PrefillEndpoint endpoint, long id, long ms, List<RequestRoute> routes) {
        for (var route : routes) assertTrue(EndpointTestSupport.offer(endpoint, route));
        try (var ignored = EndpointTestSupport.commitBatch(endpoint, id, ms, routes)) {
            // Close the temporary pin; leave real resource ownership for the concurrent actors.
        }
    }
}
