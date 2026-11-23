package org.flexlb.balance.endpoint;

import org.flexlb.balance.delivery.DeliveryStrategy;
import org.flexlb.balance.scheduler.ScheduledRequest;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.AdditionalAnswers.delegatesTo;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;

/**
 * The configured limit, real worker publication lock and canonical request ledger together.
 */
class WorkerBatcherRequestCapacityTest {
    @ParameterizedTest
    @CsvSource({"1,1", "8,2"})
    void batchFallbackPreparesClientRouteAndCanBeCancelled(int tokens, int beams) {
        FlexlbConfig config = productionConfig();
        try (Fixture fixture = fixture(config)) {
            Request request = new Request();
            request.setRequestId("fallback");
            request.setSeqLen(10L);
            request.setMaxNewTokens(tokens);
            request.setNumBeams(beams);
            BalanceContext context = new BalanceContext(config);
            context.setRequest(request);
            context.setSchedulingMetadata(SchedulingMetadata.explicit(50, Long.MAX_VALUE));
            context.setGenerateInputPb(
                    org.flexlb.engine.grpc.EngineRpcService.GenerateInputPB.newBuilder()
                            .setGenerateConfig(
                                    org.flexlb.engine.grpc.EngineRpcService.GenerateConfigPB
                                            .newBuilder()
                                            .setCanUsePdSeparation(true)
                                            .setMaxNewTokens(tokens)
                                            .setNumBeams(beams))
                            .build()
                            .toByteString());
            ScheduledRequest item =
                    new ScheduledRequest(
                            context,
                            new CompletableFuture<Response>(),
                            null,
                            null,
                            null,
                            fixture.endpoint,
                            null,
                            null,
                            System.currentTimeMillis());
            assertTrue(EndpointTestSupport.offer(fixture.endpoint, item));
            var strategy =
                    new org.flexlb.balance.scheduler.BatchDeliveryStrategy(
                            () -> {
                                throw new AssertionError("fallback must not submit a batch");
                            },
                            () -> 99L,
                            fixture.runtime.requests(),
                            mock(org.flexlb.balance.delivery.DeliveryMetrics.class));
            try (var transaction =
                    strategy.prepare(
                            List.of(item),
                            new org.flexlb.balance.prediction.FormulaPredictor("10"),
                            java.util.OptionalLong.empty())) {
                assertEquals(List.of(item), transaction.items());
            }
            assertTrue(fixture.endpoint.removeQueued(item, "cancel fallback"));
            assertEquals(0L, fixture.endpoint.observedRequestCount());
        }
    }

    @Test
    void nonBatchQueueAndExecutionCapacityAreIndependent() {
        FlexlbConfig config = productionConfig();
        config.setDispatcher(DispatcherConfig.nonBatch());
        config.getDispatcher().setMaxInflightPerPrefillWorker(1);
        try (Fixture fixture = fixture(config)) {
            var endpoint = fixture.endpoint;
            var first = item(config, endpoint, 1L);
            var second = item(config, endpoint, 2L);
            assertTrue(EndpointTestSupport.offer(endpoint, first));
            assertTrue(EndpointTestSupport.offer(endpoint, second));
            assertEquals(2, endpoint.captureQueueSnapshot().items().size());

            try (var delivery = endpoint.tryBeginRouteCommitAdmission()) {
                var firstAdmission = delivery.reserve(first, 1L);
                assertEquals(PrefillState.CapacityStatus.ACQUIRED, firstAdmission.status());
                try (var reservation = firstAdmission.reservation()) {
                    assertEquals(
                            PrefillState.CapacityStatus.CAPACITY_FULL,
                            delivery.reserve(second, 1L).status());
                }
                var secondAdmission = delivery.reserve(second, 1L);
                assertEquals(PrefillState.CapacityStatus.ACQUIRED, secondAdmission.status());
                secondAdmission.reservation().close();
            }
        }
    }

    @Test
    @Timeout(10)
    void configuredBatchGroupSizeIsIndependentOfConcurrentBatchLimit() throws Exception {
        FlexlbConfig config = productionConfig();
        var decision = config.getScheduler().getDecision();
        decision.setMaxRequests(2);
        decision.setMaxCollectionWaitMs(60_000L);
        config.getDispatcher().setMaxInflightPerPrefillWorker(1);
        try (Fixture fixture = fixture(config)) {
            CountDownLatch prepared = new CountDownLatch(1);
            AtomicInteger groupSize = new AtomicInteger();
            doAnswer(
                            invocation -> {
                                List<ScheduledRequest> items = invocation.getArgument(0);
                                groupSize.set(items.size());
                                prepared.countDown();
                                return fixture.parked.prepare(
                                        items,
                                        invocation.getArgument(1),
                                        invocation.getArgument(2));
                            })
                    .when(fixture.delivery)
                    .prepare(anyList(), any(), any());
            assertEquals(
                    2,
                    fixture.endpoint
                            .captureRouteProjectionInputs()
                            .queue()
                            .constraints()
                            .maxRequests());
            assertTrue(
                    EndpointTestSupport.offer(
                            fixture.endpoint, item(config, fixture.endpoint, 1L)));
            assertFalse(prepared.await(100L, TimeUnit.MILLISECONDS));
            assertTrue(
                    EndpointTestSupport.offer(
                            fixture.endpoint, item(config, fixture.endpoint, 2L)));
            assertTrue(prepared.await(5L, TimeUnit.SECONDS));
            assertEquals(
                    2, groupSize.get(), "concurrent batch limit must not change each group's size");
        }
    }

    @Test
    void defaultBatchLimitAllowsTwoReservationsAndReopensAfterRelease() {
        FlexlbConfig config = productionConfig();
        try (Fixture fixture = fixture(config)) {
            var items = new ArrayList<ScheduledRequest>();
            var reservations = new ArrayList<PrefillState.BatchReservation>();
            try {
                for (long id = 1L; id <= 3L; id++) {
                    ScheduledRequest queued = item(config, fixture.endpoint, id);
                    items.add(queued);
                    assertEquals(2, queued.maxInflightBatchesPerPrefillWorker());
                    assertTrue(
                            EndpointTestSupport.offer(fixture.endpoint, queued),
                            "BATCH does not impose an additional request-count limit on the waiting queue");
                }
                for (int index = 0; index < 2; index++) {
                    var result =
                            fixture.endpoint.reserveBatch(
                                    items.get(index),
                                    100L + index,
                                    items.get(index).maxInflightBatchesPerPrefillWorker());
                    assertEquals(PrefillState.CapacityStatus.ACQUIRED, result.status());
                    reservations.add(result.reservation());
                }
                assertEquals(
                        PrefillState.CapacityStatus.CAPACITY_FULL,
                        fixture.endpoint.reserveBatch(items.get(2), 102L, 2).status());
                reservations.removeFirst().close();
                var replacement = fixture.endpoint.reserveBatch(items.get(2), 102L, 2);
                assertEquals(PrefillState.CapacityStatus.ACQUIRED, replacement.status());
                reservations.add(replacement.reservation());
            } finally {
                reservations.forEach(PrefillState.BatchReservation::close);
            }
        }
    }

    private static FlexlbConfig productionConfig() {
        FlexlbConfig config = new FlexlbConfig();
        config.getRequestLifecycle().getRequest().setTimeoutMs(60_000L);
        config.getRequestLifecycle().getDecision().setLifetime(2.0);
        return config;
    }

    private static ScheduledRequest item(
            FlexlbConfig config, PrefillEndpoint endpoint, long requestId) {
        return item(config, endpoint, requestId, 50);
    }

    private static ScheduledRequest item(
            FlexlbConfig config, PrefillEndpoint endpoint, long requestId, int priority) {
        Request request = new Request();
        request.setRequestId(Long.toString(requestId));
        request.setSeqLen(100L);
        BalanceContext context = new BalanceContext(config);
        context.setRequest(request);
        context.setSchedulingMetadata(SchedulingMetadata.explicit(priority, Long.MAX_VALUE));
        return new ScheduledRequest(
                context,
                new CompletableFuture<Response>(),
                null,
                null,
                null,
                endpoint,
                null,
                null,
                System.currentTimeMillis());
    }

    private static Fixture fixture(FlexlbConfig config) {
        WorkerStatus status =
                WorkerStatus.createDiscovered(
                        RoleType.PREFILL, "test", "127.0.0.1", 8080, 9090, "test");
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRole(RoleType.PREFILL);
        response.setAlive(true);
        EndpointTestSupport.publishStatus(status, response);
        var runtime = EndpointTestSupport.requestRuntime();
        DeliveryStrategy parked = EndpointTestSupport.routeStrategy(runtime);
        DeliveryStrategy delivery = mock(DeliveryStrategy.class, delegatesTo(parked));
        PrefillEndpoint endpoint =
                new PrefillEndpoint(
                        status,
                        config,
                        delivery,
                        runtime.events(),
                        mock(BatchSchedulerReporter.class));
        endpoint.startGeneration();
        return new Fixture(endpoint, delivery, parked, runtime);
    }

    /**
     * Owns resources used to exercise worker batcher request-capacity limits.
     */
    private record Fixture(
            PrefillEndpoint endpoint,
            DeliveryStrategy delivery,
            DeliveryStrategy parked,
            EndpointTestSupport.TestRequestRuntime runtime)
            implements AutoCloseable {
        public void close() {
            endpoint.close();
            endpoint.awaitRetirement();
        }
    }
}
