package org.flexlb.balance.endpoint;

import org.flexlb.balance.delivery.DeliveryStrategy;
import org.flexlb.balance.scheduler.ScheduledRequest;
import org.flexlb.config.DecisionPolicyConfig;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.testfixture.WorkerStatusResponse;
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
import java.util.concurrent.Executors;
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

/** The configured limit, real worker publication lock and canonical request ledger together. */
class WorkerBatcherRequestCapacityTest {

    @ParameterizedTest
    @CsvSource({"false", "true"})
    @Timeout(10)
    void dispatchLimitParksTheSuffixAndCompletionWakesIt(boolean singleDecision) throws Exception {
        FlexlbConfig config = productionConfig();
        config.setDispatcher(DispatcherConfig.nonBatch());
        config.getDispatcher().setMaxInflightPerPrefillWorker(2);
        if (singleDecision) {
            config.getScheduler().setDecision(DecisionPolicyConfig.single());
        } else {
            config.getScheduler().getDecision().setMaxRequests(4);
            config.getScheduler().getDecision().setMaxCollectionWaitMs(0L);
        }
        var delivered = new java.util.concurrent.LinkedBlockingQueue<ScheduledRequest>();
        var runtime = new EndpointTestSupport.TestRequestRuntime() {
            @Override
            void onCompleted(org.flexlb.balance.scheduler.RequestSlot.DeliveryClaim claim,
                    org.flexlb.balance.delivery.DeliveryResult result) {
                delivered.add((ScheduledRequest) org.springframework.test.util.ReflectionTestUtils.getField(claim, "item"));
            }
        };
        try (Fixture fixture = fixture(config, runtime)) {
            DeliveryStrategy live = EndpointTestSupport.liveRouteStrategy(runtime);
            doAnswer(invocation -> live.prepare(invocation.getArgument(0), invocation.getArgument(1), invocation.getArgument(2)))
                    .when(fixture.delivery).prepare(anyList(), any(), any());
            for (long id = 1L; id <= 4L; id++) {
                assertTrue(EndpointTestSupport.offer(fixture.endpoint, item(config, fixture.endpoint, id)));
            }
            ScheduledRequest first = delivered.poll(3, TimeUnit.SECONDS);
            ScheduledRequest second = delivered.poll(3, TimeUnit.SECONDS);
            org.junit.jupiter.api.Assertions.assertNotNull(first);
            org.junit.jupiter.api.Assertions.assertNotNull(second);
            org.junit.jupiter.api.Assertions.assertNull(delivered.poll(150, TimeUnit.MILLISECONDS));
            assertEquals(2, fixture.endpoint.captureQueueSnapshot().items().size());
            assertFalse(fixture.endpoint.canAcceptRequest(), "dispatch slots remain full while the suffix waits");
            assertTrue(fixture.endpoint.releaseCommittedItem(first.execution()));
            ScheduledRequest third = delivered.poll(3, TimeUnit.SECONDS);
            org.junit.jupiter.api.Assertions.assertNotNull(third, "completion must wake the worker without another enqueue");
            assertFalse(fixture.endpoint.releaseCommittedItem(first.execution()));
            org.junit.jupiter.api.Assertions.assertNull(delivered.poll(150, TimeUnit.MILLISECONDS));
            assertTrue(fixture.endpoint.releaseCommittedItem(second.execution()));
            ScheduledRequest fourth = delivered.poll(3, TimeUnit.SECONDS);
            org.junit.jupiter.api.Assertions.assertNotNull(fourth);
            assertTrue(fixture.endpoint.releaseCommittedItem(third.execution()));
            assertTrue(fixture.endpoint.releaseCommittedItem(fourth.execution()));
            assertEquals(0L, fixture.endpoint.observedRequestCount());
        }
    }

    @Test
    @Timeout(10)
    void cancelledPreparedRouteKeepsItsSlotUntilTransactionRollbackAndRetirementDrains() throws Exception {
        FlexlbConfig config = productionConfig();
        config.setDispatcher(DispatcherConfig.nonBatch());
        config.getDispatcher().setMaxInflightPerPrefillWorker(1);
        try (Fixture fixture = fixture(config); var waiter = Executors.newSingleThreadExecutor()) {
            ScheduledRequest item = item(config, fixture.endpoint, 1L);
            assertTrue(EndpointTestSupport.offer(fixture.endpoint, item));
            DeliveryStrategy live = EndpointTestSupport.liveRouteStrategy(fixture.runtime);
            java.util.concurrent.Future<?> retirement;
            try (var transaction = live.prepare(List.of(item), fixture.endpoint.getPredictor().evaluator(),
                    java.util.OptionalLong.empty())) {
                assertEquals(List.of(item), transaction.items());
                assertFalse(fixture.endpoint.routeAdmissionAvailability().isAvailable());
                assertTrue(fixture.endpoint.removeQueued(item, "cancel during preparation"));
                assertFalse(fixture.endpoint.routeAdmissionAvailability().isAvailable(),
                        "the transaction still owns the provisional slot after cancellation");
                fixture.endpoint.close();
                CountDownLatch awaitingRetirement = new CountDownLatch(1);
                retirement = waiter.submit(() -> {
                    awaitingRetirement.countDown();
                    fixture.endpoint.awaitRetirement();
                });
                assertTrue(awaitingRetirement.await(1, TimeUnit.SECONDS));
                org.junit.jupiter.api.Assertions.assertThrows(java.util.concurrent.TimeoutException.class,
                        () -> retirement.get(100, TimeUnit.MILLISECONDS),
                        "retirement must wait for the prepared transaction to return its lease");
            }
            retirement.get(3, TimeUnit.SECONDS);
            assertEquals(0L, fixture.endpoint.observedRequestCount());
        }
    }

    @Test
    void routePreparationRollbackLeavesTheRequestQueuedForRetry() {
        FlexlbConfig config = productionConfig();
        config.setDispatcher(DispatcherConfig.nonBatch());
        config.getDispatcher().setMaxInflightPerPrefillWorker(1);
        try (Fixture fixture = fixture(config)) {
            ScheduledRequest item = item(config, fixture.endpoint, 1L);
            assertTrue(EndpointTestSupport.offer(fixture.endpoint, item));
            DeliveryStrategy live = EndpointTestSupport.liveRouteStrategy(fixture.runtime);
            for (int attempt = 0; attempt < 2; attempt++) {
                try (var transaction = live.prepare(List.of(item), fixture.endpoint.getPredictor().evaluator(),
                        java.util.OptionalLong.empty())) {
                    assertEquals(List.of(item), transaction.items());
                    assertFalse(fixture.endpoint.routeAdmissionAvailability().isAvailable());
                }
                assertTrue(fixture.endpoint.routeAdmissionAvailability().isAvailable());
                assertEquals(List.of(item), fixture.endpoint.captureQueueSnapshot().items());
            }
        }
    }

    @Test
    @Timeout(10)
    void nonBatchWaitingQueueDoesNotConsumeDispatchCapacity() throws Exception {
        FlexlbConfig config = productionConfig();
        config.setDispatcher(DispatcherConfig.nonBatch());
        config.getDispatcher().setMaxInflightPerPrefillWorker(4);
        config.getScheduler().getDecision().setMaxRequests(1);
        try (Fixture fixture = fixture(config);
             var writers = Executors.newFixedThreadPool(8)) {
            var results = new ArrayList<java.util.concurrent.Future<ScheduledRequest>>();
            for (long id = 1L; id <= 64L; id++) {
                ScheduledRequest item = item(config, fixture.endpoint, id);
                results.add(writers.submit(() -> EndpointTestSupport.offer(fixture.endpoint, item) ? item : null));
            }
            var admitted = new ArrayList<ScheduledRequest>();
            for (var result : results) {
                ScheduledRequest owned = result.get();
                if (owned != null) { admitted.add(owned); }
            }
            assertEquals(64, admitted.size(), "waiting requests do not consume dispatch slots");
            assertEquals(64L, fixture.endpoint.observedRequestCount());
            ScheduledRequest first = admitted.getFirst();
            assertFalse(fixture.endpoint.removeQueued(item(config, fixture.endpoint, first.requestId()), "stale identity"));
            assertTrue(fixture.endpoint.canAcceptRequest());
            assertTrue(fixture.endpoint.removeQueued(first, "cancel exact queued request"));
            assertTrue(EndpointTestSupport.offer(fixture.endpoint, item(config, fixture.endpoint, 71L)));
            assertEquals(64L, fixture.endpoint.observedRequestCount());
        }
    }

    @Test
    void singleDecisionCanQueueBeyondTheDefaultTwoDispatchSlots() {
        FlexlbConfig config = productionConfig();
        config.setDispatcher(DispatcherConfig.nonBatch());
        config.getScheduler().setDecision(DecisionPolicyConfig.single());
        try (Fixture fixture = fixture(config)) {
            for (long requestId = 1L; requestId <= 2L; requestId++) {
                assertTrue(EndpointTestSupport.offer(fixture.endpoint, item(config, fixture.endpoint, requestId)));
            }
            assertEquals(2L, fixture.endpoint.observedRequestCount());
            assertTrue(EndpointTestSupport.offer(fixture.endpoint, item(config, fixture.endpoint, 3L)));
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
            doAnswer(invocation -> {
                List<ScheduledRequest> items = invocation.getArgument(0);
                groupSize.set(items.size());
                prepared.countDown();
                return fixture.parked.prepare(items, invocation.getArgument(1), invocation.getArgument(2));
            }).when(fixture.delivery).prepare(anyList(), any(), any());
            assertEquals(2, fixture.endpoint.captureRouteProjectionInputs().queue().constraints().maxRequests());
            assertTrue(EndpointTestSupport.offer(fixture.endpoint, item(config, fixture.endpoint, 1L)));
            assertFalse(prepared.await(100L, TimeUnit.MILLISECONDS));
            assertTrue(EndpointTestSupport.offer(fixture.endpoint, item(config, fixture.endpoint, 2L)));
            assertTrue(prepared.await(5L, TimeUnit.SECONDS));
            assertEquals(2, groupSize.get(), "concurrent batch limit must not change each group's size");
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
                    assertTrue(EndpointTestSupport.offer(fixture.endpoint, queued),
                            "BATCH does not impose an additional request-count limit on the waiting queue");
                }
                for (int index = 0; index < 2; index++) {
                    var result = fixture.endpoint.reserveBatch(items.get(index), 100L + index,
                            items.get(index).maxInflightBatchesPerPrefillWorker());
                    assertEquals(PrefillState.CapacityStatus.ACQUIRED, result.status());
                    reservations.add(result.reservation());
                }
                assertEquals(PrefillState.CapacityStatus.CAPACITY_FULL,
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

    private static ScheduledRequest item(FlexlbConfig config, PrefillEndpoint endpoint, long requestId) {
        return item(config, endpoint, requestId, 50);
    }

    private static ScheduledRequest item(FlexlbConfig config, PrefillEndpoint endpoint, long requestId, int priority) {
        Request request = new Request();
        request.setRequestId(Long.toString(requestId));
        request.setSeqLen(100L);
        BalanceContext context = new BalanceContext(config);
        context.setRequest(request);
        context.setSchedulingMetadata(SchedulingMetadata.explicit(priority, Long.MAX_VALUE));
        return new ScheduledRequest(context, new CompletableFuture<Response>(), null, null, null,
                endpoint, null, null, System.currentTimeMillis());
    }

    private static Fixture fixture(FlexlbConfig config) {
        return fixture(config, EndpointTestSupport.requestRuntime());
    }

    private static Fixture fixture(FlexlbConfig config, EndpointTestSupport.TestRequestRuntime runtime) {
        WorkerStatus status = WorkerStatus.createDiscovered(RoleType.PREFILL, "test", "127.0.0.1", 8080, 9090, "test");
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRole(RoleType.PREFILL);
        response.setAlive(true);
        EndpointTestSupport.publishStatus(status, response);
        DeliveryStrategy parked = EndpointTestSupport.routeStrategy(runtime);
        DeliveryStrategy delivery = mock(DeliveryStrategy.class, delegatesTo(parked));
        PrefillEndpoint endpoint = new PrefillEndpoint(status, config, delivery,
                runtime.events(), mock(BatchSchedulerReporter.class));
        endpoint.startGeneration();
        return new Fixture(endpoint, delivery, parked, runtime);
    }

    private record Fixture(PrefillEndpoint endpoint, DeliveryStrategy delivery, DeliveryStrategy parked,
                           EndpointTestSupport.TestRequestRuntime runtime)
            implements AutoCloseable {
        public void close() { endpoint.close(); endpoint.awaitRetirement(); }
    }
}
