package org.flexlb.balance.endpoint;

import org.flexlb.balance.delivery.DeliveryStrategy;
import org.flexlb.balance.scheduler.ScheduledRequest;
import org.flexlb.config.DecisionPolicyConfig;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.QueueOrderingConfig;
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
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;

/** The configured limit, real worker publication lock and canonical request ledger together. */
class WorkerBatcherRequestCapacityTest {

    @ParameterizedTest
    @CsvSource({"false,90,false", "true,50,false", "true,0,false", "true,90,true"})
    void queuedPreemptionHonorsDisabledPolicyAndStrictlyHigherPriority(
            boolean enabled, int priority, boolean accepted) {
        FlexlbConfig config = preemptionConfig();
        if (!enabled) { config.priorityOrdering().setPreemption(null); }
        try (Fixture fixture = fixture(config)) {
            ScheduledRequest first = item(config, fixture.endpoint, 1L);
            ScheduledRequest second = item(config, fixture.endpoint, 2L);
            ScheduledRequest incoming = item(config, fixture.endpoint, 3L, priority);
            assertTrue(EndpointTestSupport.offer(fixture.endpoint, first));
            assertTrue(EndpointTestSupport.offer(fixture.endpoint, second));
            assertEquals(accepted, EndpointTestSupport.offer(fixture.endpoint, incoming));
            assertEquals(2L, fixture.endpoint.observedRequestCount());
            assertEquals(accepted ? List.of(incoming, first) : List.of(first, second),
                    fixture.endpoint.captureQueueSnapshot().items());
            verify(fixture.runtime.events(), times(accepted ? 1 : 0)).onQueuedItemPreempted(second, incoming);
            assertFalse(fixture.endpoint.removeQueued(accepted ? second : incoming, "non-owner cleanup"));
            assertEquals(2L, fixture.endpoint.observedRequestCount());
        }
    }

    @Test
    @Timeout(10)
    void concurrentHigherPriorityOffersTransferOnlyTheTwoQueuedSeats() throws Exception {
        FlexlbConfig config = preemptionConfig();
        try (Fixture fixture = fixture(config); var writers = Executors.newFixedThreadPool(8)) {
            assertTrue(EndpointTestSupport.offer(fixture.endpoint, item(config, fixture.endpoint, 1L)));
            assertTrue(EndpointTestSupport.offer(fixture.endpoint, item(config, fixture.endpoint, 2L)));
            var results = new ArrayList<java.util.concurrent.Future<Boolean>>();
            for (long id = 3L; id < 35L; id++) {
                ScheduledRequest incoming = item(config, fixture.endpoint, id, 90);
                results.add(writers.submit(() -> EndpointTestSupport.offer(fixture.endpoint, incoming)));
            }
            int accepted = 0;
            for (var result : results) { if (result.get()) { accepted++; } }
            assertEquals(2, accepted);
            assertEquals(2L, fixture.endpoint.observedRequestCount());
            assertTrue(fixture.endpoint.captureQueueSnapshot().items().stream().allMatch(item -> item.priority() == 90));
            verify(fixture.runtime.events(), times(2)).onQueuedItemPreempted(any(), any());
        }
    }

    @Test
    void committedPrefillWorkCannotBeReclaimedAsQueuedCapacity() {
        FlexlbConfig config = preemptionConfig();
        config.getDispatcher().setMaxInflightPerPrefillWorker(1);
        try (Fixture fixture = fixture(config)) {
            ScheduledRequest committed = item(config, fixture.endpoint, 1L);
            try (var reservation = EndpointTestSupport.reserveUnqueued(fixture.endpoint, committed, 100L);
                 var commit = fixture.endpoint.tryBeginRouteCommitAdmission();
                 var handoff = commit.commit(List.of(committed), List.of(reservation))) {
                assertFalse(fixture.endpoint.canPreemptQueuedRequest(90));
                assertFalse(EndpointTestSupport.offer(fixture.endpoint, item(config, fixture.endpoint, 2L, 90)));
                assertEquals(1L, fixture.endpoint.observedRequestCount());
                assertTrue(fixture.endpoint.captureQueueSnapshot().items().isEmpty());
                verify(fixture.runtime.events(), times(0)).onQueuedItemPreempted(any(), any());
            }
        }
    }

    @Test
    @Timeout(10)
    void preemptingPreparedRequestInvalidatesItsDeliveryAndPreservesReplacement() throws Exception {
        FlexlbConfig config = preemptionConfig();
        config.getScheduler().setDecision(DecisionPolicyConfig.single());
        config.getDispatcher().setMaxInflightPerPrefillWorker(1);
        CountDownLatch prepared = new CountDownLatch(1);
        CountDownLatch resume = new CountDownLatch(1);
        CountDownLatch replacementSelected = new CountDownLatch(1);
        try (Fixture fixture = fixture(config)) {
            DeliveryStrategy live = EndpointTestSupport.liveRouteStrategy(fixture.runtime);
            doAnswer(invocation -> {
                List<ScheduledRequest> candidates = invocation.getArgument(0);
                if (candidates.getFirst().requestId() == 1L) {
                    DeliveryStrategy.Transaction transaction = live.prepare(candidates,
                            invocation.getArgument(1), invocation.getArgument(2));
                    assertEquals(1, transaction.items().size());
                    prepared.countDown();
                    assertTrue(resume.await(5, TimeUnit.SECONDS));
                    return transaction;
                }
                replacementSelected.countDown();
                return fixture.parked.prepare(candidates, invocation.getArgument(1), invocation.getArgument(2));
            }).when(fixture.delivery).prepare(anyList(), any(), any());
            ScheduledRequest victim = item(config, fixture.endpoint, 1L);
            ScheduledRequest incoming = item(config, fixture.endpoint, 2L, 90);
            try {
                assertTrue(EndpointTestSupport.offer(fixture.endpoint, victim));
                assertTrue(prepared.await(5, TimeUnit.SECONDS));
                assertTrue(EndpointTestSupport.offer(fixture.endpoint, incoming));
            } finally {
                resume.countDown();
            }
            assertTrue(replacementSelected.await(5, TimeUnit.SECONDS));
            assertEquals(List.of(incoming), fixture.endpoint.captureQueueSnapshot().items());
            assertEquals(1L, fixture.endpoint.observedRequestCount());
            verify(fixture.runtime.requests(), times(0)).tryClaimRouteDelivery(any(), any());
            verify(fixture.runtime.events()).onQueuedItemPreempted(victim, incoming);
        }
    }

    private static FlexlbConfig preemptionConfig() {
        FlexlbConfig config = productionConfig();
        config.setDispatcher(DispatcherConfig.nonBatch());
        config.getScheduler().setOrdering(QueueOrderingConfig.priority());
        return config;
    }

    @Test
    @Timeout(10)
    void nonBatchConcurrentOffersShareOneCountLimitAndExactCancellationReopensIt() throws Exception {
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
            assertEquals(4, admitted.size(), "each writer attempts once; publication must not oversubscribe");
            assertEquals(4L, fixture.endpoint.observedRequestCount());
            ScheduledRequest first = admitted.getFirst();
            assertFalse(fixture.endpoint.removeQueued(item(config, fixture.endpoint, first.requestId()), "stale identity"));
            assertFalse(EndpointTestSupport.offer(fixture.endpoint, item(config, fixture.endpoint, 70L)));
            assertTrue(fixture.endpoint.removeQueued(first, "cancel exact queued request"));
            assertTrue(EndpointTestSupport.offer(fixture.endpoint, item(config, fixture.endpoint, 71L)));
            assertEquals(4L, fixture.endpoint.observedRequestCount());
        }
    }

    @Test
    void singleDecisionUsesTheDefaultTwoRequestLimit() {
        FlexlbConfig config = productionConfig();
        config.setDispatcher(DispatcherConfig.nonBatch());
        config.getScheduler().setDecision(DecisionPolicyConfig.single());
        try (Fixture fixture = fixture(config)) {
            for (long requestId = 1L; requestId <= 2L; requestId++) {
                assertTrue(EndpointTestSupport.offer(fixture.endpoint, item(config, fixture.endpoint, requestId)));
            }
            assertEquals(2L, fixture.endpoint.observedRequestCount());
            assertFalse(EndpointTestSupport.offer(fixture.endpoint, item(config, fixture.endpoint, 3L)));
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
        request.setRequestId(requestId);
        request.setSeqLen(100L);
        BalanceContext context = new BalanceContext();
        context.setRequest(request);
        context.setConfig(config);
        context.setSchedulingMetadata(SchedulingMetadata.explicit(priority, Long.MAX_VALUE));
        return new ScheduledRequest(context, new CompletableFuture<Response>(), null, null, null,
                endpoint, null, null, System.currentTimeMillis());
    }

    private static Fixture fixture(FlexlbConfig config) {
        WorkerStatus status = WorkerStatus.createDiscovered(RoleType.PREFILL, "test", "127.0.0.1", 8080, 9090, "test");
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRole(RoleType.PREFILL);
        response.setAlive(true);
        EndpointTestSupport.publishStatus(status, response);
        var runtime = EndpointTestSupport.requestRuntime();
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
