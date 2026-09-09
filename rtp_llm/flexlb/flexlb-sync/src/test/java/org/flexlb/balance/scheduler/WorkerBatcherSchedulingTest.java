package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.delivery.DeliveryStrategy;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.projection.RouteProjection;
import org.flexlb.config.DecisionPolicyConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.OptionalLong;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/** Concurrency contracts at the final, threaded Prefill scheduling boundary. */
class WorkerBatcherSchedulingTest {

    private final List<WorkerBatcher> runtimes = new ArrayList<>();

    @AfterEach
    void stopRuntimes() {
        for (WorkerBatcher runtime : runtimes) {
            assertNull(runtime.stopAndAwait());
        }
    }

    @Test
    @Timeout(value = 10, unit = TimeUnit.SECONDS)
    void blockedExactHeadWaitsForItsCapacityEventWithoutPolling()
            throws Exception {
        FlexlbConfig config = singleConfig();
        PrefillEndpoint endpoint = stableEndpoint(stableStatus());
        EventDrivenBlock delivery = new EventDrivenBlock();
        WorkerBatcher runtime = runningRuntime(config, endpoint, delivery);
        ScheduledRequest head = item(
                config, endpoint, 1L, 50, System.currentTimeMillis());

        assertTrue(runtime.offer(head));
        await(delivery.firstAttempt);
        await(delivery.firstCapacity.subscribed);

        TimeUnit.MILLISECONDS.sleep(100L);
        assertEquals(1, delivery.attempts.get(),
                "a capacity miss must not be polled");
        assertSame(head, runtime.captureQueueSnapshot().items().getFirst());

        delivery.firstCapacity.release();
        await(delivery.secondAttempt);
        await(delivery.parkedCapacity.subscribed);

        TimeUnit.MILLISECONDS.sleep(100L);
        assertEquals(2, delivery.attempts.get(),
                "one capacity signal must trigger exactly one retry");
        assertSame(head, runtime.captureQueueSnapshot().items().getFirst());
    }

    @Test
    @Timeout(value = 10, unit = TimeUnit.SECONDS)
    void projectionCacheTracksCapacityBlockAndWake() {
        FlexlbConfig config = singleConfig();
        PrefillEndpoint endpoint = stableEndpoint(stableStatus());
        ProjectionCacheBlock delivery = new ProjectionCacheBlock();
        WorkerBatcher runtime = runningRuntime(config, endpoint, delivery);
        ScheduledRequest head = item(
                config, endpoint, 11L, 50, System.currentTimeMillis());

        try {
            assertTrue(runtime.offer(head));
            await(delivery.firstPrepareEntered);
            assertNull(runtime.captureRouteProjectionInputs()
                    .queue().admissionBlock(),
                    "cache is warmed before the delivery miss is published");

            delivery.allowFirstPrepare.countDown();
            await(delivery.firstCapacity.subscribed);
            assertNotNull(runtime.captureRouteProjectionInputs()
                    .queue().admissionBlock(),
                    "publishing the block must invalidate the warm cache");

            delivery.firstCapacity.release();
            await(delivery.secondPrepareEntered);
            assertNull(runtime.captureRouteProjectionInputs()
                    .queue().admissionBlock(),
                    "capacity wake must invalidate the cached block");
        } finally {
            delivery.allowFirstPrepare.countDown();
            delivery.allowSecondPrepare.countDown();
        }
    }

    @Test
    @Timeout(value = 10, unit = TimeUnit.SECONDS)
    void removingCapturedMemberDuringPredictionInvalidatesWholeSelection()
            throws Exception {
        FlexlbConfig config = fixedConfig();
        CountDownLatch firstStatusRead = new CountDownLatch(1);
        CountDownLatch allowFirstStatusRead = new CountDownLatch(1);
        WorkerStatus status = gatedFirstStatusRead(
                firstStatusRead, allowFirstStatusRead);
        PrefillEndpoint endpoint = stableEndpoint(status);
        PredictionGate delivery = new PredictionGate();
        WorkerBatcher runtime = runningRuntime(config, endpoint, delivery);
        long now = System.currentTimeMillis();
        ScheduledRequest first = item(config, endpoint, 1L, 50, now);
        ScheduledRequest revoked = item(config, endpoint, 2L, 50, now + 1L);

        try {
            assertTrue(runtime.offer(first));
            await(firstStatusRead);
            assertTrue(runtime.offer(revoked));
            allowFirstStatusRead.countDown();
            await(delivery.groupPredictionEntered);

            assertTrue(runtime.removeQueued(
                    revoked, "test revoke during prediction"));
            delivery.allowGroupPrediction.countDown();
            await(delivery.nextDecisionStarted);

            assertEquals(0, delivery.prepareCalls.get(),
                    "a selection containing a revoked identity cannot prepare");
            List<ScheduledRequest> remaining =
                    runtime.captureQueueSnapshot().items();
            assertEquals(1, remaining.size());
            assertSame(first, remaining.getFirst());
        } finally {
            allowFirstStatusRead.countDown();
            delivery.allowGroupPrediction.countDown();
        }
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.CsvSource({
            "NON_BATCH,SINGLE,1,0,single_request",
            "BATCH,SINGLE,1,0,single_request",
            "NON_BATCH,FIXED_WINDOW,2,60000,batch_full",
            "BATCH,FIXED_WINDOW,2,60000,batch_full",
            "NON_BATCH,FIXED_WINDOW,1,0,fixed_window_timeout",
            "BATCH,FIXED_WINDOW,1,0,fixed_window_timeout"
    })
    void committedGroupIsVisibleBeforeDeliveryPublication(String dispatcher,
                                                          String policy,
                                                          int count,
                                                          long windowMs,
                                                          String expectedReason) throws Exception {
        FlexlbConfig config = singleConfig();
        if (dispatcher.equals("NON_BATCH")) {
            config.setDispatcher(org.flexlb.config.DispatcherConfig.nonBatch());
        }
        if (policy.equals("FIXED_WINDOW")) {
            var decision = SchedulingTestConfig.useFixedWindowDecision(config);
            decision.setMaxRequests(2);
            decision.setMaxCollectionWaitMs(windowMs);
        }
        PrefillEndpoint endpoint = stableEndpoint(stableStatus());
        when(endpoint.getIp()).thenReturn("10.0.0.1");
        when(endpoint.reservePublishedRouteCredit(org.mockito.ArgumentMatchers.any(),
                org.mockito.ArgumentMatchers.anyLong(), org.mockito.ArgumentMatchers.anyInt()))
                .thenAnswer(ignored -> new org.flexlb.balance.endpoint.PrefillState.ReservationResult<>(
                        org.flexlb.balance.endpoint.PrefillState.CapacityStatus.ACQUIRED,
                        mock(org.flexlb.balance.endpoint.PrefillState.RouteReservation.class)));
        DeliveryStrategy delivery = mock(DeliveryStrategy.class);
        List<org.flexlb.dao.pv.DecisionGroup> published = new CopyOnWriteArrayList<>();
        when(delivery.prepare(org.mockito.ArgumentMatchers.anyList(), org.mockito.ArgumentMatchers.any(),
                org.mockito.ArgumentMatchers.any())).thenAnswer(invocation -> {
            List<ScheduledRequest> members = List.copyOf(invocation.getArgument(0));
            var transaction = mock(DeliveryStrategy.Transaction.class);
            when(transaction.items()).thenReturn(members);
            org.mockito.Mockito.doAnswer(ignored -> {
                for (ScheduledRequest member : members) {
                    published.add(member.ctx().getDecisionGroup());
                    member.future().complete(new Response());
                }
                return null;
            }).when(transaction).handoff(org.mockito.ArgumentMatchers.anyString(), org.mockito.ArgumentMatchers.anyInt());
            return transaction;
        });
        WorkerBatcher runtime = runningRuntime(config, endpoint, delivery);
        List<ScheduledRequest> requests = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            ScheduledRequest request = item(config, endpoint, 100L + i, 50, System.currentTimeMillis());
            requests.add(request);
            assertTrue(runtime.offer(request));
        }
        for (ScheduledRequest request : requests) {
            request.future().get(5, TimeUnit.SECONDS);
        }
        assertEquals(count, published.size());
        String id = published.getFirst().id();
        assertNotNull(id);
        for (var group : published) {
            assertEquals(id, group.id());
            assertEquals(count, group.committedSize());
            assertEquals(expectedReason, group.reason());
            assertEquals(policy, group.policy());
            assertEquals(dispatcher, group.dispatcher());
            assertEquals("10.0.0.1", group.worker());
            assertTrue(group.requestWaitMs() >= 0L);
        }
    }

    @Test
    void windowUpdateInvalidatesProjectionWithoutQueueMutation() {
        FlexlbConfig initial = fixedConfig();
        initial.decisionPolicy().setMaxCollectionWaitMs(100L);
        AtomicReference<FlexlbConfig> current = new AtomicReference<>(initial);
        WorkerBatcher runtime = new WorkerBatcher("hot-window-projection", stableEndpoint(stableStatus()),
                current::get, mock(DeliveryStrategy.class), mock(EndpointEventProjector.class));
        runtimes.add(runtime);
        runtime.start();

        var before = runtime.captureRouteProjectionInputs();
        assertEquals(100L, before.queue().constraints().collectionWindowMs());
        assertSame(before, runtime.captureRouteProjectionInputs());
        FlexlbConfig updated = fixedConfig();
        updated.decisionPolicy().setMaxCollectionWaitMs(500L);
        current.set(updated);
        var after = runtime.captureRouteProjectionInputs();
        assertEquals(500L, after.queue().constraints().collectionWindowMs());
        assertSame(after, runtime.captureRouteProjectionInputs());
        assertEquals(before.queue().activeItems(), after.queue().activeItems());
        assertEquals(100L, before.queue().constraints().collectionWindowMs());

        current.set(initial);
        assertEquals(100L, runtime.captureRouteProjectionInputs().queue().constraints().collectionWindowMs());
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(booleans = {false, true})
    @Timeout(value = 10, unit = TimeUnit.SECONDS)
    void existingRuntimeUsesUpdatedWindowOnNextDecision(boolean increaseWindow) throws Exception {
        FlexlbConfig initial = fixedConfig();
        initial.decisionPolicy().setMaxPredictedExecutionMs(null);
        initial.decisionPolicy().setMaxCollectionWaitMs(increaseWindow ? 0L : 60_000L);
        AtomicReference<FlexlbConfig> current = new AtomicReference<>(initial);
        PrefillEndpoint endpoint = stableEndpoint(stableStatus());
        EventDrivenBlock delivery = new EventDrivenBlock();
        WorkerBatcher runtime = new WorkerBatcher("hot-window-delivery", endpoint, current::get, delivery,
                mock(EndpointEventProjector.class));
        runtimes.add(runtime);
        runtime.start();

        FlexlbConfig updated = fixedConfig();
        updated.decisionPolicy().setMaxPredictedExecutionMs(null);
        updated.decisionPolicy().setMaxCollectionWaitMs(increaseWindow ? 60_000L : 0L);
        current.set(updated);
        assertTrue(runtime.offer(item(initial, endpoint, 77L, 50, System.currentTimeMillis())));
        if (increaseWindow) {
            org.junit.jupiter.api.Assertions.assertFalse(delivery.firstAttempt.await(100L, TimeUnit.MILLISECONDS));
            current.set(initial);
            runtime.signalSchedulingInputsChanged();
        }
        await(delivery.firstAttempt);
    }

    private WorkerBatcher runningRuntime(FlexlbConfig config, PrefillEndpoint endpoint, DeliveryStrategy delivery) {
        WorkerBatcher runtime = new WorkerBatcher(
                "scheduling-test", endpoint, () -> config, delivery,
                mock(EndpointEventProjector.class));
        runtimes.add(runtime);
        runtime.start();
        return runtime;
    }

    private static FlexlbConfig singleConfig() {
        FlexlbConfig config = new FlexlbConfig();
        SchedulingTestConfig.useFifoQueue(config);
        SchedulingTestConfig.useSingleDecision(config);
        SchedulingTestConfig.useBatchDispatcher(config);
        return config;
    }

    private static FlexlbConfig fixedConfig() {
        FlexlbConfig config = new FlexlbConfig();
        SchedulingTestConfig.useFifoQueue(config);
        DecisionPolicyConfig decision =
                SchedulingTestConfig.useFixedWindowDecision(config);
        decision.setMaxRequests(2);
        decision.setMaxCollectionWaitMs(60_000L);
        decision.setMaxPredictedExecutionMs(500L);
        SchedulingTestConfig.useBatchDispatcher(config);
        return config;
    }

    private static ScheduledRequest item(
            FlexlbConfig config,
            PrefillEndpoint endpoint,
            long requestId,
            int priority,
            long enqueuedAtMs) {
        Request request = new Request();
        request.setRequestId(Long.toString(requestId));
        request.setPriority(priority);
        request.setSeqLen(10L);
        BalanceContext context = new BalanceContext();
        context.setRequest(request);
        context.setConfig(config);
        context.setSchedulingMetadata(
                SchedulingMetadata.explicit(priority, Long.MAX_VALUE));
        return new ScheduledRequest(
                context,
                new CompletableFuture<Response>(),
                null,
                null,
                null,
                endpoint,
                null,
                null,
                enqueuedAtMs);
    }

    private static PrefillEndpoint stableEndpoint(WorkerStatus status) {
        PrefillTimePredictor predictor = mock(PrefillTimePredictor.class);
        when(predictor.evaluator())
                .thenReturn(mock(PrefillTimePredictor.Evaluator.class));
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.getStatus()).thenReturn(status);
        when(endpoint.getPredictor()).thenReturn(predictor);
        return endpoint;
    }

    private static WorkerStatus stableStatus() {
        WorkerStatus status = mock(WorkerStatus.class);
        when(status.committedEngineObservation()).thenReturn(capacity());
        return status;
    }

    private static WorkerStatus gatedFirstStatusRead(
            CountDownLatch entered,
            CountDownLatch proceed) {
        WorkerStatus status = mock(WorkerStatus.class);
        AtomicBoolean first = new AtomicBoolean(true);
        when(status.committedEngineObservation()).thenAnswer(ignored -> {
            if (first.compareAndSet(true, false)) {
                entered.countDown();
                await(proceed);
            }
            return capacity();
        });
        return status;
    }

    private static WorkerStatus.EngineObservation capacity() {
        return new WorkerStatus.EngineObservation(
                RoleType.PREFILL,
                null,
                0L,
                0L,
                Map.of(),
                0.0,
                0L,
                0L,
                0L,
                0L,
                0L,
                1_000_000L,
                0L,
                0L);
    }

    private static void await(CountDownLatch latch) {
        try {
            assertTrue(latch.await(5, TimeUnit.SECONDS),
                    "worker did not reach the expected boundary");
        } catch (InterruptedException interruption) {
            Thread.currentThread().interrupt();
            throw new AssertionError(
                    "interrupted while awaiting worker boundary",
                    interruption);
        }
    }

    private static CapacityBoundary unavailable(
            CapacityBoundary.Availability availability) {
        return CapacityBoundary.unavailable(
                availability,
                new RouteProjection.AdmissionBlockSemantics(
                        "TEST_CAPACITY",
                        RouteProjection.AfterProbeAdmission.BLOCKED,
                        "TEST_CAPACITY",
                        RoleType.PREFILL));
    }

    private abstract static class BoundaryDelivery implements DeliveryStrategy {

        @Override
        public double projectGroupDurationMs(
                List<ScheduledRequest> items,
                PrefillTimePredictor.Evaluator evaluator) {
            return 0.0;
        }

        @Override
        public RouteProjection.DeliveryProjection projectionPolicy() {
            return mock(RouteProjection.DeliveryProjection.class);
        }
    }

    private static final class EventDrivenBlock extends BoundaryDelivery {

        private final TestAvailability firstCapacity = new TestAvailability();
        private final TestAvailability parkedCapacity = new TestAvailability();
        private final AtomicInteger attempts = new AtomicInteger();
        private final CountDownLatch firstAttempt = new CountDownLatch(1);
        private final CountDownLatch secondAttempt = new CountDownLatch(1);

        @Override
        public Transaction prepare(
                List<ScheduledRequest> candidates,
                PrefillTimePredictor.Evaluator evaluator,
                OptionalLong plannedPrediction) {
            int attempt = attempts.incrementAndGet();
            if (attempt == 1) {
                firstAttempt.countDown();
                return WorkerBatcherTestSupport.boundaryOnly(
                        candidates.getFirst(), unavailable(firstCapacity));
            }
            secondAttempt.countDown();
            return WorkerBatcherTestSupport.boundaryOnly(
                    candidates.getFirst(), unavailable(parkedCapacity));
        }
    }

    private static final class ProjectionCacheBlock extends BoundaryDelivery {

        private final TestAvailability firstCapacity = new TestAvailability();
        private final TestAvailability parkedCapacity = new TestAvailability();
        private final CountDownLatch firstPrepareEntered = new CountDownLatch(1);
        private final CountDownLatch allowFirstPrepare = new CountDownLatch(1);
        private final CountDownLatch secondPrepareEntered = new CountDownLatch(1);
        private final CountDownLatch allowSecondPrepare = new CountDownLatch(1);
        private final AtomicInteger attempts = new AtomicInteger();

        @Override
        public Transaction prepare(
                List<ScheduledRequest> candidates,
                PrefillTimePredictor.Evaluator evaluator,
                OptionalLong plannedPrediction) {
            if (attempts.incrementAndGet() == 1) {
                firstPrepareEntered.countDown();
                await(allowFirstPrepare);
                return WorkerBatcherTestSupport.boundaryOnly(
                        candidates.getFirst(), unavailable(firstCapacity));
            }
            secondPrepareEntered.countDown();
            await(allowSecondPrepare);
            return WorkerBatcherTestSupport.boundaryOnly(
                    candidates.getFirst(), unavailable(parkedCapacity));
        }
    }

    private static final class PredictionGate extends BoundaryDelivery {

        private final CountDownLatch groupPredictionEntered =
                new CountDownLatch(1);
        private final CountDownLatch allowGroupPrediction =
                new CountDownLatch(1);
        private final CountDownLatch nextDecisionStarted =
                new CountDownLatch(1);
        private final AtomicBoolean groupPredictionReturned =
                new AtomicBoolean();
        private final AtomicInteger prepareCalls = new AtomicInteger();

        @Override
        public Transaction prepare(
                List<ScheduledRequest> candidates,
                PrefillTimePredictor.Evaluator evaluator,
                OptionalLong plannedPrediction) {
            prepareCalls.incrementAndGet();
            return WorkerBatcherTestSupport.boundaryOnly(
                    candidates.getFirst(),
                    unavailable(new TestAvailability()));
        }

        @Override
        public double projectGroupDurationMs(
                List<ScheduledRequest> items,
                PrefillTimePredictor.Evaluator evaluator) {
            if (groupPredictionReturned.get()) {
                nextDecisionStarted.countDown();
            }
            if (items.size() == 2) {
                groupPredictionEntered.countDown();
                await(allowGroupPrediction);
                groupPredictionReturned.set(true);
            }
            return 100.0;
        }
    }

    private static final class TestAvailability
            implements CapacityBoundary.Availability {

        private final AtomicBoolean available = new AtomicBoolean();
        private final CopyOnWriteArrayList<Runnable> listeners =
                new CopyOnWriteArrayList<>();
        private final CountDownLatch subscribed = new CountDownLatch(1);

        @Override
        public boolean isAvailable() {
            return available.get();
        }

        @Override
        public void addListener(Runnable listener) {
            listeners.add(listener);
            subscribed.countDown();
        }

        @Override
        public void removeListener(Runnable listener) {
            listeners.remove(listener);
        }

        void release() {
            available.set(true);
            listeners.forEach(Runnable::run);
        }
    }
}
