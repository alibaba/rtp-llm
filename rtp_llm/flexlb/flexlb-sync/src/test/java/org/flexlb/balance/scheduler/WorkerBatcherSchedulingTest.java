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

import static org.junit.jupiter.api.Assertions.assertEquals;
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

    private WorkerBatcher runningRuntime(
            FlexlbConfig config,
            PrefillEndpoint endpoint,
            DeliveryStrategy delivery) {
        WorkerBatcher runtime = new WorkerBatcher(
                "scheduling-test", endpoint, config, delivery,
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
        request.setRequestId(requestId);
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
