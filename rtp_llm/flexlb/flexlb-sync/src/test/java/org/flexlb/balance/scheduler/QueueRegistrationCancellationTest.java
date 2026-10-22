package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.eviction.DecodeCapacityAcquirer;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.service.RecentCacheKeyTraceReporter;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.Test;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.nullable;
import static org.mockito.Mockito.*;

/** Real registration and queue lifecycle; endpoint selection is isolated by a mock router. */
class QueueRegistrationCancellationTest {
    @Test
    void registrationHookRunsBeforeSelectionAndNormalRequestsStillStart() throws Exception {
        try (Fixture f = new Fixture()) {
            AtomicBoolean listening = new AtomicBoolean();
            when(f.router.select(any(), nullable(String.class))).thenAnswer(call -> {
                assertTrue(listening.get(), "selection cannot precede listener installation");
                return PlacementResult.rejected(Response.error(StrategyErrorType.RESOURCE_EXHAUSTED));
            });
            RequestContext context = f.context(901L);
            var result = f.queue.submit(context, () -> {
                assertSame(f.queue, context.data().getScheduler());
                assertNotNull(context.data().getFuture());
                assertEquals(0, RequestProtocolTestSupport.queuedCount(f.queue));
                verify(f.router, never()).select(any(), nullable(String.class));
                listening.set(true);
            });
            assertFalse(result.get(2L, TimeUnit.SECONDS).isSuccess());
            verify(f.router).select(any(), nullable(String.class));
        }
    }

    @Test
    void cancellationDuringRegistrationHookPreventsEnqueueAndSelection() throws Exception {
        try (Fixture f = new Fixture()) {
            RequestContext context = f.context(902L);
            var result = f.queue.submit(context, () -> {
                assertSame(f.queue, context.data().getScheduler());
                assertNotNull(f.queue.cancel(902L, 0L, CancelReason.CLIENT_CANCELLED));
            });
            assertFalse(result.get(2L, TimeUnit.SECONDS).isSuccess());
            assertEquals(0, RequestProtocolTestSupport.queuedCount(f.queue));
            verify(f.router, never()).select(any(), nullable(String.class));
        }
    }

    @Test
    void concurrentCancellationBeforeHookReturnsPreventsSelection() throws Exception {
        try (Fixture f = new Fixture(); var executor = Executors.newSingleThreadExecutor()) {
            var registered = new CountDownLatch(1);
            var resume = new CountDownLatch(1);
            var submitted = executor.submit(() -> f.queue.submit(f.context(903L), () -> {
                registered.countDown();
                try {
                    assertTrue(resume.await(2L, TimeUnit.SECONDS));
                } catch (InterruptedException error) {
                    Thread.currentThread().interrupt();
                    throw new IllegalStateException(error);
                }
            }));
            try {
                assertTrue(registered.await(2L, TimeUnit.SECONDS));
                assertNotNull(f.queue.cancel(903L, 0L, CancelReason.CLIENT_CANCELLED));
                SchedulerTestSupport.runtime(f.queue).shutdown();

                assertFalse(submitted.isDone(), "a cancelled hook holds no scheduling resources");
            } finally {
                resume.countDown();
            }
            assertFalse(submitted.get(2L, TimeUnit.SECONDS).get(2L, TimeUnit.SECONDS).isSuccess());
            verify(f.router, never()).select(any(), nullable(String.class));
            SchedulerTestSupport.runtime(f.queue).shutdown();

        }
    }

    @Test
    void failedHookTerminatesRequestAndRejectedSubmissionsDoNotInvokeHook() throws Exception {
        try (Fixture f = new Fixture()) {
            var result = f.queue.submit(f.context(904L), () -> {
                throw new IllegalStateException("listener installation failed");
            });
            assertFalse(result.get(2L, TimeUnit.SECONDS).isSuccess());
            verify(f.router, never()).select(any(), nullable(String.class));
            SchedulerTestSupport.runtime(f.queue).requests().stopAccepting();
            AtomicBoolean called = new AtomicBoolean();
            assertFalse(f.queue.submit(f.context(905L), () -> called.set(true)).get().isSuccess());
            assertFalse(called.get());
        }
    }

    @Test
    void cacheVictimTerminatesAndHighHitStillExpiresWithoutNewArrivals() throws Exception {
        var policy = new org.flexlb.config.PrefillQueueSheddingConfig();
        policy.setExpectedWaitThresholdMs(100L);
        policy.setMaxWaitMs(800L);
        try (Fixture f = new Fixture(policy)) {
            when(f.router.select(any(), nullable(String.class))).thenReturn(PlacementResult.blocked(
                    new PlacementKey(org.flexlb.dao.route.RoleType.PREFILL, null, null)));
            when(f.router.estimateQueuedWork(any(), nullable(String.class))).thenAnswer(call -> {
                RequestContext context = call.getArgument(0);
                return new PrefillQueueShedding.Estimate(100, context.data().getRequestId() == 910L ? 0 : 99, 100, 1);
            });
            var cold = f.context(910L);
            var hot = f.context(911L);
            cold.data().setSchedulingMetadata(org.flexlb.dao.SchedulingMetadata.explicit(50,
                    f.config.resolveExpiresAtMs(System.currentTimeMillis())));
            hot.data().setSchedulingMetadata(org.flexlb.dao.SchedulingMetadata.explicit(50,
                    f.config.resolveExpiresAtMs(System.currentTimeMillis())));
            var coldResult = f.queue.submit(cold);
            var hotResult = f.queue.submit(hot);
            assertEquals(StrategyErrorType.RESOURCE_EXHAUSTED.getErrorCode(), coldResult.get(2, TimeUnit.SECONDS).getCode());
            assertFalse(hotResult.get(2, TimeUnit.SECONDS).isSuccess());
            assertTrue(System.currentTimeMillis() >= hot.data().getRequestExpiresAtMs());
            assertEquals(CancelReason.DEADLINE_EXCEEDED, hot.snapshot().cancellation().reason());
            RequestProtocolTestSupport.awaitCondition(() -> RequestProtocolTestSupport.queuedCount(f.queue) == 0);
            assertEquals(0, RequestProtocolTestSupport.queuedCount(f.queue));
        }
    }

    @Test
    void decodeOnlyPressureDoesNotTriggerCacheEviction() throws Exception {
        var policy = new org.flexlb.config.PrefillQueueSheddingConfig();
        policy.setExpectedWaitThresholdMs(1L);
        try (Fixture f = new Fixture(policy)) {
            when(f.router.select(any(), nullable(String.class))).thenReturn(PlacementResult.blocked(
                    new PlacementKey(org.flexlb.dao.route.RoleType.DECODE, null, null)));
            when(f.router.estimateQueuedWork(any(), nullable(String.class)))
                    .thenReturn(new PrefillQueueShedding.Estimate(100, 0, 10_000, 1));
            var result = f.queue.submit(f.context(912L));
            assertThrows(java.util.concurrent.TimeoutException.class, () -> result.get(300, TimeUnit.MILLISECONDS));
            f.queue.cancel(912L, 0L, CancelReason.CLIENT_CANCELLED);
            assertEquals(StrategyErrorType.REQUEST_CANCELLED.getErrorCode(), result.get(2, TimeUnit.SECONDS).getCode());
        }
    }

    private static final class Fixture implements AutoCloseable {
        final FlexlbConfig config = SchedulingTestConfig.batchConfig();
        final RequestWorkerSelector router = mock(RequestWorkerSelector.class);
        final SchedulerRuntime runtime;
        final QueuedRequestScheduler queue;

        Fixture() { this(null); }

        Fixture(org.flexlb.config.PrefillQueueSheddingConfig policy) {
            config.queueScheduler().setPrefillQueueShedding(policy);
            SchedulingTestConfig.useFifoQueue(config);
            if (policy != null) {
                SchedulingTestConfig.useNonBatchDispatcher(config);
                config.queueScheduler().getDecision().setType(org.flexlb.config.DecisionPolicyConfig.Type.SINGLE);
            }
            var service = mock(ConfigService.class);
            when(service.loadBalanceConfig()).thenReturn(config);
            var batches = mock(DeliveryMetricsReporter.class);
            runtime = new SchedulerRuntime(new RequestRepository(), mock(EndpointRegistry.class), batches,
                    mock(RequestSchedulerReporter.class), SchedulerTestSupport.dispatcher(), service, mock(EngineGrpcClient.class));
            queue = new QueuedRequestScheduler(config, router, batches, mock(DecodeCapacityAcquirer.class),
                    runtime, new PlacementAvailability(), SchedulingTestConfig.capacity(config), org.mockito.Mockito.mock(org.flexlb.service.RecentCacheKeyTraceReporter.class));
            runtime.registerScheduler(queue);
            queue.start();
        }

        RequestContext context(long id) { return RequestProtocolTestSupport.context(config, id); }

        @Override public void close() { runtime.shutdown(); }
    }
}
