package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.eviction.DecodeAdmission;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.nullable;
import static org.mockito.Mockito.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.nullable;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/** Real registration and queue lifecycle; endpoint selection is isolated by a mock router. */
class QueueRegistrationCancellationTest {
    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void cancellationSettlesBeforeQueueStartsAndSurvivesShutdown(boolean futureCancellation)
            throws Exception {
        try (Fixture f = new Fixture(false)) {
            var context = f.context(908L);
            var future = f.queue.submit(context);
            try {
                if (futureCancellation) {
                    assertTrue(future.cancel(false));
                    assertTrue(future.isCancelled());
                } else {
                    f.queue.cancel(context, 0L, CancelReason.CLIENT_CANCELLED);
                    assertEquals(
                            StrategyErrorType.REQUEST_CANCELLED.getErrorCode(),
                            future.get(2, TimeUnit.SECONDS).getCode());
                }
                RequestProtocolTestSupport.awaitCondition(
                        () -> f.runtime.requests().liveRequestCount() == 0);
            } finally {
                f.queue.start();
            }
            f.queue.close();
            assertEquals(
                    RequestState.Phase.CANCELLED,
                    f.runtime.requests().getRequestState(908L, 0L).state());
            verify(f.router, never()).select(any(), nullable(String.class));
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void completionBeforeSubmitEventCannotRestoreTheRequest(boolean cancel) throws Exception {
        try (Fixture f = new Fixture();
                var executor = Executors.newSingleThreadExecutor()) {
            var enteringQueue = new CountDownLatch(1);
            var continueSubmit = new CountDownLatch(1);
            when(f.router.resolvePolicyGroup(any()))
                    .thenAnswer(
                            call -> {
                                enteringQueue.countDown();
                                RequestProtocolTestSupport.await(continueSubmit);
                                return "g";
                            });
            RequestContext context = f.context(906L);
            var submit = executor.submit(() -> f.queue.submit(context));
            try {
                assertTrue(enteringQueue.await(2, TimeUnit.SECONDS));
                var future = context.data().getFuture();
                assertTrue(
                        cancel
                                ? future.cancel(false)
                                : future.completeExceptionally(
                                        new IllegalStateException("client failed")));
            } finally {
                continueSubmit.countDown();
            }
            assertTrue(submit.get(2, TimeUnit.SECONDS).isDone());
            f.queue.close();
            assertEquals(0, RequestProtocolTestSupport.queuedCount(f.queue));
            verify(f.router, never()).select(any(), nullable(String.class));
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void externalCompletionRemovesParkedEntryWithoutWaitingForCapacity(boolean cancel)
            throws Exception {
        try (Fixture f = new Fixture()) {
            when(f.router.select(any(), nullable(String.class)))
                    .thenReturn(
                            PlacementResult.blocked(
                                    new PlacementKey(
                                            org.flexlb.dao.route.RoleType.DECODE, "g", null)));
            var future = f.queue.submit(f.context(907L));
            RequestProtocolTestSupport.awaitGlobalCapacityWaiters(f.queue, 1);
            assertTrue(
                    cancel
                            ? future.cancel(false)
                            : future.completeExceptionally(
                                    new IllegalStateException("client failed")));
            RequestProtocolTestSupport.awaitCondition(
                    () -> RequestProtocolTestSupport.queuedCount(f.queue) == 0);
        }
    }

    @Test
    void registrationHookRunsBeforeSelectionAndNormalRequestsStillStart() throws Exception {
        try (Fixture f = new Fixture()) {
            AtomicBoolean listening = new AtomicBoolean();
            when(f.router.select(any(), nullable(String.class)))
                    .thenAnswer(
                            call -> {
                                assertTrue(
                                        listening.get(),
                                        "selection cannot precede listener installation");
                                return PlacementResult.rejected(
                                        Response.error(StrategyErrorType.RESOURCE_EXHAUSTED));
                            });
            RequestContext context = f.context(901L);
            var result =
                    f.queue.submit(
                            context,
                            () -> {
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
            var result =
                    f.queue.submit(
                            context,
                            () -> {
                                assertSame(f.queue, context.data().getScheduler());
                                assertNotNull(
                                        f.queue.cancel(902L, 0L, CancelReason.CLIENT_CANCELLED));
                            });
            assertFalse(result.get(2L, TimeUnit.SECONDS).isSuccess());
            assertEquals(0, RequestProtocolTestSupport.queuedCount(f.queue));
            verify(f.router, never()).select(any(), nullable(String.class));
        }
    }

    @Test
    void concurrentCancellationBeforeHookReturnsPreventsSelection() throws Exception {
        try (Fixture f = new Fixture();
                var executor = Executors.newSingleThreadExecutor()) {
            var registered = new CountDownLatch(1);
            var resume = new CountDownLatch(1);
            var submitted =
                    executor.submit(
                            () ->
                                    f.queue.submit(
                                            f.context(903L),
                                            () -> {
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
            var result =
                    f.queue.submit(
                            f.context(904L),
                            () -> {
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

    private static final class Fixture implements AutoCloseable {
        final FlexlbConfig config = SchedulingTestConfig.batchConfig();
        final RequestWorkerSelector router = mock(RequestWorkerSelector.class);
        final SchedulerRuntime runtime;
        final QueuedRequestScheduler queue;

        Fixture() {
            this(true);
        }

        Fixture(boolean start) {
            SchedulingTestConfig.useFifoQueue(config);
            var service = mock(ConfigService.class);
            when(service.loadBalanceConfig()).thenReturn(config);
            var batches = mock(DeliveryMetricsReporter.class);
            runtime =
                    new SchedulerRuntime(
                            new RequestRepository(),
                            mock(EndpointRegistry.class),
                            batches,
                            mock(RequestSchedulerReporter.class),
                            SchedulerTestSupport.dispatcher(),
                            service,
                            mock(EngineGrpcClient.class));
            queue =
                    new QueuedRequestScheduler(
                            config,
                            router,
                            batches,
                            mock(DecodeAdmission.class),
                            runtime,
                            new PlacementAvailability(),
                            SchedulingTestConfig.capacity(config),
                            org.mockito.Mockito.mock(
                                    org.flexlb.service.RecentCacheKeyTraceReporter.class));
            runtime.registerScheduler(queue);
            if (start) {
                queue.start();
            }
        }

        RequestContext context(long id) {
            return RequestProtocolTestSupport.context(config, id);
        }

        @Override
        public void close() {
            runtime.shutdown();
        }
    }
}
