package org.flexlb.balance.schedule;

import org.flexlb.balance.metrics.DeliveryMetricsReporter;
import org.flexlb.balance.metrics.RequestSchedulerReporter;
import org.flexlb.balance.requestlog.TheoryCacheHitReporter;
import org.flexlb.config.ConfigService;
import org.flexlb.config.ConfigTestSupport;
import org.flexlb.worker.endpoint.PrefillEndpoint;
import org.flexlb.worker.endpoint.Response;
import org.flexlb.worker.endpoint.StrategyErrorType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.flexlb.balance.schedule.RequestContext.EndKind;
import static org.flexlb.balance.schedule.RequestContext.Phase;
import static org.flexlb.balance.schedule.RequestContext.RequestEndEvent;
import static org.flexlb.balance.schedule.RequestContext.ResponseKind;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.any;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Checks that publication failures preserve exact cleanup ownership and release response registrations.
 */
@Timeout(15)
class RequestPublicationFailureOwnershipTest {
    @Test
    void lastWorkerReportArchivesRepositoryAndForgetsCancellationTask() throws Exception {
        try (var f = new Fixture()) {
            var context = RequestProtocolTestSupport.context(f.config, 9398L);
            var future = RequestProtocolTestSupport.submitForManualPlacement(f.owner, context);
            var endpoint = mock(PrefillEndpoint.class);
            var route =
                    SchedulingTestConfig.createRoute(
                            context,
                            new Response(),
                            null,
                            null,
                            endpoint,
                            null,
                            null,
                            System.currentTimeMillis());
            RequestProtocolTestSupport.bind(
                    f.owner, new RequestProtocolTestSupport.Registered(route, future));
            context.endRequest(
                    RequestEndEvent.failure(
                            route,
                            EndKind.FAILURE,
                            StrategyErrorType.DISPATCH_FAILED,
                            "execution failed"),
                    System.currentTimeMillis(),
                    null);
            context.recordCleanupProgress(route, false, true, true, System.currentTimeMillis());
            var original = f.runtime.dispatcher;
            var cancellations = mock(EngineCancellation.class);
            SchedulerTestServices.setDependency(f.runtime, "dispatcher", cancellations);
            try {
                f.owner.onRequestEvent(
                        context,
                        new RequestContext.RequestEvent(
                                route,
                                null,
                                org.flexlb.config.service.RoleType.PREFILL,
                                endpoint,
                                org.flexlb.worker.endpoint.RequestEventType.COMPLETED,
                                0L,
                                "prefill finished",
                                System.currentTimeMillis()));
                RequestProtocolTestSupport.awaitCondition(
                        () -> f.owner.repository.findTerminal("9398") != null);
                assertEquals(Phase.CLOSED, context.snapshot().phase());
                verify(cancellations).forget(route);
            } finally {
                SchedulerTestServices.setDependency(f.runtime, "dispatcher", original);
            }
        }
    }

    @Test
    void responseReservationRejectionKeepsRequestRecoverableWithoutRecursiveClose()
            throws Exception {
        try (var f = new Fixture()) {
            var context = RequestProtocolTestSupport.context(f.config, 9399L);
            var future = RequestProtocolTestSupport.submitForManualPlacement(f.owner, context);
            var attempt = SchedulerTestSupport.claim(f.owner, "9399", future);
            attempt.requestContext()
                    .getRequestScheduler()
                    .finishSelectionWithoutCommit(attempt, null);
            var original = SchedulerTestSupport.responseCompletions(f.runtime);
            var completions = spy(original);
            var accepting = new AtomicBoolean(false);
            doAnswer(call -> accepting.get() ? call.callRealMethod() : false)
                    .when(completions)
                    .register();
            SchedulerTestServices.setDependency(f.runtime, "responseCompletions", completions);
            try {
                assertThrows(
                        IllegalStateException.class,
                        () ->
                                f.owner.closeRequest(
                                        context,
                                        RequestEndEvent.failure(
                                                null,
                                                EndKind.FAILURE,
                                                StrategyErrorType.DISPATCH_FAILED,
                                                "execution failed")));
                assertEquals(Phase.ENDING, context.snapshot().phase());
                assertFalse(future.isDone());
                org.junit.jupiter.api.Assertions.assertNull(context.snapshot().responseForFuture());
                org.junit.jupiter.api.Assertions.assertNull(
                        context.snapshot().requestRecordForArchive());
                assertNotNull(context.snapshot().deadlines().inactivity());
                accepting.set(true);
                f.owner.onRequestStateChanged(context);
                assertFalse(future.get(3, TimeUnit.SECONDS).isSuccess());
                RequestProtocolTestSupport.awaitCondition(
                        () -> context.snapshot().phase() == Phase.CLOSED);
                verify(completions, times(2)).register();
            } finally {
                SchedulerTestServices.setDependency(f.runtime, "responseCompletions", original);
            }
        }
    }

    @Test
    void resourceSettlementCannotOvertakeTheRunningCloseNotification() throws Exception {
        try (var f = new Fixture();
                var workers = java.util.concurrent.Executors.newFixedThreadPool(2)) {
            var context = RequestProtocolTestSupport.context(f.config, 9304L);
            var future = RequestProtocolTestSupport.submitForManualPlacement(f.owner, context);
            var endpoint = mock(PrefillEndpoint.class);
            var route =
                    SchedulingTestConfig.createRoute(
                            context,
                            new Response(),
                            null,
                            null,
                            endpoint,
                            null,
                            null,
                            System.currentTimeMillis());
            RequestProtocolTestSupport.bind(
                    f.owner, new RequestProtocolTestSupport.Registered(route, future));
            var notificationEntered = new CountDownLatch(1);
            var finishNotification = new CountDownLatch(1);
            var released = new CountDownLatch(1);
            var needsTimer = new AtomicBoolean(true);
            var timer = mock(ExpirationTimer.class);
            when(timer.needsUpdate(eq(context), any())).thenAnswer(call -> needsTimer.get());
            doAnswer(
                            call -> {
                                notificationEntered.countDown();
                                assertTrue(finishNotification.await(5, TimeUnit.SECONDS));
                                needsTimer.set(false);
                                return null;
                            })
                    .when(timer)
                    .update(eq(context), any());
            when(endpoint.releaseRequest(route))
                    .thenAnswer(
                            call -> {
                                released.countDown();
                                return true;
                            });
            var originalTimer = f.runtime.deadlines();
            org.flexlb.balance.schedule.SchedulerTestServices.setDependency(
                    f.runtime, "timer", timer);
            var closing =
                    workers.submit(
                            () ->
                                    f.owner.closeRequest(
                                            context,
                                            RequestEndEvent.failure(
                                                    route,
                                                    EndKind.FAILURE,
                                                    StrategyErrorType.DISPATCH_FAILED,
                                                    "test")));
            java.util.concurrent.Future<?> settlement = null;
            try {
                assertTrue(notificationEntered.await(5, TimeUnit.SECONDS));
                settlement = workers.submit(() -> f.owner.closeRequest(context, null));
                assertFalse(
                        released.await(250, TimeUnit.MILLISECONDS),
                        "one request must serialize notification and resource settlement");
            } finally {
                finishNotification.countDown();
                closing.get(5, TimeUnit.SECONDS);
                if (settlement != null) {
                    settlement.get(5, TimeUnit.SECONDS);
                }
                SchedulerTestSupport.awaitRequestWork(f.runtime);
                org.flexlb.balance.schedule.SchedulerTestServices.setDependency(
                        f.runtime, "timer", originalTimer);
            }
            assertEquals(Phase.CLOSED, context.snapshot().phase());
            verify(endpoint, times(1)).releaseRequest(route);
        }
    }

    @Test
    void externalCancelBeforeCommitCompletesWhileSelectionFinishes() throws Exception {
        try (var f = new Fixture()) {
            var context = RequestProtocolTestSupport.context(f.config, 9303L);
            var future = RequestProtocolTestSupport.submitForManualPlacement(f.owner, context);
            var endpoint = mock(PrefillEndpoint.class);
            var route =
                    SchedulingTestConfig.createRoute(
                            context,
                            new Response(),
                            null,
                            null,
                            endpoint,
                            null,
                            null,
                            System.currentTimeMillis());
            var admission = SchedulerTestSupport.claim(f.owner, "9303", future);
            {
                assertNotNull(admission);
                context.registerSelectedRoute(admission, route);
                try {
                    assertTrue(future.cancel(false));
                    assertTrue(future.isCancelled());
                    assertEquals(
                            ResponseKind.CANCELLATION,
                            context.snapshot().responseForFuture().responseKind());
                    assertEquals(
                            CancelReason.CLIENT_CANCELLED, context.snapshot().cancellationReason());
                    assertEquals(0, registrations(f.runtime));
                    assertNotNull(
                            context.currentSelectionAttempt(),
                            "the real admission owner must remain until it finishes");
                } finally {
                    admission
                            .requestContext()
                            .getRequestScheduler()
                            .finishSelectionWithoutCommit(admission, null);
                }
            }
            SchedulerTestSupport.awaitRequestWork(f.runtime);
            assertEquals(Phase.CLOSED, context.snapshot().phase());
            assertEquals(0, f.runtime.requests().liveRequestCount());
            assertTrue(future.isCancelled());
            verify(endpoint).releaseRequest(route);
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void closeSubmissionFailureRetainsCleanupWithoutReplacingCommittedResponse(
            boolean startedBeforeFailure) throws Exception {
        try (var f = new Fixture()) {
            var context = RequestProtocolTestSupport.context(f.config, 9304L);
            var future = RequestProtocolTestSupport.submitForManualPlacement(f.owner, context);
            var endpoint = mock(PrefillEndpoint.class);
            var route =
                    SchedulingTestConfig.createRoute(
                            context,
                            new Response(),
                            null,
                            null,
                            endpoint,
                            null,
                            null,
                            System.currentTimeMillis());
            RequestProtocolTestSupport.bind(
                    f.owner, new RequestProtocolTestSupport.Registered(route, future));
            var workers =
                    (java.util.concurrent.ExecutorService)
                            ReflectionTestUtils.getField(f.runtime.tasks, "requestWorkers");
            var dispatch = mock(java.util.concurrent.ExecutorService.class);
            var failure =
                    new java.util.concurrent.RejectedExecutionException("close dispatch rejected");
            var originalTimer = f.runtime.deadlines();
            var timer = spy(originalTimer);
            var started = new CountDownLatch(1);
            var finish = new CountDownLatch(1);
            var block = new AtomicBoolean(startedBeforeFailure);
            doAnswer(
                            call -> {
                                if (block.getAndSet(false)) {
                                    started.countDown();
                                    assertTrue(finish.await(5, TimeUnit.SECONDS));
                                }
                                return call.callRealMethod();
                            })
                    .when(timer)
                    .update(eq(context), any());
            SchedulerTestServices.setDependency(f.runtime, "timer", timer);
            doAnswer(
                            call -> {
                                if (startedBeforeFailure) {
                                    workers.execute(call.getArgument(0));
                                    assertTrue(started.await(5, TimeUnit.SECONDS));
                                }
                                throw failure;
                            })
                    .when(dispatch)
                    .execute(any(Runnable.class));
            ReflectionTestUtils.setField(f.runtime.tasks, "requestWorkers", dispatch);
            try {
                var end =
                        RequestEndEvent.failure(
                                route,
                                EndKind.INACTIVITY,
                                StrategyErrorType.SCHEDULING_DEADLINE_EXCEEDED,
                                "request inactive");
                assertSame(
                        failure,
                        assertThrows(
                                java.util.concurrent.RejectedExecutionException.class,
                                () -> f.owner.closeRequest(context, end)));
                verify(dispatch, times(1)).execute(any(Runnable.class));
                if (startedBeforeFailure) {
                    assertTrue(future.get(5, TimeUnit.SECONDS).isSuccess());
                    assertEquals(0, registrations(f.runtime));
                    assertFalse(context.snapshot().notificationsDone());
                    verify(endpoint, never()).releaseRequest(route);
                    finish.countDown();
                } else {
                    assertTrue(future.get(5, TimeUnit.SECONDS).isSuccess());
                    f.runtime.tasks.awaitIdle();
                    assertEquals(0, registrations(f.runtime));
                    assertEquals(Phase.ENDING, context.snapshot().phase());
                    assertSame(route, context.snapshot().route());
                    assertTrue(f.runtime.requests().isCurrent(context));
                    verify(endpoint, never()).releaseRequest(route);
                    ReflectionTestUtils.setField(f.runtime.tasks, "requestWorkers", workers);
                    f.owner.closeRequest(context, null);
                }
                if (startedBeforeFailure) {
                    assertTrue(future.get(5, TimeUnit.SECONDS).isSuccess());
                }
                f.runtime.tasks.awaitIdle();
                RequestProtocolTestSupport.awaitCondition(() -> registrations(f.runtime) == 0);
                assertEquals(EndKind.INACTIVITY, context.snapshot().requestEndResult().endKind());
                assertEquals(Phase.CLOSED, context.snapshot().phase());
                assertEquals(0, f.runtime.requests().liveRequestCount());
                verify(endpoint, times(1)).releaseRequest(route);
                if (startedBeforeFailure) {
                    assertDoesNotThrow(
                            () -> SchedulerTestServices.throwIfFailed(f.runtime.tasks),
                            "the caller's submission exception is not a rejection of an executing task");
                } else {
                    assertSame(
                            failure,
                            assertThrows(
                                    java.util.concurrent.RejectedExecutionException.class,
                                    () -> SchedulerTestServices.throwIfFailed(f.runtime.tasks)));
                }
            } finally {
                finish.countDown();
                ReflectionTestUtils.setField(f.runtime.tasks, "requestWorkers", workers);
                SchedulerTestServices.setDependency(f.runtime, "timer", originalTimer);
            }
        }
    }

    private static int registrations(SchedulerTestServices runtime) {
        var executorOwner = SchedulerTestSupport.responseCompletions(runtime);
        var lock = ReflectionTestUtils.getField(executorOwner, "completionLock");
        synchronized (lock) {
            return (int) ReflectionTestUtils.getField(executorOwner, "pendingCompletions");
        }
    }

    /**
     * Builds a direct scheduler for publication-failure tests and checks response registrations are
     * released at shutdown.
     */
    private static final class Fixture implements AutoCloseable {
        private final org.flexlb.config.FlexlbConfig config = SchedulingTestConfig.newConfig();
        private final AbstractRequestScheduler owner;
        private final SchedulerTestServices runtime;

        private Fixture() {
            config.setScheduler(ConfigTestSupport.direct());
            var service = mock(ConfigService.class);
            when(service.loadBalanceConfig()).thenReturn(config);
            owner =
                    SchedulerTestSupport.create(
                            service,
                            mock(DeliveryMetricsReporter.class),
                            mock(RequestSchedulerReporter.class),
                            mock(TheoryCacheHitReporter.class));
            runtime = org.flexlb.balance.schedule.SchedulerTestSupport.runtime(owner);
        }

        @Override
        public void close() {
            // A failed mutant must report its leaked obligation without hanging fixture shutdown.
            assertEquals(
                    0,
                    registrations(runtime),
                    "response operations must release their registrations");
            org.flexlb.util.Failures.run(null, () -> SchedulerTestSupport.shutdown(runtime));
        }
    }
}
