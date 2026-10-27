package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.config.ConfigService;
import org.flexlb.config.ConfigTestSupport;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.RecentCacheKeyTraceReporter;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.flexlb.balance.scheduler.ContextResult.EndEvidence;
import static org.flexlb.balance.scheduler.ContextResult.EndKind;
import static org.flexlb.balance.scheduler.ContextResult.EngineEvidence;
import static org.flexlb.balance.scheduler.ContextResult.EngineKind;
import static org.flexlb.balance.scheduler.ContextResult.Phase;
import static org.flexlb.balance.scheduler.ContextResult.ResponseKind;
import static org.flexlb.balance.scheduler.ContextResult.RouteDelivered;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyBoolean;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.any;
import static org.mockito.Mockito.anyBoolean;
import static org.mockito.Mockito.anyLong;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.doCallRealMethod;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@Timeout(15)
class RequestPublicationFailureOwnershipTest {
    @Test
    void deliveryPublicationFailureSettlesTheTransferredOwnership() throws Exception {
        try (var f = new Fixture()) {
            var context = spy(RequestProtocolTestSupport.context(f.config, 9305L));
            var future = RequestProtocolTestSupport.register(f.owner, context);
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
            var failure = new IllegalStateException("publication failed after transfer");
            doAnswer(
                            call -> {
                                call.callRealMethod();
                                throw failure;
                            })
                    .when(context)
                    .publishDelivery(any());
            try {
                assertSame(
                        failure,
                        assertThrows(
                                IllegalStateException.class,
                                () ->
                                        f.owner.claimDelivery(
                                                route,
                                                DeliveryClaimKind.ROUTE_DECISION,
                                                0L,
                                                null)));
                assertFalse(future.get(2, TimeUnit.SECONDS).isSuccess());
                SchedulerTestSupport.awaitRequestWork(f.runtime);
                assertEquals(Phase.CLOSED, context.snapshot().phase());
                verify(endpoint, times(1)).release(null, route);
            } finally {
                var delivery = context.snapshot().delivery();
                if (delivery != null && context.snapshot().phase() != Phase.CLOSED) {
                    f.owner.onDeliveryEvidence(
                            delivery.identity(),
                            new ContextResult.PreparationFailure(
                                    delivery.identity(),
                                    route,
                                    "test cleanup",
                                    System.currentTimeMillis()));
                }
            }
        }
    }

    @Test
    void resourceSettlementCannotOvertakeTheRunningCloseNotification() throws Exception {
        try (var f = new Fixture();
                var workers = java.util.concurrent.Executors.newFixedThreadPool(2)) {
            var context = RequestProtocolTestSupport.context(f.config, 9304L);
            var future = RequestProtocolTestSupport.register(f.owner, context);
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
            when(endpoint.release(null, route))
                    .thenAnswer(
                            call -> {
                                released.countDown();
                                return true;
                            });
            var originalTimer = f.runtime.deadlines();
            ReflectionTestUtils.setField(f.runtime, "timer", timer);
            var closing =
                    workers.submit(
                            () ->
                                    f.owner.closeRequest(
                                            context,
                                            EndEvidence.failure(
                                                    route,
                                                    EndKind.FAILURE,
                                                    StrategyErrorType.DISPATCH_FAILED,
                                                    "test")));
            java.util.concurrent.Future<?> settlement = null;
            try {
                assertTrue(notificationEntered.await(5, TimeUnit.SECONDS));
                settlement = workers.submit(() -> f.owner.scheduleClose(context));
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
                ReflectionTestUtils.setField(f.runtime, "timer", originalTimer);
            }
            assertEquals(Phase.CLOSED, context.snapshot().phase());
            verify(endpoint, times(1)).release(null, route);
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void closePreparationFailureReturnsPublicationAndFrozenOutcomeStillCompletes(
            boolean beforeFreeze) throws Exception {
        try (var f = new Fixture()) {
            var context = spy(RequestProtocolTestSupport.context(f.config, 9301L));
            var future = RequestProtocolTestSupport.register(f.owner, context);
            var endpoint = mock(PrefillEndpoint.class);
            var route =
                    spy(
                            SchedulingTestConfig.createRoute(
                                    context,
                                    new Response(),
                                    null,
                                    null,
                                    endpoint,
                                    null,
                                    null,
                                    System.currentTimeMillis()));
            RequestProtocolTestSupport.bind(
                    f.owner, new RequestProtocolTestSupport.Registered(route, future));
            var failure = new IllegalStateException("response preparation failed");
            if (beforeFreeze) {
                doThrow(failure).when(context).beginClose(any(), anyLong());
            } else {
                doThrow(failure).when(route).successResponse(anyBoolean());
            }
            var end =
                    new EndEvidence(
                            route,
                            EndKind.WORKER,
                            RoleType.PDFUSION,
                            true,
                            0L,
                            StrategyErrorType.DISPATCH_FAILED,
                            "worker completed",
                            false,
                            false);
            assertSame(
                    failure,
                    assertThrows(
                            IllegalStateException.class, () -> f.owner.closeRequest(context, end)));
            if (beforeFreeze) {
                assertNull(context.snapshot().outcome());
                assertFalse(future.isDone());
                assertEquals(0, registrations(f.runtime).size());
                doCallRealMethod().when(context).beginClose(any(), anyLong());
                f.owner.closeRequest(context, end);
                assertTrue(future.get(5, TimeUnit.SECONDS).isSuccess());
            } else {
                assertSame(
                        failure,
                        assertThrows(
                                        ExecutionException.class,
                                        () -> future.get(5, TimeUnit.SECONDS))
                                .getCause());
                assertEquals(EndKind.WORKER, context.snapshot().outcome().kind());
                assertEquals("worker completed", context.snapshot().outcome().detail());
            }
            SchedulerTestSupport.awaitRequestWork(f.runtime);
            RequestProtocolTestSupport.awaitCondition(() -> registrations(f.runtime).isEmpty());
            assertEquals(Phase.CLOSED, context.snapshot().phase());
            assertEquals(0, f.runtime.requests().liveRequestCount());
            verify(endpoint).release(null, route);
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void acknowledgementPreparationFailureCompletesFutureAndKeepsPhysicalOwnershipUntilWorkerProof(
            boolean timerFailure) throws Exception {
        try (var f = new Fixture()) {
            var context = RequestProtocolTestSupport.context(f.config, 9302L);
            var future = RequestProtocolTestSupport.register(f.owner, context);
            var endpoint = mock(PrefillEndpoint.class);
            var route =
                    spy(
                            SchedulingTestConfig.createRoute(
                                    context,
                                    new Response(),
                                    null,
                                    null,
                                    endpoint,
                                    null,
                                    null,
                                    System.currentTimeMillis()));
            RequestProtocolTestSupport.bind(
                    f.owner, new RequestProtocolTestSupport.Registered(route, future));
            var delivery = f.owner.claimDelivery(route, DeliveryClaimKind.ROUTE_DECISION, 0L, null);
            assertNotNull(delivery);
            var originalTimer = f.runtime.deadlines();
            var failure = new IllegalStateException("ACK preparation failed");
            if (timerFailure) {
                var timer = spy(originalTimer);
                doThrow(failure).when(timer).update(eq(context), any());
                ReflectionTestUtils.setField(f.runtime, "timer", timer);
            } else {
                doThrow(failure).when(route).successResponse(anyBoolean());
            }
            try {
                assertDoesNotThrow(
                        () ->
                                f.owner.onDeliveryEvidence(
                                        delivery,
                                        new RouteDelivered(
                                                delivery, 0L, System.currentTimeMillis())));
                assertSame(failure, SchedulerTestSupport.failure(f.owner));
                assertSame(
                        failure,
                        assertThrows(
                                        ExecutionException.class,
                                        () -> future.get(5, TimeUnit.SECONDS))
                                .getCause());
                RequestProtocolTestSupport.awaitCondition(() -> registrations(f.runtime).isEmpty());
                assertTrue(context.snapshot().delivery().acknowledged());
                assertEquals(ResponseKind.ACK, context.snapshot().selectedResponse().kind());
                assertEquals(Phase.OPEN, context.snapshot().phase());
                assertSame(route, context.snapshot().route());
                assertEquals(1, f.runtime.requests().liveRequestCount());
                verify(endpoint, never()).release(org.mockito.ArgumentMatchers.isNull(), any());
            } finally {
                ReflectionTestUtils.setField(f.runtime, "timer", originalTimer);
                doCallRealMethod().when(route).successResponse(anyBoolean());
            }
            originalTimer.update(context, context.snapshot().deadlines());
            f.owner.onEngineEvidence(
                    context,
                    new EngineEvidence(
                            route,
                            null,
                            RoleType.PDFUSION,
                            EngineKind.COMPLETED,
                            0L,
                            null,
                            System.currentTimeMillis()));
            SchedulerTestSupport.awaitRequestWork(f.runtime);
            assertEquals(Phase.CLOSED, context.snapshot().phase());
            assertEquals(0, f.runtime.requests().liveRequestCount());
            assertSame(failure, assertThrows(ExecutionException.class, future::get).getCause());
        }
    }

    @Test
    void rejectedCompletionUsesNewPublicationAndLateOldTaskCannotReviveItsCandidate()
            throws Exception {
        try (var f = new Fixture()) {
            var context = RequestProtocolTestSupport.context(f.config, 9305L);
            var future = RequestProtocolTestSupport.register(f.owner, context);
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
            var delivery = f.owner.claimDelivery(route, DeliveryClaimKind.ROUTE_DECISION, 0L, null);
            var publisher = SchedulerTestSupport.responseCompletions(f.runtime);
            var originalWorkers = ReflectionTestUtils.getField(publisher, "completionWorkers");
            var workers = mock(java.util.concurrent.ThreadPoolExecutor.class);
            var oldTask = new java.util.concurrent.atomic.AtomicReference<Runnable>();
            doAnswer(
                            call -> {
                                oldTask.compareAndSet(null, call.getArgument(0));
                                throw new java.util.concurrent.RejectedExecutionException(
                                        "completion executor rejected");
                            })
                    .when(workers)
                    .execute(any(Runnable.class));
            ReflectionTestUtils.setField(publisher, "completionWorkers", workers);
            try {
                assertDoesNotThrow(
                        () ->
                                f.owner.onDeliveryEvidence(
                                        delivery,
                                        new RouteDelivered(
                                                delivery, 0L, System.currentTimeMillis())));
                var failure =
                        assertInstanceOf(
                                java.util.concurrent.RejectedExecutionException.class,
                                assertThrows(
                                                ExecutionException.class,
                                                () -> future.get(5, TimeUnit.SECONDS))
                                        .getCause());
                RequestProtocolTestSupport.awaitCondition(() -> registrations(f.runtime).isEmpty());
                assertNull(context.snapshot().selectedResponse());
                assertEquals(Phase.OPEN, context.snapshot().phase());
                assertEquals(1, f.runtime.requests().liveRequestCount());
                var before = context.snapshot();
                assertNotNull(oldTask.get());
                oldTask.get().run();
                assertSame(before, context.snapshot());
                assertSame(failure, assertThrows(ExecutionException.class, future::get).getCause());
                verify(endpoint, never()).release(org.mockito.ArgumentMatchers.isNull(), any());
            } finally {
                ReflectionTestUtils.setField(publisher, "completionWorkers", originalWorkers);
            }
            f.owner.onEngineEvidence(
                    context,
                    new EngineEvidence(
                            route,
                            null,
                            RoleType.PDFUSION,
                            EngineKind.COMPLETED,
                            0L,
                            null,
                            System.currentTimeMillis()));
            SchedulerTestSupport.awaitRequestWork(f.runtime);
            assertEquals(Phase.CLOSED, context.snapshot().phase());
            assertEquals(0, f.runtime.requests().liveRequestCount());
        }
    }

    @Test
    void externalCancelCompletesWhileTheAdmissionOwnerFinishes() throws Exception {
        try (var f = new Fixture()) {
            var context = RequestProtocolTestSupport.context(f.config, 9303L);
            var future = RequestProtocolTestSupport.register(f.owner, context);
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
            try (var admission = f.owner.claimAdmissionHandle(9303L, future)) {
                assertNotNull(admission);
                assertEquals(
                        org.flexlb.balance.PlacementResult.Status.SUCCESS,
                        f.owner.commitRoute(
                                route, RequestProtocolTestSupport.publication(() -> true)));
                try {
                    assertTrue(future.cancel(false));
                    assertTrue(future.isCancelled());
                    assertEquals(
                            ResponseKind.CANCELLATION,
                            context.snapshot().selectedResponse().kind());
                    assertEquals(
                            CancelReason.CLIENT_CANCELLED,
                            context.snapshot().cancellation().reason());
                    assertEquals(0, registrations(f.runtime).size());
                    assertNotNull(
                            context.snapshot().placement(),
                            "the real admission owner must remain until it finishes");
                } finally {
                    admission.finish();
                }
            }
            SchedulerTestSupport.awaitRequestWork(f.runtime);
            assertEquals(Phase.CLOSED, context.snapshot().phase());
            assertEquals(0, f.runtime.requests().liveRequestCount());
            assertTrue(future.isCancelled());
            verify(endpoint).release(null, route);
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void deferredSubmissionFailureDistinguishesUnusedPublicationFromAnAlreadyRunningOwner(
            boolean alreadyRunning) throws Exception {
        try (var f = new Fixture()) {
            var context = RequestProtocolTestSupport.context(f.config, 9304L);
            var future = RequestProtocolTestSupport.register(f.owner, context);
            var source = f.runtime.registerRequestWork(context);
            SchedulerRuntime.WorkRegistration accepted;
            try {
                accepted = spy(source);
            } catch (RuntimeException | Error failure) {
                source.close();
                throw failure;
            }
            try (accepted) {
                var runtime = f.runtime;
                var firstRegistration = new AtomicBoolean(true);
                doAnswer(
                                call ->
                                        firstRegistration.getAndSet(false)
                                                ? accepted
                                                : call.callRealMethod())
                        .when(runtime)
                        .registerRequestWork(context);
                ReflectionTestUtils.setField(f.owner, "runtime", runtime);
                var originalTimer = f.runtime.deadlines();
                var timer = spy(originalTimer);
                var started = new CountDownLatch(1);
                var finish = new CountDownLatch(1);
                var block = new AtomicBoolean(alreadyRunning);
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
                ReflectionTestUtils.setField(runtime, "timer", timer);
                var failure = new IllegalStateException("submit failed");
                doAnswer(
                                call -> {
                                    if (alreadyRunning) {
                                        call.callRealMethod();
                                        assertTrue(started.await(5, TimeUnit.SECONDS));
                                    }
                                    throw failure;
                                })
                        .when(accepted)
                        .submit(any(Runnable.class));
                try {
                    var end =
                            new EndEvidence(
                                    null,
                                    EndKind.INACTIVITY,
                                    null,
                                    false,
                                    0L,
                                    StrategyErrorType.BATCH_SLO_EXPIRED,
                                    "request inactive",
                                    false,
                                    false);
                    assertSame(
                            failure,
                            assertThrows(
                                    IllegalStateException.class,
                                    () -> f.owner.closeRequest(context, end)));
                    if (alreadyRunning) {
                        assertTrue(accepted.hasTaskOwner());
                        assertEquals(1, registrations(f.runtime).size());
                        assertFalse(future.isDone());
                        assertFalse(context.snapshot().close().notificationsAttempted());
                        finish.countDown();
                        assertFalse(future.get(5, TimeUnit.SECONDS).isSuccess());
                    } else {
                        assertFalse(accepted.hasTaskOwner());
                        assertSame(
                                failure,
                                assertThrows(
                                                ExecutionException.class,
                                                () -> future.get(5, TimeUnit.SECONDS))
                                        .getCause());
                    }
                    SchedulerTestSupport.awaitRequestWork(f.runtime);
                    RequestProtocolTestSupport.awaitCondition(
                            () -> registrations(f.runtime).isEmpty());
                    assertEquals(EndKind.INACTIVITY, context.snapshot().outcome().kind());
                    assertEquals(Phase.CLOSED, context.snapshot().phase());
                    assertEquals(0, f.runtime.requests().liveRequestCount());
                } finally {
                    finish.countDown();
                    doCallRealMethod().when(runtime).registerRequestWork(context);
                    ReflectionTestUtils.setField(runtime, "timer", originalTimer);
                    ReflectionTestUtils.setField(f.owner, "runtime", f.runtime);
                }
            }
        }
    }

    @SuppressWarnings("unchecked")
    private static Set<ResponsePublication> registrations(SchedulerRuntime runtime) {
        var registered =
                (Set<ResponsePublication>)
                        ReflectionTestUtils.getField(
                                SchedulerTestSupport.responseCompletions(runtime), "registrations");
        synchronized (registered) {
            return Set.copyOf(registered);
        }
    }

    private static final class Fixture implements AutoCloseable {
        private final org.flexlb.config.FlexlbConfig config = SchedulingTestConfig.newConfig();
        private final AbstractRequestScheduler owner;
        private final SchedulerRuntime runtime;

        private Fixture() {
            config.setScheduler(ConfigTestSupport.direct());
            config.setDispatcher(ConfigTestSupport.nonBatch());
            var service = mock(ConfigService.class);
            when(service.loadBalanceConfig()).thenReturn(config);
            owner =
                    SchedulerTestSupport.create(
                            service,
                            mock(DeliveryMetricsReporter.class),
                            mock(RequestSchedulerReporter.class),
                            mock(RecentCacheKeyTraceReporter.class));
            runtime = owner.runtime;
        }

        @Override
        public void close() {
            // A failed mutant must report its leaked obligation without hanging fixture shutdown.
            for (var registration : registrations(runtime).toArray(ResponsePublication[]::new)) {
                registration.close();
            }
            org.flexlb.util.Failures.run(null, runtime::shutdown);
        }
    }
}
