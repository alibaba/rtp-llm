package org.flexlb.balance.scheduler;

import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.Request;
import org.junit.jupiter.api.Test;
import org.springframework.test.util.ReflectionTestUtils;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

class ExpirationTimerTest {
    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.EnumSource(ExpirationTimer.Kind.class)
    @org.junit.jupiter.api.Timeout(15)
    void realHandleRearmsAfterWallClockEarlyWakeAndConsumesTheReplacementAtItsDeadline(ExpirationTimer.Kind kind) throws Exception {
        var config = SchedulingTestConfig.newConfig();
        config.setScheduler(org.flexlb.config.ConfigTestSupport.direct());
        config.setDispatcher(org.flexlb.config.ConfigTestSupport.nonBatch());
        var context = RequestProtocolTestSupport.context(config, 9400L + kind.ordinal());
        config.getRequestLifecycle().getRequest().setTimeoutMs(60_000L);
        long nowMs = System.currentTimeMillis();
        context.data().setSchedulingMetadata(org.flexlb.dao.SchedulingMetadata.explicit(50,
                kind == ExpirationTimer.Kind.REQUEST ? nowMs + 20_000L : Long.MAX_VALUE));
        var service = mock(org.flexlb.config.ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        var owner = SchedulerTestSupport.create(service, mock(org.flexlb.service.monitor.DeliveryMetricsReporter.class),
                mock(org.flexlb.service.monitor.RequestSchedulerReporter.class), mock(org.flexlb.service.RecentCacheKeyTraceReporter.class));
        var runtime = owner.runtime;
        var timer = runtime.deadlines();
        ((ScheduledThreadPoolExecutor) ReflectionTestUtils.getField(timer, "executor")).shutdownNow();
        var executor = spy(new ScheduledThreadPoolExecutor(1));
        doAnswer(call -> mock(ScheduledFuture.class)).when(executor)
                .schedule(any(Runnable.class), anyLong(), eq(TimeUnit.MILLISECONDS));
        ReflectionTestUtils.setField(timer, "executor", executor);
        var wallClock = new java.util.concurrent.atomic.AtomicLong(nowMs);
        ReflectionTestUtils.setField(timer, "clock", (java.util.function.LongSupplier) wallClock::get);
        try {
            var future = RequestProtocolTestSupport.register(owner, context);
            if (kind == ExpirationTimer.Kind.VISIBILITY) {
                var endpoint = mock(org.flexlb.balance.endpoint.PrefillEndpoint.class);
                var route = SchedulingTestConfig.createRoute(context, new org.flexlb.dao.loadbalance.Response(),
                        null, null, endpoint, null, null, nowMs);
                RequestProtocolTestSupport.bind(owner, new RequestProtocolTestSupport.Registered(route, future));
                var claim = owner.claimDelivery(route, DeliveryClaimKind.ROUTE_DECISION, 0L, null);
                assertNotNull(claim);
                owner.onDeliveryEvidence(claim, new ContextResult.Prediction(claim, nowMs + 20_000L, nowMs));
            }
            var before = context.snapshot();
            var exact = before.deadlines().token(kind);
            assertNotNull(exact);
            var originalHandle = currentHandle(timer, context, kind);
            wallClock.set(exact.atMs() - 1_000L);
            ReflectionTestUtils.invokeMethod(originalHandle, "fire");
            SchedulerTestSupport.awaitRequestWork(runtime);
            var after = context.snapshot();
            var replacement = after.deadlines().token(kind);
            assertNotNull(replacement);
            assertNotSame(exact, replacement);
            assertEquals(exact.atMs(), replacement.atMs());
            assertTrue(replacement.version() > exact.version());
            assertNotSame(originalHandle, currentHandle(timer, context, kind));
            verify(executor).schedule(any(Runnable.class), eq(1_000L), eq(TimeUnit.MILLISECONDS));
            for (var other : ExpirationTimer.Kind.values()) {
                if (other != kind) { assertSame(before.deadlines().token(other), after.deadlines().token(other)); }
            }
            assertEquals(ContextResult.Phase.OPEN, after.phase());
            assertNull(after.cancellation());
            assertNull(after.outcome());
            assertFalse(after.close().expired());
            assertFalse(future.isDone());
            var replacementHandle = currentHandle(timer, context, kind);
            wallClock.set(replacement.atMs());
            ReflectionTestUtils.invokeMethod(originalHandle, "fire");
            assertSame(replacement, context.snapshot().deadlines().token(kind));
            ReflectionTestUtils.invokeMethod(replacementHandle, "fire");
            SchedulerTestSupport.awaitRequestWork(runtime);
            if (kind == ExpirationTimer.Kind.VISIBILITY) {
                assertTrue(context.snapshot().engine().visibilityOverdue());
                assertNull(context.snapshot().deadlines().visibility());
                assertFalse(future.isDone());
                owner.cancel(context, 0L, CancelReason.SHUTDOWN);
            }
            assertFalse(future.get(5, TimeUnit.SECONDS).isSuccess());
            SchedulerTestSupport.awaitRequestWork(runtime);
            assertEquals(ContextResult.Phase.CLOSED, context.snapshot().phase());
            assertEquals(0, runtime.requests().liveRequestCount());
            assertEquals(0, ReflectionTestUtils.getField(timer, "registrations"));
            assertEquals(0L, ReflectionTestUtils.getField(runtime, "pendingWork"));
        } finally { runtime.shutdown(); }
    }

    private static Object currentHandle(ExpirationTimer timer, RequestContext context, ExpirationTimer.Kind kind) {
        var slots = (Map<?, ?>) ReflectionTestUtils.getField(timer, "slots");
        var handles = (Map<?, ?>) ReflectionTestUtils.getField(slots.get(context), "handles");
        return handles.get(kind);
    }

    @Test
    void archiveDuringInstallationReturnsRegistrationEvenWhenDetachFails() {
        var requests = mock(RequestRepository.class);
        var context = registeredContext();
        var current = new java.util.concurrent.atomic.AtomicBoolean(true);
        when(requests.isCurrent(context)).thenAnswer(call -> current.get());
        var timer = new ExpirationTimer(requests);
        var original = (ScheduledThreadPoolExecutor) ReflectionTestUtils.getField(timer, "executor");
        original.shutdownNow();
        var executor = spy(new ScheduledThreadPoolExecutor(1));
        ReflectionTestUtils.setField(timer, "executor", executor);
        var task = mock(ScheduledFuture.class);
        var failure = new IllegalStateException("detach failed after archive");
        when(task.cancel(false)).thenThrow(failure);
        doAnswer(call -> { current.set(false); return task; }).when(executor)
                .schedule(any(Runnable.class), anyLong(), eq(TimeUnit.MILLISECONDS));
        try {
            var token = new ExpirationTimer.Token(context, ExpirationTimer.Kind.REQUEST, 1L, Long.MAX_VALUE);
            assertSame(failure, assertThrows(IllegalStateException.class,
                    () -> timer.update(context, new ExpirationTimer.Plan(1L, token, null, null))));
            assertEquals(0, ReflectionTestUtils.getField(timer, "registrations"));
            timer.close();
            assertTrue(executor.isTerminated());
            verify(task).cancel(false);
        } finally {
            // Keep a deliberately broken implementation from hanging the test's cleanup.
            ReflectionTestUtils.setField(timer, "registrations", 0);
            timer.close();
        }
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(booleans = {false, true})
    @org.junit.jupiter.api.Timeout(15)
    void concurrentCloseDrainsRegistrationAndSharesResultWithInterruptedWaiter(boolean cleanupFails) throws Exception {
        var access = mock(RequestRepository.class);
        var context = registeredContext();
        when(access.isCurrent(context)).thenReturn(true);
        var timer = new ExpirationTimer(access);
        var timerExecutor = spy(new ScheduledThreadPoolExecutor(1));
        ((ScheduledThreadPoolExecutor) ReflectionTestUtils.getField(timer, "executor")).shutdownNow();
        ReflectionTestUtils.setField(timer, "executor", timerExecutor);
        var installing = new CountDownLatch(1);
        var finishInstall = new CountDownLatch(1);
        var exact = new ExpirationTimer.Token(context, ExpirationTimer.Kind.REQUEST, 1L, Long.MAX_VALUE);
        var task = mock(ScheduledFuture.class);
        var cleanupFailure = cleanupFails ? new IllegalStateException("detach failed") : null;
        if (cleanupFails) { when(task.cancel(false)).thenThrow(cleanupFailure); }
        doAnswer(invocation -> {
            installing.countDown();
            assertTrue(finishInstall.await(5, TimeUnit.SECONDS));
            return task;
        }).when(timerExecutor).schedule(any(Runnable.class), anyLong(), eq(TimeUnit.MILLISECONDS));
        var owner = new AtomicReference<Thread>();
        var waiter = new AtomicReference<Thread>();
        var interrupted = new java.util.concurrent.atomic.AtomicBoolean();
        try (var executor = Executors.newFixedThreadPool(3)) {
            try {
                var registration = executor.submit(() -> {
                    timer.update(context, new ExpirationTimer.Plan(1L, exact, null, null));
                    return exact;
                });
                assertTrue(installing.await(5, TimeUnit.SECONDS));
                var first = executor.submit(() -> {
                    owner.set(Thread.currentThread());
                    return org.flexlb.util.Failures.run(null, timer::close);
                });
                RequestProtocolTestSupport.awaitCondition(() -> owner.get() != null && owner.get().getState() == Thread.State.WAITING);
                assertThrows(java.util.concurrent.RejectedExecutionException.class,
                        () -> timer.update(context, new ExpirationTimer.Plan(2L, exact, null, null)));
                var second = executor.submit(() -> {
                    waiter.set(Thread.currentThread());
                    Thread.currentThread().interrupt();
                    try { return org.flexlb.util.Failures.run(null, timer::close); }
                    finally { interrupted.set(Thread.currentThread().isInterrupted()); }
                });
                RequestProtocolTestSupport.awaitCondition(() -> waiter.get() != null && waiter.get().getState() == Thread.State.WAITING);
                assertFalse(first.isDone());
                assertFalse(second.isDone());
                finishInstall.countDown();
                assertSame(exact, registration.get(5, TimeUnit.SECONDS));
                assertSame(cleanupFailure, first.get(5, TimeUnit.SECONDS));
                assertSame(cleanupFailure, second.get(5, TimeUnit.SECONDS));
                assertTrue(interrupted.get());
                assertTrue(timerExecutor.isTerminated());
                assertTrue(timerExecutor.getQueue().isEmpty());
                assertSame(cleanupFailure, org.flexlb.util.Failures.run(null, timer::close));
                verify(task).cancel(false);
            } finally {
                finishInstall.countDown();
                org.flexlb.util.Failures.run(null, timer::close);
            }
        }
    }

    @Test void detachedCleanupAttemptsEveryDeadlineOnceDespiteFailures() {
        try (var fixture = new Fixture()) {
            var first = new IllegalStateException("inactivity");
            var second = new IllegalStateException("request");
            var third = new IllegalStateException("decision");
            when(fixture.inactivityTask.cancel(false)).thenThrow(first);
            when(fixture.requestTask.cancel(false)).thenThrow(second);
            when(fixture.decisionTask.cancel(false)).thenThrow(third);
            assertSame(first, assertThrows(IllegalStateException.class, () -> fixture.timer.remove(fixture.context)));
            assertEquals(List.of(second, third), List.of(first.getSuppressed()));
            fixture.timer.remove(fixture.context);
            fixture.assertCanceled();
            var order = inOrder(fixture.inactivityTask, fixture.requestTask, fixture.decisionTask);
            order.verify(fixture.inactivityTask).cancel(false);
            order.verify(fixture.requestTask).cancel(false);
            order.verify(fixture.decisionTask).cancel(false);
            order.verifyNoMoreInteractions();
        }
    }

    @Test void concurrentDetachedReleaseCancelsEachTaskOnceAndPreventsDeadlineConsumption() throws Exception {
        try (var fixture = new Fixture(); var executor = Executors.newFixedThreadPool(2)) {
            var entered = new CountDownLatch(1);
            var finish = new CountDownLatch(1);
            when(fixture.inactivityTask.cancel(false)).thenAnswer(invocation -> {
                entered.countDown();
                assertTrue(finish.await(5, TimeUnit.SECONDS));
                return true;
            });
            var first = executor.submit(() -> fixture.timer.remove(fixture.context));
            try {
                assertTrue(entered.await(5, TimeUnit.SECONDS));
                executor.submit(() -> fixture.timer.remove(fixture.context)).get(5, TimeUnit.SECONDS);
                fixture.assertCanceled();
            } finally { finish.countDown(); }
            first.get(5, TimeUnit.SECONDS);
            verify(fixture.inactivityTask).cancel(false);
            verify(fixture.requestTask).cancel(false);
            verify(fixture.decisionTask).cancel(false);
        }
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.MethodSource("deadlineRaces")
    void deadlineRegistrationCoversFireInstallAndCancelRaces(String kind, boolean earlyFire, boolean installed, boolean cancel) {
        var requests = mock(RequestRepository.class);
        var context = registeredContext();
        var owner = mock(AbstractRequestScheduler.class);
        when(context.data()).thenReturn(data(owner));
        when(requests.isCurrent(context)).thenReturn(true);
        var timer = new ExpirationTimer(requests);
        ((ScheduledThreadPoolExecutor) ReflectionTestUtils.getField(timer, "executor")).shutdownNow();
        var executor = spy(new ScheduledThreadPoolExecutor(1));
        var task = mock(ScheduledFuture.class);
        var callback = new AtomicReference<Runnable>();
        var tokenKind = kind.equals("DECISION") ? ExpirationTimer.Kind.VISIBILITY : ExpirationTimer.Kind.valueOf(kind);
        var token = new ExpirationTimer.Token(context, tokenKind, 1L, Long.MAX_VALUE);
        doAnswer(call -> {
            callback.set(call.getArgument(0));
            if (earlyFire) { callback.get().run(); }
            if (!installed || cancel) { timer.update(context, new ExpirationTimer.Plan(2L, null, null, null)); }
            return task;
        }).when(executor).schedule(any(Runnable.class), anyLong(), eq(TimeUnit.MILLISECONDS));
        ReflectionTestUtils.setField(timer, "executor", executor);
        try {
            timer.update(context, new ExpirationTimer.Plan(1L,
                    tokenKind == ExpirationTimer.Kind.REQUEST ? token : null,
                    tokenKind == ExpirationTimer.Kind.VISIBILITY ? token : null,
                    tokenKind == ExpirationTimer.Kind.INACTIVITY ? token : null));
            callback.get().run();
            callback.get().run();
            int expected = installed && !cancel ? 1 : 0;
            verify(owner, times(expected)).onDeadline(eq(token), anyLong());
            verify(task, times(!installed || cancel ? 1 : 0)).cancel(false);
        } finally { timer.close(); }
    }

    static java.util.stream.Stream<org.junit.jupiter.params.provider.Arguments> deadlineRaces() {
        return java.util.stream.Stream.of("REQUEST", "DECISION", "INACTIVITY").flatMap(kind ->
                java.util.stream.IntStream.range(0, 8).mapToObj(bits -> org.junit.jupiter.params.provider.Arguments.of(
                        kind, (bits & 1) != 0, (bits & 2) != 0, (bits & 4) != 0)));
    }
    private static RequestContext registeredContext() {
        var context = mock(RequestContext.class);
        var snapshot = mock(ContextResult.Snapshot.class);
        when(snapshot.key()).thenReturn(new ContextResult.Identity(1L, 1L));
        when(context.snapshot()).thenReturn(snapshot);
        return context;
    }
    private static RequestData data(AbstractRequestScheduler owner) {
        var data = new RequestData(new FlexlbConfig());
        var request = mock(Request.class);
        data.setRequest(request);
        data.setSchedulingMetadata(org.flexlb.dao.SchedulingMetadata.explicit(1, Long.MAX_VALUE));
        data.register(owner, new RequestFuture((kind, response, failure) -> false));
        return data;
    }
    private static final class Fixture implements AutoCloseable {
        private final RequestContext context = registeredContext();
        private final AbstractRequestScheduler owner = mock(AbstractRequestScheduler.class);
        private final ExpirationTimer timer;
        private final java.util.EnumMap<ExpirationTimer.Kind, Object> handles;
        private final ScheduledFuture<?> inactivityTask;
        private final ScheduledFuture<?> requestTask;
        private final ScheduledFuture<?> decisionTask;
        private Fixture() {
            var access = mock(RequestRepository.class);
            when(access.isCurrent(context)).thenReturn(true);
            when(context.data()).thenReturn(data(owner));
            timer = new ExpirationTimer(access);
            timer.update(context, new ExpirationTimer.Plan(1L,
                    new ExpirationTimer.Token(context, ExpirationTimer.Kind.REQUEST, 1L, Long.MAX_VALUE),
                    new ExpirationTimer.Token(context, ExpirationTimer.Kind.VISIBILITY, 1L, Long.MAX_VALUE),
                    new ExpirationTimer.Token(context, ExpirationTimer.Kind.INACTIVITY, 1L, Long.MAX_VALUE)));
            var slots = (Map<?, ?>) ReflectionTestUtils.getField(timer, "slots");
            handles = (java.util.EnumMap<ExpirationTimer.Kind, Object>) ReflectionTestUtils.getField(slots.get(context), "handles");
            requestTask = replaceTask(handles.get(ExpirationTimer.Kind.REQUEST));
            decisionTask = replaceTask(handles.get(ExpirationTimer.Kind.VISIBILITY));
            inactivityTask = replaceTask(handles.get(ExpirationTimer.Kind.INACTIVITY));
        }
        private static ScheduledFuture<?> replaceTask(Object handle) {
            ((ScheduledFuture<?>) ReflectionTestUtils.getField(handle, "scheduled")).cancel(false);
            var task = mock(ScheduledFuture.class);
            ReflectionTestUtils.setField(handle, "scheduled", task);
            return task;
        }
        private void assertCanceled() {
            for (Object handle : handles.values()) { ReflectionTestUtils.invokeMethod(handle, "fire"); }
            verify(owner, never()).onDeadline(any(), anyLong());
        }
        @Override public void close() { timer.close(); }
    }
}
