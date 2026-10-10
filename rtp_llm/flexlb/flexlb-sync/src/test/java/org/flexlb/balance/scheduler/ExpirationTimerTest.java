package org.flexlb.balance.scheduler;

import org.junit.jupiter.api.Test;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.List;
import java.util.OptionalLong;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.inOrder;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class ExpirationTimerTest {
    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(booleans = { false, true })
    @org.junit.jupiter.api.Timeout(15)
    void concurrentCloseDrainsRegistrationAndSharesResultWithInterruptedWaiter(boolean cleanupFails) throws Exception {
        var access = mock(RequestRepository.class);
        var context = mock(RequestContext.class);
        when(access.isCurrent(context)).thenReturn(true);
        when(access.snapshotActive()).thenReturn(List.of(context));
        var timer = new ExpirationTimer(access);
        var timerExecutor = (java.util.concurrent.ScheduledThreadPoolExecutor)
                ReflectionTestUtils.getField(timer, "executor");
        CountDownLatch installing = new CountDownLatch(1);
        CountDownLatch finishInstall = new CountDownLatch(1);
        var exact = new java.util.concurrent.atomic.AtomicReference<ExpirationTimer.RequestDeadline>();
        when(context.installRequestDeadline(any())).thenAnswer(invocation -> {
            exact.set(invocation.getArgument(0));
            installing.countDown();
            assertTrue(finishInstall.await(5, TimeUnit.SECONDS));
            return true;
        });
        var cleanupFailure = cleanupFails ? new IllegalStateException("detach failed") : null;
        when(context.detachDeadlines()).thenAnswer(invocation -> {
            if (cleanupFailure != null) { throw cleanupFailure; }
            return new ExpirationTimer.DetachedDeadlines(exact.get(), null, null);
        });
        var owner = new java.util.concurrent.atomic.AtomicReference<Thread>();
        var waiter = new java.util.concurrent.atomic.AtomicReference<Thread>();
        var interrupted = new java.util.concurrent.atomic.AtomicBoolean();
        try (var executor = Executors.newFixedThreadPool(3)) {
            try {
                var registration = executor.submit(() -> timer.scheduleRequestDeadline(context, Long.MAX_VALUE));
                assertTrue(installing.await(5, TimeUnit.SECONDS));
                var first = executor.submit(() -> {
                    owner.set(Thread.currentThread());
                    return org.flexlb.util.Failures.run(null, timer::close);
                });
                RequestProtocolTestSupport.awaitCondition(() -> owner.get() != null
                        && owner.get().getState() == Thread.State.WAITING);
                assertThrows(java.util.concurrent.RejectedExecutionException.class,
                        () -> timer.scheduleRequestDeadline(mock(RequestContext.class), Long.MAX_VALUE));
                var second = executor.submit(() -> {
                    waiter.set(Thread.currentThread());
                    Thread.currentThread().interrupt();
                    try { return org.flexlb.util.Failures.run(null, timer::close); }
                    finally { interrupted.set(Thread.currentThread().isInterrupted()); }
                });
                RequestProtocolTestSupport.awaitCondition(() -> waiter.get() != null
                        && waiter.get().getState() == Thread.State.WAITING);
                assertFalse(first.isDone());
                assertFalse(second.isDone());
                finishInstall.countDown();
                assertSame(exact.get(), registration.get(5, TimeUnit.SECONDS));
                assertSame(cleanupFailure, first.get(5, TimeUnit.SECONDS));
                assertSame(cleanupFailure, second.get(5, TimeUnit.SECONDS));
                assertTrue(interrupted.get());
                assertTrue(timerExecutor.isTerminated());
                assertTrue(timerExecutor.getQueue().isEmpty());
                assertSame(cleanupFailure, org.flexlb.util.Failures.run(null, timer::close));
                verify(access).snapshotActive();
            } finally {
                finishInstall.countDown();
                org.flexlb.util.Failures.run(null, timer::close);
            }
        }
    }

    @Test
    void detachedCleanupAttemptsEveryDeadlineOnceDespiteFailures() {
        try (var fixture = new Fixture()) {
            var first = new IllegalStateException("inactivity");
            var second = new IllegalStateException("request");
            var third = new IllegalStateException("decision");
            when(fixture.inactivityTask.cancel(false)).thenThrow(first);
            when(fixture.requestTask.cancel(false)).thenThrow(second);
            when(fixture.decisionTask.cancel(false)).thenThrow(third);

            assertSame(first, assertThrows(IllegalStateException.class,
                    () -> fixture.deadlines.release()));
            assertEquals(List.of(second, third), List.of(first.getSuppressed()));
            fixture.deadlines.release();
            fixture.assertCanceled();
            var order = inOrder(fixture.inactivityTask, fixture.requestTask, fixture.decisionTask);
            order.verify(fixture.inactivityTask).cancel(false);
            order.verify(fixture.requestTask).cancel(false);
            order.verify(fixture.decisionTask).cancel(false);
            order.verifyNoMoreInteractions();
        }
    }

    @Test
    void concurrentDetachedReleaseCancelsEachTaskOnceAndPreventsDeadlineConsumption() throws Exception {
        try (var fixture = new Fixture(); var executor = Executors.newFixedThreadPool(2)) {
            CountDownLatch cancelEntered = new CountDownLatch(1);
            CountDownLatch finishCancel = new CountDownLatch(1);
            when(fixture.inactivityTask.cancel(false)).thenAnswer(invocation -> {
                cancelEntered.countDown();
                assertTrue(finishCancel.await(5, TimeUnit.SECONDS));
                return true;
            });
            var first = executor.submit(() -> fixture.deadlines.release());
            try {
                assertTrue(cancelEntered.await(5, TimeUnit.SECONDS));
                executor.submit(() -> fixture.deadlines.release()).get(5, TimeUnit.SECONDS);
                fixture.assertCanceled();
            } finally {
                finishCancel.countDown();
            }
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
        var context = mock(RequestContext.class);
        var owner = mock(AbstractRequestScheduler.class);
        when(context.scheduler()).thenReturn(owner);
        when(requests.isCurrent(context)).thenReturn(true);
        when(requests.snapshotActive()).thenReturn(List.of());
        when(context.inactivityDeadlineAtMs()).thenReturn(OptionalLong.of(Long.MAX_VALUE));
        var timer = new ExpirationTimer(requests);
        var original = (java.util.concurrent.ScheduledThreadPoolExecutor) ReflectionTestUtils.getField(timer, "executor");
        original.shutdownNow();
        var executor = org.mockito.Mockito.spy(new java.util.concurrent.ScheduledThreadPoolExecutor(1));
        var task = mock(ScheduledFuture.class);
        var callback = new java.util.concurrent.atomic.AtomicReference<Runnable>();
        org.mockito.Mockito.doAnswer(call -> {
            callback.set(call.getArgument(0));
            return task;
        }).when(executor).schedule(any(Runnable.class), org.mockito.ArgumentMatchers.anyLong(), org.mockito.ArgumentMatchers.eq(TimeUnit.MILLISECONDS));
        ReflectionTestUtils.setField(timer, "executor", executor);
        org.mockito.stubbing.Answer<Boolean> install = call -> {
            ExpirationTimer.DeadlineRegistration exact = call.getArgument(0);
            assertTrue(Thread.holdsLock(context), "installation is atomic with request state");
            if (earlyFire) { callback.get().run(); }
            if (cancel) { assertTrue(exact.cancel()); }
            return installed;
        };
        when(context.installRequestDeadline(any())).thenAnswer(install);
        when(context.installDecisionDeadline(any())).thenAnswer(install);
        when(context.installInactivityDeadline(any())).thenAnswer(install);
        try {
            ExpirationTimer.DeadlineRegistration exact = switch (kind) {
                case "REQUEST" -> timer.scheduleRequestDeadline(context, Long.MAX_VALUE);
                case "DECISION" -> timer.registerDecisionDeadline(context, Long.MAX_VALUE);
                case "INACTIVITY" -> timer.scheduleInactivityDeadline(context);
                default -> throw new AssertionError(kind);
            };
            assertEquals(installed, exact != null);
            callback.get().run();
            callback.get().run();
            int expected = installed && !cancel ? 1 : 0;
            verify(owner, org.mockito.Mockito.times(kind.equals("REQUEST") ? expected : 0)).onSchedulingDeadline(org.mockito.ArgumentMatchers.eq(context), any());
            verify(context, org.mockito.Mockito.times(kind.equals("DECISION") ? expected : 0)).onDecisionVisibilityDeadline(any());
            verify(owner, org.mockito.Mockito.times(kind.equals("INACTIVITY") ? expected : 0)).enqueueInactivityDeadline(
                    org.mockito.ArgumentMatchers.eq(context), any(), org.mockito.ArgumentMatchers.anyLong(), any());
            if (exact != null) { assertFalse(exact.consume()); }
            verify(task, org.mockito.Mockito.times(!installed || cancel ? 1 : 0)).cancel(false);
        } finally {
            timer.close();
        }
    }

    static java.util.stream.Stream<org.junit.jupiter.params.provider.Arguments> deadlineRaces() {
        return java.util.stream.Stream.of("REQUEST", "DECISION", "INACTIVITY").flatMap(kind ->
                java.util.stream.IntStream.range(0, 8).mapToObj(bits -> org.junit.jupiter.params.provider.Arguments.of(
                        kind, (bits & 1) != 0, (bits & 2) != 0, (bits & 4) != 0)));
    }

    private static final class Fixture implements AutoCloseable {
        private final ExpirationTimer timer;
        private final ExpirationTimer.DetachedDeadlines deadlines;
        private final ScheduledFuture<?> inactivityTask;
        private final ScheduledFuture<?> requestTask;
        private final ScheduledFuture<?> decisionTask;

        private Fixture() {
            var access = mock(RequestRepository.class);
            var context = mock(RequestContext.class);
            when(access.isCurrent(context)).thenReturn(true);
            when(access.snapshotActive()).thenReturn(List.of());
            when(context.installRequestDeadline(any())).thenReturn(true);
            when(context.installDecisionDeadline(any())).thenReturn(true);
            when(context.installInactivityDeadline(any())).thenReturn(true);
            when(context.inactivityDeadlineAtMs()).thenReturn(OptionalLong.of(Long.MAX_VALUE));
            timer = new ExpirationTimer(access);
            var request = timer.scheduleRequestDeadline(context, Long.MAX_VALUE);
            var decision = timer.registerDecisionDeadline(context, Long.MAX_VALUE);
            var inactivity = timer.scheduleInactivityDeadline(context);
            deadlines = new ExpirationTimer.DetachedDeadlines(request, decision, inactivity);
            requestTask = replaceTask(request);
            decisionTask = replaceTask(decision);
            inactivityTask = replaceTask(inactivity);
        }

        private static ScheduledFuture<?> replaceTask(ExpirationTimer.DeadlineRegistration deadline) {
            // Keep the real registration state machine; inject failure only at executor cancellation.
            ((ScheduledFuture<?>) ReflectionTestUtils.getField(deadline, "scheduled")).cancel(false);
            ScheduledFuture<?> task = mock(ScheduledFuture.class);
            ReflectionTestUtils.setField(deadline, "scheduled", task);
            return task;
        }

        private void assertCanceled() {
            assertFalse(deadlines.inactivityDeadline().consume());
            assertFalse(deadlines.requestDeadline().consume());
            assertFalse(deadlines.detachedDecisionDeadline().consume());
        }

        @Override
        public void close() { timer.close(); }
    }
}
