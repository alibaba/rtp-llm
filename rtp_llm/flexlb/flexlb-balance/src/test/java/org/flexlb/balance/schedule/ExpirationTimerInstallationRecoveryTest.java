package org.flexlb.balance.schedule;

import org.flexlb.balance.schedule.ResponseCompletionExecutor.RequestFuture;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.ArrayList;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.ArgumentMatchers.same;
import static org.mockito.Mockito.any;
import static org.mockito.Mockito.anyLong;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.same;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Checks failed timer installation remains retryable without allowing older revisions or already consumed callbacks to replace current deadlines.
 */
class ExpirationTimerInstallationRecoveryTest {
    @ParameterizedTest
    @EnumSource(ExpirationTimer.Kind.class)
    void rejectedInstallationRetriesOnlyTheFailedKindWithItsExactToken(ExpirationTimer.Kind kind) {
        try (var f = new Fixture()) {
            var before = f.context.snapshot();
            var plan = before.deadlines();
            var count = new AtomicInteger();
            var rejected = new ArrayList<Runnable>();
            var failure = new RejectedExecutionException("first schedule rejected");
            doAnswer(
                            call -> {
                                if (count.getAndIncrement() == kind.ordinal()) {
                                    rejected.add(call.getArgument(0));
                                    throw failure;
                                }
                                return call.callRealMethod();
                            })
                    .when(f.executor)
                    .schedule(any(Runnable.class), anyLong(), eq(TimeUnit.MILLISECONDS));
            assertSame(
                    failure,
                    assertThrows(
                            RejectedExecutionException.class,
                            () -> f.timer.update(f.context, plan)));
            assertEquals(
                    before,
                    f.context.snapshot(),
                    "a failed installation cannot alter domain facts");
            assertEquals(0, ReflectionTestUtils.getField(f.timer, "registrations"));
            var handles =
                    new java.util.EnumMap<ExpirationTimer.Kind, Object>(ExpirationTimer.Kind.class);
            for (var each : ExpirationTimer.Kind.values()) {
                handles.put(each, f.handle(each));
            }
            f.timer.update(f.context, plan);
            assertEquals(4, count.get(), "same revision must install only the failed kind");
            for (var each : ExpirationTimer.Kind.values()) {
                if (each != kind) {
                    assertSame(handles.get(each), f.handle(each));
                }
                assertSame(plan.token(each), ReflectionTestUtils.getField(f.handle(each), "token"));
                assertEquals(before.deadlines().token(each).atMs(), plan.token(each).atMs());
            }
            assertNotSame(handles.get(kind), f.handle(kind));
            assertEquals(before, f.context.snapshot());
            rejected.getFirst().run();
            verify(f.owner, never()).onDeadline(any(), anyLong());
            f.clock.set(plan.token(kind).atMs());
            ReflectionTestUtils.invokeMethod(f.handle(kind), "fire");
            ReflectionTestUtils.invokeMethod(f.handle(kind), "fire");
            verify(f.owner).onDeadline(same(plan.token(kind)), eq(plan.token(kind).atMs()));
            f.timer.update(f.context, plan);
            assertEquals(4, count.get(), "a claimed callback must never be installed twice");
            assertEquals(0, ReflectionTestUtils.getField(f.timer, "registrations"));
        }
    }

    @org.junit.jupiter.api.Test
    void missingTimerHandleLeavesTheExactDeadlineRetryable() {
        try (var f = new Fixture()) {
            var token = f.context.snapshot().deadlines().request();
            var plan = new ExpirationTimer.Plan(1L, token, null, null);
            org.mockito.Mockito.doReturn(null)
                    .doCallRealMethod()
                    .when(f.executor)
                    .schedule(any(Runnable.class), anyLong(), eq(TimeUnit.MILLISECONDS));
            assertThrows(NullPointerException.class, () -> f.timer.update(f.context, plan));
            verify(f.owner, never()).onDeadline(any(), anyLong());
            f.timer.update(f.context, plan);
            f.clock.set(token.atMs());
            ReflectionTestUtils.invokeMethod(f.handle(ExpirationTimer.Kind.REQUEST), "fire");
            verify(f.owner).onDeadline(same(token), eq(token.atMs()));
        }
    }

    @ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(booleans = {false, true})
    @Timeout(15)
    void sameRevisionCannotStealAnInstallationWhileScheduleIsStillRunning(
            boolean firstAttemptRejected) throws Exception {
        try (var f = new Fixture();
                var caller = Executors.newSingleThreadExecutor()) {
            var token = f.context.snapshot().deadlines().request();
            var plan = new ExpirationTimer.Plan(1L, token, null, null);
            var entered = new CountDownLatch(1);
            var release = new CountDownLatch(1);
            var attempts = new AtomicInteger();
            doAnswer(
                            call -> {
                                if (attempts.getAndIncrement() == 0 && firstAttemptRejected) {
                                    throw new RejectedExecutionException("first attempt rejected");
                                }
                                entered.countDown();
                                assertTrue(release.await(5, TimeUnit.SECONDS));
                                return call.callRealMethod();
                            })
                    .when(f.executor)
                    .schedule(any(Runnable.class), anyLong(), eq(TimeUnit.MILLISECONDS));
            if (firstAttemptRejected) {
                assertThrows(
                        RejectedExecutionException.class, () -> f.timer.update(f.context, plan));
            }
            var installing = caller.submit(() -> f.timer.update(f.context, plan));
            try {
                assertTrue(entered.await(5, TimeUnit.SECONDS));
                var first = f.handle(ExpirationTimer.Kind.REQUEST);
                f.timer.update(f.context, plan);
                assertSame(first, f.handle(ExpirationTimer.Kind.REQUEST));
                assertEquals(1, ReflectionTestUtils.getField(f.timer, "registrations"));
                verify(f.executor, times(firstAttemptRejected ? 2 : 1))
                        .schedule(any(Runnable.class), anyLong(), eq(TimeUnit.MILLISECONDS));
            } finally {
                release.countDown();
            }
            installing.get(5, TimeUnit.SECONDS);
            assertEquals(0, ReflectionTestUtils.getField(f.timer, "registrations"));
        }
    }

    @Test
    void newerDomainRevisionCannotBeOverwrittenByRetryingTheFailedPlan() {
        try (var f = new Fixture()) {
            var before = f.context.snapshot();
            var old = before.deadlines();
            var attempts = new AtomicInteger();
            doAnswer(
                            call -> {
                                if (attempts.getAndIncrement() == 0) {
                                    throw new RejectedExecutionException("first request rejected");
                                }
                                return call.callRealMethod();
                            })
                    .when(f.executor)
                    .schedule(any(Runnable.class), anyLong(), eq(TimeUnit.MILLISECONDS));
            assertThrows(RejectedExecutionException.class, () -> f.timer.update(f.context, old));
            org.flexlb.balance.schedule.RequestChangeAssertions.record(
                    f.context,
                    observedRequest ->
                            observedRequest.recordDeadline(
                                    old.request(), old.request().atMs() - 1L, null));
            var newer = f.context.snapshot().deadlines();
            assertTrue(newer.revision() > old.revision());
            assertNotSame(old.request(), newer.request());
            assertEquals(old.request().atMs(), newer.request().atMs());
            f.timer.update(f.context, newer);
            var installed = f.handle(ExpirationTimer.Kind.REQUEST);
            f.timer.update(f.context, old);
            assertSame(installed, f.handle(ExpirationTimer.Kind.REQUEST));
            assertSame(newer.request(), ReflectionTestUtils.getField(installed, "token"));
            assertEquals(4, attempts.get());
            assertEquals(0, ReflectionTestUtils.getField(f.timer, "registrations"));
            assertNull(f.context.snapshot().requestEndResult());
            assertNull(f.context.snapshot().cancellationReason());
        }
    }

    @Test
    void failedCallbackAfterAnEarlyFireCannotBeReinstalled() {
        try (var f = new Fixture()) {
            var token = f.context.snapshot().deadlines().request();
            var plan = new ExpirationTimer.Plan(1L, token, null, null);
            var failure = new IllegalStateException("already claimed callback failed");
            doThrow(failure).when(f.owner).onDeadline(same(token), anyLong());
            doAnswer(
                            call -> {
                                ((Runnable) call.getArgument(0)).run();
                                return call.callRealMethod();
                            })
                    .when(f.executor)
                    .schedule(any(Runnable.class), anyLong(), eq(TimeUnit.MILLISECONDS));
            assertSame(
                    failure,
                    assertThrows(
                            IllegalStateException.class, () -> f.timer.update(f.context, plan)));
            f.timer.update(f.context, plan);
            verify(f.executor).schedule(any(Runnable.class), anyLong(), eq(TimeUnit.MILLISECONDS));
            verify(f.owner).onDeadline(same(token), anyLong());
            assertEquals(0, ReflectionTestUtils.getField(f.timer, "registrations"));
        }
    }

    /**
     * Combines a registered request, controlled clock and replaceable timer scheduling to inject installation failures and inspect exact deadline tokens.
     */
    private static final class Fixture implements AutoCloseable {
        final RequestContext context;
        final AbstractRequestScheduler owner = mock(AbstractRequestScheduler.class);
        final AtomicLong clock = new AtomicLong(1002L);
        final ExpirationTimer timer;
        final ScheduledThreadPoolExecutor executor = spy(new ScheduledThreadPoolExecutor(1));

        Fixture() {
            context = new RequestContext(SchedulingTestConfig.newConfig());
            var request = mock(org.flexlb.worker.endpoint.Request.class);
            when(request.getRequestId()).thenReturn(String.valueOf(7502L));
            context.setRequest(request);
            context.setSchedulingMetadata(
                    org.flexlb.balance.schedule.SchedulingMetadata.explicit(50, 100_000L));
            context.register(
                    owner,
                    new RequestFuture((kind, response, failure) -> false),
                    1000L,
                    10_000L,
                    org.flexlb.worker.endpoint.StrategyErrorType.SCHEDULING_DEADLINE_EXCEEDED);
            ReflectionTestUtils.setField(context, "prefillCompletedAtMs", 1001L);
            ReflectionTestUtils.setField(context, "visibilityAtMs", 3000L);
            ReflectionTestUtils.invokeMethod(context, "commitSnapshot", 1002L);
            var repository = mock(RequestRepository.class);
            when(repository.isCurrent(context)).thenReturn(true);
            timer =
                    new ExpirationTimer(
                            repository,
                            (task, delay) -> executor.schedule(task, delay, TimeUnit.MILLISECONDS),
                            clock::get);
        }

        @Override
        public void close() {
            timer.close();
            executor.shutdownNow();
        }

        Object handle(ExpirationTimer.Kind kind) {
            var slots = (Map<?, ?>) ReflectionTestUtils.getField(timer, "slots");
            var handles = (Object[]) ReflectionTestUtils.getField(slots.get(context), "handles");
            return handles[kind.ordinal()];
        }
    }
}
