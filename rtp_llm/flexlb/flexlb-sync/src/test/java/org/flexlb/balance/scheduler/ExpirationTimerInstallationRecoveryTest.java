package org.flexlb.balance.scheduler;

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

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

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
            doAnswer(call -> {
                if (count.getAndIncrement() == kind.ordinal()) {
                    rejected.add(call.getArgument(0));
                    throw failure;
                }
                return call.callRealMethod();
            }).when(f.executor).schedule(any(Runnable.class), anyLong(), eq(TimeUnit.MILLISECONDS));
            assertSame(failure, assertThrows(RejectedExecutionException.class, () -> f.timer.update(f.context, plan)));
            assertSame(before, f.context.snapshot(), "a failed installation cannot alter domain facts");
            assertEquals(0, ReflectionTestUtils.getField(f.timer, "registrations"));
            var handles = new java.util.EnumMap<ExpirationTimer.Kind, Object>(ExpirationTimer.Kind.class);
            for (var each : ExpirationTimer.Kind.values()) { handles.put(each, f.handle(each)); }
            f.timer.update(f.context, plan);
            assertEquals(4, count.get(), "same revision must install only the failed kind");
            for (var each : ExpirationTimer.Kind.values()) {
                if (each != kind) { assertSame(handles.get(each), f.handle(each)); }
                assertSame(plan.token(each), ReflectionTestUtils.getField(f.handle(each), "token"));
                assertEquals(before.deadlines().token(each).atMs(), plan.token(each).atMs());
            }
            assertNotSame(handles.get(kind), f.handle(kind));
            assertSame(before, f.context.snapshot());
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

    @ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(booleans = {false, true})
    @Timeout(15)
    void sameRevisionCannotStealAnInstallationWhileScheduleIsStillRunning(boolean firstAttemptRejected) throws Exception {
        try (var f = new Fixture(); var caller = Executors.newSingleThreadExecutor()) {
            var token = f.context.snapshot().deadlines().request();
            var plan = new ExpirationTimer.Plan(1L, token, null, null);
            var entered = new CountDownLatch(1);
            var release = new CountDownLatch(1);
            var attempts = new AtomicInteger();
            doAnswer(call -> {
                if (attempts.getAndIncrement() == 0 && firstAttemptRejected) {
                    throw new RejectedExecutionException("first attempt rejected");
                }
                entered.countDown();
                assertTrue(release.await(5, TimeUnit.SECONDS));
                return call.callRealMethod();
            }).when(f.executor).schedule(any(Runnable.class), anyLong(), eq(TimeUnit.MILLISECONDS));
            if (firstAttemptRejected) {
                assertThrows(RejectedExecutionException.class, () -> f.timer.update(f.context, plan));
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
            } finally { release.countDown(); }
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
            doAnswer(call -> {
                if (attempts.getAndIncrement() == 0) { throw new RejectedExecutionException("first request rejected"); }
                return call.callRealMethod();
            }).when(f.executor).schedule(any(Runnable.class), anyLong(), eq(TimeUnit.MILLISECONDS));
            assertThrows(RejectedExecutionException.class, () -> f.timer.update(f.context, old));
            f.context.recordDeadline(old.request(), old.request().atMs() - 1L, null);
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
            assertNull(f.context.snapshot().outcome());
            assertNull(f.context.snapshot().cancellation());
        }
    }

    @Test
    void failedCallbackAfterAnEarlyFireCannotBeReinstalled() {
        try (var f = new Fixture()) {
            var token = f.context.snapshot().deadlines().request();
            var plan = new ExpirationTimer.Plan(1L, token, null, null);
            var failure = new IllegalStateException("already claimed callback failed");
            doThrow(failure).when(f.owner).onDeadline(same(token), anyLong());
            doAnswer(call -> {
                ((Runnable) call.getArgument(0)).run();
                return call.callRealMethod();
            }).when(f.executor).schedule(any(Runnable.class), anyLong(), eq(TimeUnit.MILLISECONDS));
            assertSame(failure, assertThrows(IllegalStateException.class, () -> f.timer.update(f.context, plan)));
            f.timer.update(f.context, plan);
            verify(f.executor).schedule(any(Runnable.class), anyLong(), eq(TimeUnit.MILLISECONDS));
            verify(f.owner).onDeadline(same(token), anyLong());
            assertEquals(0, ReflectionTestUtils.getField(f.timer, "registrations"));
        }
    }

    private static final class Fixture implements AutoCloseable {
        final RequestContext context;
        final AbstractRequestScheduler owner = mock(AbstractRequestScheduler.class);
        final AtomicLong clock = new AtomicLong(1002L);
        final ExpirationTimer timer;
        final ScheduledThreadPoolExecutor executor = spy(new ScheduledThreadPoolExecutor(1));

        Fixture() {
            var delivery = RequestDeliveryClaimStateTest.fixture(DeliveryClaimKind.BATCH_ENQUEUE, false);
            context = delivery.context();
            context.recordDeliveryResult(new ContextResult.Prediction(delivery.delivery(), 3000L, 1002L));
            var request = mock(org.flexlb.dao.loadbalance.Request.class);
            when(request.getRequestId()).thenReturn(7502L);
            context.data().setRequest(request);
            context.data().setSchedulingMetadata(org.flexlb.dao.SchedulingMetadata.explicit(50, 100_000L));
            context.data().register(owner, new RequestFuture((kind, response, failure) -> false));
            var repository = mock(RequestRepository.class);
            when(repository.isCurrent(context)).thenReturn(true);
            timer = new ExpirationTimer(repository, clock::get);
            ((ScheduledThreadPoolExecutor) ReflectionTestUtils.getField(timer, "executor")).shutdownNow();
            ReflectionTestUtils.setField(timer, "executor", executor);
        }

        Object handle(ExpirationTimer.Kind kind) {
            var slots = (Map<?, ?>) ReflectionTestUtils.getField(timer, "slots");
            var handles = (Map<?, ?>) ReflectionTestUtils.getField(slots.get(context), "handles");
            return handles.get(kind);
        }

        @Override public void close() { timer.close(); }
    }
}
