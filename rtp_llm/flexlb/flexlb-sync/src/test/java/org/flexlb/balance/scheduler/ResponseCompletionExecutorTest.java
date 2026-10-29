package org.flexlb.balance.scheduler;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.any;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;

@Timeout(15)
class ResponseCompletionExecutorTest {
    @Test
    void synchronousCompletionCannotStealAnAcceptedAsynchronousTask() {
        try (var f = new Fixture()) {
            var queued = new AtomicReference<Runnable>();
            var executions = new AtomicInteger();
            var primary = mock(ThreadPoolExecutor.class);
            doAnswer(
                            call -> {
                                queued.set(call.getArgument(0));
                                return null;
                            })
                    .when(primary)
                    .execute(any());
            ReflectionTestUtils.setField(f.executor, "completionWorkers", primary);
            var registration = SchedulerTestSupport.responsePublication(f.executor);
            registration.submit(
                    () -> {
                        executions.incrementAndGet();
                        return true;
                    });
            try {
                assertFalse(
                        registration.completeNow(
                                () -> {
                                    executions.addAndGet(100);
                                    return true;
                                }));
            } finally {
                queued.get().run();
            }
            assertEquals(1, executions.get());
            assertTrue(registrations(f.executor).isEmpty());
        }
    }

    @Test
    void rejectedSubmissionFailsThePublishedCompletion() {
        try (var f = new Fixture()) {
            var executions = new AtomicInteger();
            var rejected = mock(ThreadPoolExecutor.class);
            doThrow(new RejectedExecutionException("primary rejected"))
                    .when(rejected)
                    .execute(any());
            ReflectionTestUtils.setField(f.executor, "completionWorkers", rejected);
            var registration = SchedulerTestSupport.responsePublication(f.executor);
            assertNotNull(registration);
            assertThrows(
                    RejectedExecutionException.class,
                    () ->
                            registration.submit(
                                    () -> {
                                        executions.incrementAndGet();
                                        return true;
                                    }));
            assertFalse(registration.hasCompletionOwner());
            assertEquals(0, executions.get());
            assertTrue(registrations(f.executor).isEmpty());
        }
    }

    @Test
    void submissionExceptionAfterExecutionCannotReleaseOrExecuteTheCompletionTwice() {
        try (var f = new Fixture()) {
            var executions = new AtomicInteger();
            var failed = mock(ThreadPoolExecutor.class);
            var failure = new RejectedExecutionException("executor threw after actual completion");
            doAnswer(
                            call -> {
                                call.<Runnable>getArgument(0).run();
                                throw failure;
                            })
                    .when(failed)
                    .execute(any());
            ReflectionTestUtils.setField(f.executor, "completionWorkers", failed);
            var registration = SchedulerTestSupport.responsePublication(f.executor);
            assertNotNull(registration);
            assertSame(
                    failure,
                    assertThrows(
                            RejectedExecutionException.class,
                            () ->
                                    registration.submit(
                                            () -> {
                                                executions.incrementAndGet();
                                                return true;
                                            })));
            assertTrue(registration.hasCompletionOwner());
            assertEquals(1, executions.get());
            assertTrue(registrations(f.executor).isEmpty());
            assertFalse(
                    registration.completeNow(
                            () -> {
                                executions.incrementAndGet();
                                return true;
                            }));
            assertEquals(1, executions.get());
        }
    }

    @Test
    void rejectedRegistrationCannotBeRevivedByALateQueuedTask() throws Exception {
        try (var f = new Fixture()) {
            var queued = new AtomicReference<Runnable>();
            var executions = new AtomicInteger();
            var primary = mock(ThreadPoolExecutor.class);
            var first = new RejectedExecutionException("queued then rejected");
            doAnswer(
                            call -> {
                                queued.set(call.getArgument(0));
                                throw first;
                            })
                    .when(primary)
                    .execute(any());
            ReflectionTestUtils.setField(f.executor, "completionWorkers", primary);
            var abandoned = SchedulerTestSupport.responsePublication(f.executor);
            assertNotNull(abandoned);
            assertSame(
                    first,
                    assertThrows(
                            RejectedExecutionException.class,
                            () ->
                                    abandoned.submit(
                                            () -> {
                                                executions.incrementAndGet();
                                                return true;
                                            })));
            assertFalse(abandoned.hasCompletionOwner());
            assertTrue(registrations(f.executor).isEmpty());
            assertNotNull(queued.get());
            queued.get().run();
            assertEquals(0, executions.get());
            assertTrue(registrations(f.executor).isEmpty());
            f.restore();
            var completed = new CompletableFuture<Boolean>();
            var actual = SchedulerTestSupport.responsePublication(f.executor);
            assertNotNull(actual);
            actual.submit(
                    () -> {
                        executions.incrementAndGet();
                        return completed.complete(true);
                    });
            assertTrue(completed.get(5, TimeUnit.SECONDS));
            RequestProtocolTestSupport.awaitCondition(() -> registrations(f.executor).isEmpty());
            assertEquals(1, executions.get());
        }
    }

    @Test
    void closeDrainsARunningCompletionAndRejectsSelfWaiting() throws Exception {
        try (var f = new Fixture();
                var closers = Executors.newSingleThreadExecutor()) {
            var started = new CountDownLatch(1);
            var finish = new CountDownLatch(1);
            var actual = SchedulerTestSupport.responsePublication(f.executor);
            assertNotNull(actual);
            actual.submit(
                    () -> {
                        assertThrows(IllegalStateException.class, f.executor::close);
                        var stillAccepting = SchedulerTestSupport.responsePublication(f.executor);
                        assertNotNull(
                                stillAccepting,
                                "self-wait rejection must precede shutdown mutation");
                        stillAccepting.close();
                        started.countDown();
                        try {
                            assertTrue(finish.await(5, TimeUnit.SECONDS));
                        } catch (InterruptedException interrupted) {
                            throw new AssertionError(interrupted);
                        }
                        return true;
                    });
            try {
                assertTrue(started.await(5, TimeUnit.SECONDS));
                actual.close();
                assertTrue(
                        registrations(f.executor).contains(actual),
                        "a running callback still owes its registration");
                var closing = closers.submit(f.executor::close);
                RequestProtocolTestSupport.awaitCondition(
                        () -> ReflectionTestUtils.getField(f.executor, "closeResult") != null);
                assertFalse(closing.isDone());
                finish.countDown();
                closing.get(5, TimeUnit.SECONDS);
                assertTrue(registrations(f.executor).isEmpty());
                assertTrue(f.primary.isTerminated());
            } finally {
                finish.countDown();
            }
        }
    }

    @SuppressWarnings("unchecked")
    private static Set<ResponsePublication> registrations(ResponseCompletionExecutor executor) {
        var registrations =
                (Set<ResponsePublication>) ReflectionTestUtils.getField(executor, "registrations");
        synchronized (registrations) {
            return Set.copyOf(registrations);
        }
    }

    private static final class Fixture implements AutoCloseable {
        private final ResponseCompletionExecutor executor = new ResponseCompletionExecutor(1);
        private final ThreadPoolExecutor primary =
                (ThreadPoolExecutor) ReflectionTestUtils.getField(executor, "completionWorkers");

        private void restore() {
            ReflectionTestUtils.setField(executor, "completionWorkers", primary);
        }

        @Override
        public void close() {
            restore();
            executor.close();
        }
    }
}
