package org.flexlb.balance.schedule;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.springframework.test.util.ReflectionTestUtils;

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

/**
 * Checks response registration executes at most once across rejected submission, late tasks and shutdown.
 * Closing must wait for accepted callbacks without allowing a callback to wait on itself.
 */
@Timeout(15)
class ResponseCompletionExecutorTest {
    @Test
    void preparedResponseCarriesItsValueAndArbitratesOnlyWhenExecuted() {
        try (var f = new Fixture()) {
            var queued = new AtomicReference<Runnable>();
            var workers = mock(ThreadPoolExecutor.class);
            doAnswer(
                            call -> {
                                queued.set(call.getArgument(0));
                                return null;
                            })
                    .when(workers)
                    .execute(any());
            ReflectionTestUtils.setField(f.executor, "completionWorkers", workers);
            var selected = new AtomicInteger();
            var future =
                    new ResponseCompletionExecutor.RequestFuture((kind, value, failure) -> false);
            var registration =
                    new ResponseCompletionExecutor.Registration(
                            f.executor,
                            new Object(),
                            future,
                            () -> {
                                selected.incrementAndGet();
                                return true;
                            });
            var value = mock(org.flexlb.worker.endpoint.Response.class);
            registration.prepare(
                    ResponseCompletionExecutor.RequestFuture.Completion.RESPONSE, value, null);
            assertThrows(
                    IllegalStateException.class,
                    () ->
                            registration.prepare(
                                    ResponseCompletionExecutor.RequestFuture.Completion
                                            .CANCELLATION,
                                    null,
                                    null));
            registration.publish(false);
            assertEquals(0, selected.get());
            assertFalse(future.isDone());
            assertEquals(1, registrations(f.executor));
            queued.get().run();
            assertSame(value, future.join());
            assertEquals(1, selected.get());
            assertEquals(0, registrations(f.executor));
        }
    }

    @Test
    void synchronousPreparedResponseReturnsLostArbitrationWithoutLeaking() {
        try (var f = new Fixture()) {
            var future =
                    new ResponseCompletionExecutor.RequestFuture((kind, value, failure) -> false);
            var registration =
                    new ResponseCompletionExecutor.Registration(
                            f.executor, new Object(), future, () -> false);
            registration.prepare(
                    ResponseCompletionExecutor.RequestFuture.Completion.RESPONSE,
                    mock(org.flexlb.worker.endpoint.Response.class),
                    null);
            assertFalse(registration.publish(true));
            assertFalse(future.isDone());
            assertEquals(0, registrations(f.executor));
        }
    }

    @Test
    void rejectedPreparedResponseNeedsNoCallerCleanup() {
        try (var f = new Fixture()) {
            var workers = mock(ThreadPoolExecutor.class);
            var failure = new RejectedExecutionException("response rejected");
            doThrow(failure).when(workers).execute(any());
            ReflectionTestUtils.setField(f.executor, "completionWorkers", workers);
            var future =
                    new ResponseCompletionExecutor.RequestFuture((kind, value, error) -> false);
            var registration =
                    new ResponseCompletionExecutor.Registration(
                            f.executor, new Object(), future, () -> true);
            registration.prepare(
                    ResponseCompletionExecutor.RequestFuture.Completion.RESPONSE,
                    mock(org.flexlb.worker.endpoint.Response.class),
                    null);
            assertSame(
                    failure,
                    assertThrows(
                            RejectedExecutionException.class, () -> registration.publish(false)));
            assertSame(
                    failure,
                    assertThrows(java.util.concurrent.CompletionException.class, future::join)
                            .getCause());
            assertEquals(0, registrations(f.executor));
        }
    }

    @Test
    void rejectedFailureRecoveryCompletesFutureAndReleasesTheReservation() {
        try (var f = new Fixture()) {
            var workers = mock(ThreadPoolExecutor.class);
            var rejected = new RejectedExecutionException("response executor rejected");
            doThrow(rejected).when(workers).execute(any());
            ReflectionTestUtils.setField(f.executor, "completionWorkers", workers);
            var future =
                    new ResponseCompletionExecutor.RequestFuture((kind, value, failure) -> false);
            var registration =
                    new ResponseCompletionExecutor.Registration(
                            f.executor, new Object(), future, () -> true);
            var original = new IllegalStateException("response preparation failed");
            registration.failIfUnsubmitted(original);
            registration.close();
            registration.failIfUnsubmitted(original);
            assertSame(
                    rejected,
                    assertThrows(java.util.concurrent.CompletionException.class, future::join)
                            .getCause());
            assertEquals(0, registrations(f.executor));
            assertEquals(1, original.getSuppressed().length);
            assertSame(rejected, original.getSuppressed()[0]);
        }
    }

    @Test
    void unsubmittedFailureUsesBoundArbitrationOutsideTheRequestLock() throws Exception {
        try (var f = new Fixture()) {
            var lock = new Object();
            var future =
                    new ResponseCompletionExecutor.RequestFuture((kind, value, failure) -> false);
            var selections = new AtomicInteger();
            var registration =
                    new ResponseCompletionExecutor.Registration(
                            f.executor,
                            lock,
                            future,
                            () -> {
                                assertFalse(Thread.holdsLock(lock));
                                selections.incrementAndGet();
                                return true;
                            });
            var failure = new IllegalStateException("response preparation failed");
            registration.failIfUnsubmitted(failure);
            registration.close();
            assertSame(
                    failure,
                    assertThrows(java.util.concurrent.CompletionException.class, future::join)
                            .getCause());
            RequestProtocolTestSupport.awaitCondition(() -> registrations(f.executor) == 0);
            assertEquals(1, selections.get());
        }
    }

    @Test
    void failedPreparationCannotReplaceAnAlreadySubmittedResponse() {
        try (var f = new Fixture()) {
            var queued = new AtomicReference<Runnable>();
            var workers = mock(ThreadPoolExecutor.class);
            doAnswer(
                            call -> {
                                queued.set(call.getArgument(0));
                                return null;
                            })
                    .when(workers)
                    .execute(any());
            ReflectionTestUtils.setField(f.executor, "completionWorkers", workers);
            var future =
                    new ResponseCompletionExecutor.RequestFuture((kind, value, failure) -> false);
            var registration =
                    new ResponseCompletionExecutor.Registration(
                            f.executor, new Object(), future, () -> true);
            var response = mock(org.flexlb.worker.endpoint.Response.class);
            registration.submit(
                    () ->
                            registration.complete(
                                    ResponseCompletionExecutor.RequestFuture.Completion.RESPONSE,
                                    response,
                                    null));
            registration.failIfUnsubmitted(new IllegalStateException("late failure"));
            assertFalse(future.isDone());
            assertEquals(1, registrations(f.executor));
            queued.get().run();
            assertSame(response, future.join());
            assertEquals(0, registrations(f.executor));
        }
    }

    @Test
    void losingFailureCandidateReleasesRegistrationWithoutWritingTheFuture() throws Exception {
        try (var f = new Fixture()) {
            var future =
                    new ResponseCompletionExecutor.RequestFuture((kind, value, failure) -> false);
            var registration =
                    new ResponseCompletionExecutor.Registration(
                            f.executor, new Object(), future, () -> false);
            registration.failIfUnsubmitted(new IllegalStateException("obsolete response"));
            RequestProtocolTestSupport.awaitCondition(() -> registrations(f.executor) == 0);
            assertFalse(future.isDone());
        }
    }

    @Test
    void closingAfterHandoffDoesNotReleaseOrRunTheQueuedResponse() {
        try (var f = new Fixture()) {
            var queued = new AtomicReference<Runnable>();
            var executions = new AtomicInteger();
            var workers = mock(ThreadPoolExecutor.class);
            doAnswer(
                            call -> {
                                queued.set(call.getArgument(0));
                                return null;
                            })
                    .when(workers)
                    .execute(any());
            ReflectionTestUtils.setField(f.executor, "completionWorkers", workers);
            var registration = SchedulerTestSupport.responsePublication(f.executor);
            registration.submit(
                    () -> {
                        executions.incrementAndGet();
                        return true;
                    });
            registration.close();
            registration.close();
            assertEquals(1, registrations(f.executor));
            assertEquals(0, executions.get());
            queued.get().run();
            queued.get().run();
            assertEquals(1, executions.get());
            assertEquals(0, registrations(f.executor));
        }
    }

    @Test
    void failingResponseActionReleasesItsRegistrationOnce() {
        try (var f = new Fixture()) {
            var registration = SchedulerTestSupport.responsePublication(f.executor);
            var failure = new IllegalStateException("response construction failed");
            assertSame(
                    failure,
                    assertThrows(
                            IllegalStateException.class,
                            () ->
                                    registration.completeNow(
                                            () -> {
                                                throw failure;
                                            })));
            registration.close();
            assertEquals(0, registrations(f.executor));
            assertFalse(
                    registration.completeNow(
                            () -> {
                                throw new AssertionError("replayed");
                            }));
        }
    }

    @Test
    void closedExecutorCannotAcceptAnotherRegistration() {
        var executor = new ResponseCompletionExecutor(1);
        executor.close();
        assertThrows(
                IllegalStateException.class,
                () -> SchedulerTestSupport.responsePublication(executor));
        assertFalse(executor.register());
        assertEquals(0, registrations(executor));
    }

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
            assertTrue(registrations(f.executor) == 0);
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
            assertFalse(registration.isUnsubmitted());
            assertEquals(0, executions.get());
            assertTrue(registrations(f.executor) == 0);
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
            assertFalse(registration.isUnsubmitted());
            assertEquals(1, executions.get());
            assertTrue(registrations(f.executor) == 0);
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
            assertFalse(abandoned.isUnsubmitted());
            assertTrue(registrations(f.executor) == 0);
            assertNotNull(queued.get());
            queued.get().run();
            assertEquals(0, executions.get());
            assertTrue(registrations(f.executor) == 0);
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
            RequestProtocolTestSupport.awaitCondition(() -> registrations(f.executor) == 0);
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
                        registrations(f.executor) == 1,
                        "a running callback still owes its registration");
                var closing = closers.submit(f.executor::close);
                RequestProtocolTestSupport.awaitCondition(
                        () -> ReflectionTestUtils.getField(f.executor, "closeResult") != null);
                assertFalse(closing.isDone());
                finish.countDown();
                closing.get(5, TimeUnit.SECONDS);
                assertTrue(registrations(f.executor) == 0);
                assertTrue(f.primary.isTerminated());
            } finally {
                finish.countDown();
            }
        }
    }

    private static int registrations(ResponseCompletionExecutor executor) {
        var executorOwner = executor;
        var lock = ReflectionTestUtils.getField(executorOwner, "completionLock");
        synchronized (lock) {
            return (int) ReflectionTestUtils.getField(executorOwner, "pendingCompletions");
        }
    }

    /**
     * Retains the original response worker pool while tests replace submission behavior, then restores and closes it for cleanup.
     */
    private static final class Fixture implements AutoCloseable {
        private final ResponseCompletionExecutor executor = new ResponseCompletionExecutor(1);
        private final ThreadPoolExecutor primary =
                (ThreadPoolExecutor) ReflectionTestUtils.getField(executor, "completionWorkers");

        @Override
        public void close() {
            restore();
            executor.close();
        }

        private void restore() {
            ReflectionTestUtils.setField(executor, "completionWorkers", primary);
        }
    }
}
