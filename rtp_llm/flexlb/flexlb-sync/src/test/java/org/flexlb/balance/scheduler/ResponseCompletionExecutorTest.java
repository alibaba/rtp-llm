package org.flexlb.balance.scheduler;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.Set;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

@Timeout(15)
class ResponseCompletionExecutorTest {
    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void rejectedWorkersKeepAnAsynchronousCompletionOwner(boolean rejectRecovery) throws Exception {
        try (var f = new Fixture()) {
            var caller = Thread.currentThread();
            var completed = new CompletableFuture<Thread>();
            var executions = new AtomicInteger();
            var rejected = mock(ThreadPoolExecutor.class);
            doThrow(new RejectedExecutionException("primary rejected")).when(rejected).execute(any());
            ReflectionTestUtils.setField(f.executor, "completionWorkers", rejected);
            if (rejectRecovery) {
                var recovery = mock(ExecutorService.class);
                doThrow(new RejectedExecutionException("recovery rejected")).when(recovery).execute(any());
                ReflectionTestUtils.setField(f.executor, "rejectedTaskWorker", recovery);
            }
            var registration = f.executor.tryRegister();
            assertNotNull(registration);
            registration.submit(() -> {
                executions.incrementAndGet();
                completed.complete(Thread.currentThread());
                return true;
            });
            assertNotSame(caller, completed.get(5, TimeUnit.SECONDS));
            RequestProtocolTestSupport.awaitCondition(() -> registrations(f.executor).isEmpty());
            assertTrue(registration.hasCompletionOwner());
            assertEquals(1, executions.get());
        }
    }

    @Test
    void submissionExceptionAfterExecutionCannotReleaseOrExecuteTheCompletionTwice() {
        try (var f = new Fixture()) {
            var executions = new AtomicInteger();
            var failed = mock(ThreadPoolExecutor.class);
            var failure = new RejectedExecutionException("executor threw after actual completion");
            doAnswer(call -> { call.<Runnable>getArgument(0).run(); throw failure; })
                    .when(failed).execute(any());
            ReflectionTestUtils.setField(f.executor, "completionWorkers", failed);
            var registration = f.executor.tryRegister();
            assertNotNull(registration);
            assertSame(failure, assertThrows(RejectedExecutionException.class,
                    () -> registration.submit(() -> { executions.incrementAndGet(); return true; })));
            assertTrue(registration.hasCompletionOwner());
            assertEquals(1, executions.get());
            assertTrue(registrations(f.executor).isEmpty());
            assertFalse(registration.completeNow(() -> { executions.incrementAndGet(); return true; }));
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
            doAnswer(call -> { queued.set(call.getArgument(0)); throw first; }).when(primary).execute(any());
            var recovery = mock(ExecutorService.class);
            doThrow(new RejectedExecutionException("recovery rejected")).when(recovery).execute(any());
            var factory = mock(ThreadFactory.class);
            doThrow(new IllegalStateException("emergency unavailable")).when(factory).newThread(any());
            ReflectionTestUtils.setField(f.executor, "completionWorkers", primary);
            ReflectionTestUtils.setField(f.executor, "rejectedTaskWorker", recovery);
            ReflectionTestUtils.setField(f.executor, "emergencyWorkers", factory);
            var abandoned = f.executor.tryRegister();
            assertNotNull(abandoned);
            assertSame(first, assertThrows(RejectedExecutionException.class,
                    () -> abandoned.submit(() -> { executions.incrementAndGet(); return true; })));
            assertFalse(abandoned.hasCompletionOwner());
            assertTrue(registrations(f.executor).isEmpty());
            assertNotNull(queued.get());
            queued.get().run();
            assertEquals(0, executions.get());
            assertTrue(registrations(f.executor).isEmpty());
            f.restore();
            var completed = new CompletableFuture<Boolean>();
            var actual = f.executor.tryRegister();
            assertNotNull(actual);
            actual.submit(() -> { executions.incrementAndGet(); return completed.complete(true); });
            assertTrue(completed.get(5, TimeUnit.SECONDS));
            RequestProtocolTestSupport.awaitCondition(() -> registrations(f.executor).isEmpty());
            assertEquals(1, executions.get());
        }
    }

    @Test
    void closeDrainsARunningEmergencyCompletionAndRejectsSelfWaiting() throws Exception {
        try (var f = new Fixture(); var closers = Executors.newSingleThreadExecutor()) {
            var started = new CountDownLatch(1);
            var finish = new CountDownLatch(1);
            var primary = mock(ThreadPoolExecutor.class);
            var recovery = mock(ExecutorService.class);
            doThrow(new RejectedExecutionException("primary rejected")).when(primary).execute(any());
            doThrow(new RejectedExecutionException("recovery rejected")).when(recovery).execute(any());
            ReflectionTestUtils.setField(f.executor, "completionWorkers", primary);
            ReflectionTestUtils.setField(f.executor, "rejectedTaskWorker", recovery);
            var actual = f.executor.tryRegister();
            assertNotNull(actual);
            actual.submit(() -> {
                assertThrows(IllegalStateException.class, f.executor::close);
                var stillAccepting = f.executor.tryRegister();
                assertNotNull(stillAccepting, "self-wait rejection must precede shutdown mutation");
                stillAccepting.close();
                started.countDown();
                try { assertTrue(finish.await(5, TimeUnit.SECONDS)); }
                catch (InterruptedException interrupted) { throw new AssertionError(interrupted); }
                return true;
            });
            try {
                assertTrue(started.await(5, TimeUnit.SECONDS));
                actual.close();
                assertTrue(registrations(f.executor).contains(actual), "a running callback still owes its registration");
                f.restore();
                var closing = closers.submit(f.executor::close);
                RequestProtocolTestSupport.awaitCondition(() -> ReflectionTestUtils.getField(f.executor, "closeResult") != null);
                assertFalse(closing.isDone());
                finish.countDown();
                closing.get(5, TimeUnit.SECONDS);
                assertTrue(registrations(f.executor).isEmpty());
                assertTrue(f.primary.isTerminated());
                assertTrue(f.recovery.isTerminated());
            } finally { finish.countDown(); }
        }
    }

    @SuppressWarnings("unchecked")
    private static Set<ResponseCompletionExecutor.CompletionRegistration> registrations(ResponseCompletionExecutor executor) {
        var registrations = (Set<ResponseCompletionExecutor.CompletionRegistration>) ReflectionTestUtils.getField(executor, "registrations");
        synchronized (registrations) { return Set.copyOf(registrations); }
    }

    private static final class Fixture implements AutoCloseable {
        private final ResponseCompletionExecutor executor = new ResponseCompletionExecutor(1);
        private final ThreadPoolExecutor primary = (ThreadPoolExecutor) ReflectionTestUtils.getField(executor, "completionWorkers");
        private final ExecutorService recovery = (ExecutorService) ReflectionTestUtils.getField(executor, "rejectedTaskWorker");
        private final ThreadFactory emergency = (ThreadFactory) ReflectionTestUtils.getField(executor, "emergencyWorkers");

        private void restore() {
            ReflectionTestUtils.setField(executor, "completionWorkers", primary);
            ReflectionTestUtils.setField(executor, "rejectedTaskWorker", recovery);
            ReflectionTestUtils.setField(executor, "emergencyWorkers", emergency);
        }

        @Override public void close() { restore(); executor.close(); }
    }
}
