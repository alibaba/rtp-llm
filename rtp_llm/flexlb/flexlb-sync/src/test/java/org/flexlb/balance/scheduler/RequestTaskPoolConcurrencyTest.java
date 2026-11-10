package org.flexlb.balance.scheduler;

import org.junit.jupiter.api.RepeatedTest;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.ArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Uses the real task pool and its serial cleanup queue, including concurrent shutdown.
 */
@Timeout(20)
class RequestTaskPoolConcurrencyTest {
    @RepeatedTest(20)
    void concurrentNotificationsAreSerializedAndShutdownWaitsForTheirLastEffect() throws Exception {
        var pool = new RequestTaskPool();
        var context = new RequestContext(SchedulingTestConfig.batchConfig());
        var firstEntered = new CountDownLatch(1);
        var resume = new CountDownLatch(1);
        var published = new AtomicInteger();
        var applied = new AtomicInteger();
        var active = new AtomicInteger();
        var maximum = new AtomicInteger();
        var executions = new AtomicInteger();
        Runnable effect =
                () -> {
                    maximum.accumulateAndGet(active.incrementAndGet(), Math::max);
                    try {
                        int value = published.get();
                        if (executions.incrementAndGet() == 1) {
                            firstEntered.countDown();
                            try {
                                assertTrue(resume.await(5, TimeUnit.SECONDS));
                            } catch (InterruptedException e) {
                                throw new AssertionError(e);
                            }
                        }
                        applied.set(value);
                    } finally {
                        active.decrementAndGet();
                    }
                };
        var drivers = Executors.newFixedThreadPool(9, Thread.ofPlatform().daemon().factory());
        Future<?> closing = null;
        try {
            try (var work = pool.registerRequestWork(context)) {
                work.submitClose(effect, false, null);
            }
            assertTrue(firstEntered.await(5, TimeUnit.SECONDS));
            var start = new CyclicBarrier(8);
            var writers = new ArrayList<Future<?>>();
            for (int actor = 0; actor < 8; actor++) {
                writers.add(
                        drivers.submit(
                                () -> {
                                    start.await(5, TimeUnit.SECONDS);
                                    for (int i = 0; i < 50; i++) {
                                        published.incrementAndGet();
                                        try (var work = pool.registerRequestWork(context)) {
                                            work.submitClose(effect, false, null);
                                        }
                                    }
                                    return null;
                                }));
            }
            for (var writer : writers) writer.get(5, TimeUnit.SECONDS);
            var closeEntered = new CountDownLatch(1);
            closing =
                    drivers.submit(
                            () -> {
                                closeEntered.countDown();
                                pool.close();
                            });
            assertTrue(closeEntered.await(5, TimeUnit.SECONDS));
            // Rejected admission proves close has crossed its gate, without sleeping or peeking
            // fields.
            long limit = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
            while (true) {
                var probe = pool.tryRegisterRequestWork(context);
                if (probe == null) break;
                probe.close();
                assertTrue(System.nanoTime() < limit, "shutdown did not close admission");
                Thread.yield();
            }
            assertFalse(closing.isDone(), "accepted work still owns the blocked first effect");
            resume.countDown();
            closing.get(5, TimeUnit.SECONDS);
            pool.throwIfFailed();
            assertEquals(400, applied.get(), "coalescing must not lose the final published state");
            assertEquals(2, executions.get(), "one blocked effect and one coalesced continuation");
            assertEquals(1, maximum.get(), "the same request must never settle concurrently");
            assertNull(pool.tryRegisterRequestWork(context));
        } finally {
            resume.countDown();
            if (closing == null) closing = drivers.submit(pool::close);
            try {
                closing.get(5, TimeUnit.SECONDS);
            } finally {
                drivers.shutdownNow();
                assertTrue(drivers.awaitTermination(5, TimeUnit.SECONDS));
            }
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void shutdownDrainsAnOwnersContinuationButRejectsUnrelatedWork(boolean stopWork)
            throws Exception {
        var pool = new RequestTaskPool();
        var context = new RequestContext(SchedulingTestConfig.batchConfig());
        var other = new RequestContext(SchedulingTestConfig.batchConfig());
        var entered = new CountDownLatch(1);
        var continueOwner = new CountDownLatch(1);
        var childEntered = new CountDownLatch(1);
        var finishChild = new CountDownLatch(1);
        var childDone = new AtomicInteger();
        var driver = Executors.newSingleThreadExecutor(Thread.ofPlatform().daemon().factory());
        Future<?> closing = null;
        try {
            Runnable owner =
                    () -> {
                        entered.countDown();
                        await(continueOwner);
                        assertNull(
                                pool.tryRegisterRequestWork(other),
                                "ownership is exact, not pool-wide");
                        pool.execute(
                                context,
                                () -> {
                                    childEntered.countDown();
                                    await(finishChild);
                                    childDone.incrementAndGet();
                                });
                    };
            if (stopWork) assertTrue(pool.executeStopWork(context, owner));
            else pool.execute(context, owner);
            assertTrue(entered.await(5, TimeUnit.SECONDS));
            closing = driver.submit(pool::close);
            long limit = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
            while (true) {
                var probe = pool.tryRegisterRequestWork(context);
                if (probe == null) break;
                probe.close();
                assertTrue(System.nanoTime() < limit, "shutdown did not close admission");
                Thread.yield();
            }
            continueOwner.countDown();
            assertTrue(childEntered.await(5, TimeUnit.SECONDS));
            assertFalse(closing.isDone(), "shutdown must wait for the accepted continuation");
            finishChild.countDown();
            closing.get(5, TimeUnit.SECONDS);
            pool.throwIfFailed();
            assertEquals(1, childDone.get());
        } finally {
            continueOwner.countDown();
            finishChild.countDown();
            if (closing == null) closing = driver.submit(pool::close);
            try {
                closing.get(5, TimeUnit.SECONDS);
            } finally {
                driver.shutdownNow();
                assertTrue(driver.awaitTermination(5, TimeUnit.SECONDS));
            }
        }
    }

    @org.junit.jupiter.api.Test
    void shutdownWaitsForRejectedResponseHandoffAndItsAcceptedContinuation() throws Exception {
        var pool = new RequestTaskPool();
        var context = RequestProtocolTestSupport.context(SchedulingTestConfig.batchConfig(), 7105L);
        var dispatchEntered = new CountDownLatch(1);
        var rejectDispatch = new CountDownLatch(1);
        var rejectionEntered = new CountDownLatch(1);
        var continueRejection = new CountDownLatch(1);
        var childEntered = new CountDownLatch(1);
        var finishChild = new CountDownLatch(1);
        var firstSubmission = new java.util.concurrent.atomic.AtomicBoolean(true);
        var rejections = new AtomicInteger();
        var responses = new AtomicInteger();
        var childDone = new AtomicInteger();
        var failure = new java.util.concurrent.RejectedExecutionException("dispatch rejected");
        var dispatch = org.mockito.Mockito.mock(java.util.concurrent.ExecutorService.class);
        var workers = Executors.newFixedThreadPool(3, Thread.ofPlatform().daemon().factory());
        org.mockito.Mockito.doAnswer(
                        call -> {
                            var task = call.getArgument(0, Runnable.class);
                            if (firstSubmission.getAndSet(false)) {
                                dispatchEntered.countDown();
                                await(rejectDispatch);
                                throw failure;
                            }
                            // Leave pending starts behind the blocked handoff. Once rejection
                            // callbacks begin, their independent continuation can run.
                            if (rejectionEntered.getCount() == 0) workers.execute(task);
                            return null;
                        })
                .when(dispatch)
                .execute(org.mockito.ArgumentMatchers.any());
        org.mockito.Mockito.when(dispatch.isTerminated()).thenReturn(true);
        org.springframework.test.util.ReflectionTestUtils.setField(
                pool, "requestWorkers", dispatch);
        var first = pool.registerRequestWork(context);
        Future<?> closing = null;
        try {
            var submission =
                    workers.submit(
                            () ->
                                    org.junit.jupiter.api.Assertions.assertSame(
                                            failure,
                                            org.junit.jupiter.api.Assertions.assertThrows(
                                                    java.util.concurrent.RejectedExecutionException
                                                            .class,
                                                    () ->
                                                            first.submitClose(
                                                                    responses::incrementAndGet,
                                                                    false,
                                                                    cause ->
                                                                            rejections
                                                                                    .incrementAndGet()))));
            assertTrue(dispatchEntered.await(5, TimeUnit.SECONDS));
            first.close();
            try (var pending = pool.registerRequestWork(context)) {
                pending.submitCloseResponse(
                        responses::incrementAndGet,
                        false,
                        cause -> {
                            org.junit.jupiter.api.Assertions.assertSame(failure, cause);
                            rejections.incrementAndGet();
                            rejectionEntered.countDown();
                            await(continueRejection);
                            pool.execute(
                                    context,
                                    () -> {
                                        childEntered.countDown();
                                        await(finishChild);
                                        childDone.incrementAndGet();
                                    });
                        });
            }
            closing = workers.submit(pool::close);
            long limit = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
            while (true) {
                var probe = pool.tryRegisterRequestWork(context);
                if (probe == null) break;
                probe.close();
                assertTrue(System.nanoTime() < limit, "shutdown did not close admission");
                Thread.yield();
            }
            rejectDispatch.countDown();
            assertTrue(rejectionEntered.await(5, TimeUnit.SECONDS));
            assertEquals(2, rejections.get());
            var pendingShutdown = closing;
            org.junit.jupiter.api.Assertions.assertThrows(
                    java.util.concurrent.TimeoutException.class,
                    () -> pendingShutdown.get(100, TimeUnit.MILLISECONDS),
                    "rejection cleanup still owns the last counted task");
            continueRejection.countDown();
            assertTrue(childEntered.await(5, TimeUnit.SECONDS));
            submission.get(5, TimeUnit.SECONDS);
            assertFalse(closing.isDone(), "the accepted rejection continuation must also drain");
            finishChild.countDown();
            closing.get(5, TimeUnit.SECONDS);
            assertEquals(0, responses.get());
            assertEquals(1, childDone.get());
            org.junit.jupiter.api.Assertions.assertSame(
                    failure,
                    org.junit.jupiter.api.Assertions.assertThrows(
                            java.util.concurrent.RejectedExecutionException.class,
                            pool::throwIfFailed));
        } finally {
            rejectDispatch.countDown();
            continueRejection.countDown();
            finishChild.countDown();
            first.close();
            if (closing == null) closing = workers.submit(pool::close);
            try {
                closing.get(5, TimeUnit.SECONDS);
            } finally {
                workers.shutdownNow();
                assertTrue(workers.awaitTermination(5, TimeUnit.SECONDS));
            }
        }
    }

    private static void await(CountDownLatch latch) {
        try {
            assertTrue(latch.await(5, TimeUnit.SECONDS));
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new AssertionError(e);
        }
    }
}
