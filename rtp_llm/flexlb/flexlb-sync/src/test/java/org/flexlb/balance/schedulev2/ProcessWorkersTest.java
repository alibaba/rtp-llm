package org.flexlb.balance.schedulev2;

import org.flexlb.balance.scheduler.CancelReason;
import org.flexlb.balance.scheduler.SchedulingTestConfig;
import org.flexlb.balance.schedulev2.request.Request;
import org.flexlb.config.FlexlbConfig;
import org.junit.jupiter.api.Test;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ProcessWorkersTest {

    private final FlexlbConfig config = SchedulingTestConfig.newConfig();
    private final LoopSignal signal = new LoopSignal();

    @Test
    void twoSelectionsRunTogetherAndPendingWorkIsBounded() throws Exception {
        ExecutorService executor = Executors.newFixedThreadPool(2);
        CountDownLatch entered = new CountDownLatch(2);
        CountDownLatch finish = new CountDownLatch(1);
        Request first = request("1", 50);
        Request second = request("2", 50);
        try (ProcessWorkers<AutoCloseable> workers = new ProcessWorkers<>(executor, 2, signal, true)) {
            assertTrue(workers.submit(first, () -> waitingResult(entered, finish)));
            assertTrue(workers.submit(second, () -> waitingResult(entered, finish)));
            assertTrue(entered.await(5, TimeUnit.SECONDS));
            assertFalse(workers.submit(first, () -> () -> {}));
            assertFalse(workers.submit(request("3", 50), () -> () -> {}));
            finish.countDown();
            awaitResult(workers, first).close();
            awaitResult(workers, second).close();
            assertTrue(workers.submit(request("4", 50), () -> () -> {}));
        } finally {
            finish.countDown();
            executor.shutdownNow();
            assertTrue(executor.awaitTermination(5, TimeUnit.SECONDS));
        }
    }

    @Test
    void cancelledSelectionClosesLateResultOnce() throws Exception {
        verifyLateResult(false);
    }

    @Test
    void shutdownClosesLateResultOnce() throws Exception {
        verifyLateResult(true);
    }

    private void verifyLateResult(boolean shutdown) throws Exception {
        ExecutorService executor = Executors.newSingleThreadExecutor();
        CountDownLatch entered = new CountDownLatch(1);
        CountDownLatch finish = new CountDownLatch(1);
        AtomicInteger closes = new AtomicInteger();
        Request request = request("1", 50);
        ProcessWorkers<AutoCloseable> workers = new ProcessWorkers<>(executor, 1, signal, true);
        try {
            workers.submit(request, () -> {
                waitingResult(entered, finish);
                return closes::incrementAndGet;
            });
            assertTrue(entered.await(5, TimeUnit.SECONDS));
            if (shutdown) {
                workers.close();
            } else {
                request.cancel(CancelReason.CLIENT_CANCELLED);
            }
            finish.countDown();
            executor.shutdown();
            assertTrue(executor.awaitTermination(5, TimeUnit.SECONDS));
            workers.removeEnded();
            workers.close();
            assertEquals(1, closes.get());
        } finally {
            finish.countDown();
            workers.close();
            executor.shutdownNow();
        }
    }

    @Test
    void highPriorityHeadCanReplaceLowerPriorityCompletedResult() throws Exception {
        AtomicInteger closes = new AtomicInteger();
        try (ProcessWorkers<AutoCloseable> workers =
                new ProcessWorkers<>(new DirectExecutorService(), 1, signal, true)) {
            Request low = request("1", 10);
            Request high = request("2", 80);
            assertTrue(workers.submit(low, () -> closes::incrementAndGet));
            assertTrue(workers.submit(high, () -> () -> {}));
            assertEquals(1, closes.get());
            assertFalse(low.future().isDone());
            awaitResult(workers, high).close();
        }
    }

    @Test
    void errorReturnsToSchedulerAndWorkerCanTakeNextRequest() throws Exception {
        AssertionError failure = new AssertionError("selection failed");
        try (ProcessWorkers<AutoCloseable> workers =
                new ProcessWorkers<>(new DirectExecutorService(), 1, signal, true)) {
            Request first = request("1", 50);
            workers.submit(first, () -> {
                throw failure;
            });
            try (SelectionResult<AutoCloseable> result = awaitResult(workers, first)) {
                assertSame(failure, result.failure());
            }
            assertTrue(workers.submit(request("2", 50), () -> () -> {}));
        }
    }

    private Request request(String id, int priority) {
        Request request = V2TestSupport.request(config, id, priority, 16L);
        request.setQueueSeq(Long.parseLong(id));
        return request;
    }

    @Test
    void cachedMaterialsDoNotOccupyWorkerSlotsAndCloseOnceOnCancellation() {
        AtomicInteger closes = new AtomicInteger();
        try (ProcessWorkers<AutoCloseable> workers =
                new ProcessWorkers<>(new DirectExecutorService(), 1, signal, true)) {
            Request first = request("1", 50);
            workers.submit(first, () -> closes::incrementAndGet);
            SelectionResult<AutoCloseable> result = workers.get(first, 30_000L);
            assertSame(result, workers.get(first, 30_000L));
            assertTrue(workers.submit(request("2", 50), () -> () -> {}));
            first.cancel(CancelReason.CLIENT_CANCELLED);
            workers.removeEnded();
            workers.discard(first);
            assertNull(workers.get(first, 30_000L));
            assertEquals(1, closes.get());
        }
    }

    @Test
    void discardedRunningPreparationCannotReturnAfterRequestChangesStage() throws Exception {
        ExecutorService executor = Executors.newSingleThreadExecutor();
        CountDownLatch entered = new CountDownLatch(1);
        CountDownLatch finish = new CountDownLatch(1);
        AtomicInteger closes = new AtomicInteger();
        try (ProcessWorkers<AutoCloseable> workers = new ProcessWorkers<>(executor, 1, signal, true)) {
            Request request = request("1", 50);
            workers.submit(request, () -> {
                waitingResult(entered, finish);
                return closes::incrementAndGet;
            });
            assertTrue(entered.await(5, TimeUnit.SECONDS));
            workers.discard(request);
            assertFalse(workers.submit(request, () -> () -> {}));
            finish.countDown();
            executor.shutdown();
            assertTrue(executor.awaitTermination(5, TimeUnit.SECONDS));
            assertNull(workers.get(request, 30_000L));
            assertEquals(1, closes.get());
        } finally {
            finish.countDown();
            executor.shutdownNow();
        }
    }

    private static AutoCloseable waitingResult(CountDownLatch entered, CountDownLatch finish) {
        entered.countDown();
        try {
            assertTrue(finish.await(5, TimeUnit.SECONDS));
        } catch (InterruptedException failure) {
            Thread.currentThread().interrupt();
            throw new AssertionError(failure);
        }
        return () -> {};
    }

    private SelectionResult<AutoCloseable> awaitResult(ProcessWorkers<AutoCloseable> workers, Request request) {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
        SelectionResult<AutoCloseable> result;
        while ((result = workers.take(request)) == null && System.nanoTime() < deadline) {
            signal.await(10L);
        }
        assertNotNull(result);
        return result;
    }
}
