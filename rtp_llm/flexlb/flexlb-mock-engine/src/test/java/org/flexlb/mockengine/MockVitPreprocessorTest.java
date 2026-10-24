package org.flexlb.mockengine;

import org.junit.jupiter.api.Test;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import static org.junit.jupiter.api.Assertions.*;

class MockVitPreprocessorTest {
    @Test void localPreprocessingIsInlineAndHasNoQueue() throws Exception {
        try (var pool = new MockVitPreprocessor(true, 1)) {
            var caller = Thread.currentThread();
            assertSame(caller, pool.submit(Thread::currentThread).get());
            assertEquals(0, pool.pending());
        }
    }
    @Test void workerLimitAndPendingIncludeRunningAndQueuedUntilActualCompletion() throws Exception {
        var entered = new CountDownLatch(1);
        var release = new CountDownLatch(1);
        try (var pool = new MockVitPreprocessor(false, 1)) {
            var first = pool.submit(() -> { entered.countDown(); release.await(); return 1; });
            assertTrue(entered.await(1, TimeUnit.SECONDS));
            var next = pool.submit(() -> 2);
            assertEquals(2, pool.pending());
            assertFalse(next.isDone());
            // Real preprocess futures do not dequeue canceled work from multiprocessing.Pool.
            first.cancel(false);
            assertEquals(2, pool.pending());
            release.countDown();
            assertEquals(2, next.get(1, TimeUnit.SECONDS));
            long until = System.nanoTime() + TimeUnit.SECONDS.toNanos(1);
            while (pool.pending() != 0 && System.nanoTime() < until) Thread.yield();
            assertEquals(0, pool.pending());
        } finally { release.countDown(); }
    }
    @Test void failedWorkReleasesCapacityAndDoesNotPoisonWorker() throws Exception {
        try (var pool = new MockVitPreprocessor(false, 1)) {
            var failed = pool.submit(() -> { throw new IllegalArgumentException("invalid image"); });
            assertThrows(java.util.concurrent.ExecutionException.class, () -> failed.get(1, TimeUnit.SECONDS));
            assertEquals(7, pool.submit(() -> 7).get(1, TimeUnit.SECONDS));
        }
    }
    @Test void shutdownSettlesQueuedFuturesAndClearsPending() throws Exception {
        var pool = new MockVitPreprocessor(false, 1);
        var entered = new CountDownLatch(1); var release = new CountDownLatch(1);
        var active = pool.submit(() -> { entered.countDown(); release.await(); return 1; });
        assertTrue(entered.await(1, TimeUnit.SECONDS));
        var queued = pool.submit(() -> 2);
        pool.close(); release.countDown();
        assertTrue(active.isDone()); assertTrue(queued.isCancelled());
        assertEquals(0, pool.pending());
        assertTrue(pool.submit(() -> 3).isCompletedExceptionally());
    }

}
