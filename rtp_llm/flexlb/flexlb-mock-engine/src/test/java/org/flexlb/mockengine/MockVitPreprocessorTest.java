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

    @Test void brokenSubmissionRebuildsAndRetriesExactlyOnce() throws Exception {
        try (var pool = new MockVitPreprocessor(false, 1)) {
            var calls = new java.util.concurrent.atomic.AtomicInteger();
            pool.dispatch = (executor, task) -> {
                if (calls.incrementAndGet() == 1) throw new java.io.UncheckedIOException(new java.io.IOException("broken pipe"));
                executor.execute(task);
            };
            assertEquals(7, pool.await(pool.submit(() -> 7), 1000)); assertEquals(2, calls.get());
            pool.dispatch = (executor, task) -> { calls.incrementAndGet(); throw new java.io.UncheckedIOException(new java.io.IOException("still broken")); };
            int before = calls.get();
            assertThrows(java.util.concurrent.ExecutionException.class, () -> pool.await(pool.submit(() -> 8), 1000));
            assertEquals(before + 2, calls.get()); assertEquals(0, pool.pending());
        }
    }
    @Test void brokenResultRebuildsAndDropsOldWorkerUrlCache() throws Exception {
        try (var pool = new MockVitPreprocessor(false, 1)) {
            pool.configureMediaCache(100);
            pool.await(pool.submit(() -> { pool.mediaCache().put("a", new byte[]{1}); return 1; }), 1000);
            assertEquals(1, pool.mediaCacheItems());
            assertThrows(java.util.concurrent.ExecutionException.class,
                    () -> pool.await(pool.submit(() -> { throw new java.io.EOFException("worker died"); }), 1000));
            assertEquals(0, pool.mediaCacheItems());
            assertEquals(3, pool.await(pool.submit(() -> 3), 1000));
        }
    }
    @Test void consecutiveTimeoutThresholdRebuildsWithoutReportingLateTiming() throws Exception {
        try (var pool = new MockVitPreprocessor(false, 2)) {
            var timings = new java.util.concurrent.atomic.AtomicInteger();
            pool.setMetrics(values -> { if (values.containsKey("py_rtp_vit_preprocess_rt")) timings.incrementAndGet(); });
            var never = new java.util.concurrent.CompletableFuture<Integer>();
            pool.await(pool.submit(() -> { pool.mediaCache().put("a", new byte[]{1}); return 1; }), 1000);
            assertThrows(java.util.concurrent.TimeoutException.class, () -> pool.await(never, 1));
            assertEquals(1, pool.mediaCacheItems());
            assertThrows(java.util.concurrent.TimeoutException.class, () -> pool.await(never, 1));
            assertEquals(0, pool.mediaCacheItems());
            var completed = pool.submit(() -> 7); assertEquals(7, completed.get(1, TimeUnit.SECONDS));
            assertEquals(1, timings.get(), "completion alone does not report consumed preprocessing timing");
            assertEquals(7, pool.await(completed, 1000)); assertEquals(2, timings.get());
        }
    }
    @Test void spawnWorkersOwnIndependentDefaultCachesWhileLocalUsesConfiguredCapacity() throws Exception {
        try (var pool = new MockVitPreprocessor(false, 2)) {
            pool.configureMediaCache(1);
            for (int i = 0; i < 2; i++) pool.await(pool.submit(() -> {
                pool.mediaCache().put("a", new byte[]{1}); pool.mediaCache().put("b", new byte[]{2}); return 1;
            }), 1000);
            assertEquals(4, pool.mediaCacheItems(), "spawn children import separate default 10-item caches");
        }
        try (var pool = new MockVitPreprocessor(true, 2)) {
            pool.configureMediaCache(1); pool.mediaCache().put("a", new byte[]{1}); pool.mediaCache().put("b", new byte[]{2});
            assertEquals(1, pool.mediaCacheItems());
        }
    }
}
