package org.flexlb.balance.schedule;

import org.junit.jupiter.api.Test;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Verifies that executor drain covers submitted tasks and their accepted continuations.
 */
class RequestTaskPoolConcurrencyTest {
    @Test
    void interruptedCloseStillRunsEveryAcceptedTask() throws Exception {
        var pool = new RequestTaskPool();
        var workers =
                (java.util.concurrent.ThreadPoolExecutor)
                        org.springframework.test.util.ReflectionTestUtils.getField(
                                pool, "requestWorkers");
        var entered = new CountDownLatch(workers.getCorePoolSize());
        var release = new CountDownLatch(1);
        var queuedRan = new java.util.concurrent.atomic.AtomicBoolean();
        var closed = new CompletableFuture<Boolean>();
        for (int i = 0; i < workers.getCorePoolSize(); i++) {
            pool.execute(
                    () -> {
                        entered.countDown();
                        RequestProtocolTestSupport.await(release);
                    });
        }
        assertTrue(entered.await(2, TimeUnit.SECONDS));
        pool.execute(() -> queuedRan.set(true));
        var closer =
                Thread.ofPlatform()
                        .start(
                                () -> {
                                    Thread.currentThread().interrupt();
                                    pool.close();
                                    closed.complete(Thread.currentThread().isInterrupted());
                                });
        try {
            RequestProtocolTestSupport.awaitCondition(workers::isShutdown);
            assertFalse(closed.isDone());
        } finally {
            release.countDown();
            assertTrue(closed.get(2, TimeUnit.SECONDS));
            closer.join();
        }
        assertTrue(queuedRan.get());
    }

    @Test
    void drainWaitsForSubmittedTaskAndItsContinuation() throws Exception {
        var pool = new RequestTaskPool();
        var entered = new CountDownLatch(1);
        var release = new CountDownLatch(1);
        var childEntered = new CountDownLatch(1);
        var childRelease = new CountDownLatch(1);
        try {
            pool.execute(
                    () -> {
                        entered.countDown();
                        RequestProtocolTestSupport.await(release);
                        pool.execute(
                                () -> {
                                    childEntered.countDown();
                                    RequestProtocolTestSupport.await(childRelease);
                                });
                    });
            assertTrue(entered.await(2, TimeUnit.SECONDS));
            var drained = CompletableFuture.runAsync(pool::awaitIdle);
            release.countDown();
            assertTrue(childEntered.await(2, TimeUnit.SECONDS));
            assertFalse(drained.isDone());
            childRelease.countDown();
            drained.get(2, TimeUnit.SECONDS);
        } finally {
            release.countDown();
            childRelease.countDown();
            pool.close();
        }
    }

    @Test
    void closeRejectsNewTasksAndWaitsForAcceptedTask() throws Exception {
        var pool = new RequestTaskPool();
        var entered = new CountDownLatch(1);
        var release = new CountDownLatch(1);
        pool.execute(
                () -> {
                    entered.countDown();
                    RequestProtocolTestSupport.await(release);
                });
        assertTrue(entered.await(2, TimeUnit.SECONDS));
        var closing = CompletableFuture.runAsync(pool::close);
        try {
            var workers =
                    (java.util.concurrent.ExecutorService)
                            org.springframework.test.util.ReflectionTestUtils.getField(
                                    pool, "requestWorkers");
            RequestProtocolTestSupport.awaitCondition(workers::isShutdown);
            assertThrows(RejectedExecutionException.class, () -> pool.execute(() -> {}));
            assertFalse(closing.isDone());
        } finally {
            release.countDown();
            closing.get(2, TimeUnit.SECONDS);
        }
    }

    @Test
    void taskFailureDoesNotStrandDrainOrPreventLaterTasks() throws Exception {
        var pool = new RequestTaskPool();
        try {
            pool.execute(
                    () -> {
                        throw new IllegalStateException("task failure");
                    });
            pool.awaitIdle();
            var next = new CompletableFuture<Void>();
            pool.execute(() -> next.complete(null));
            next.get(2, TimeUnit.SECONDS);
            pool.awaitIdle();
        } finally {
            pool.close();
        }
    }

    @Test
    void taskCannotWaitForItsOwnExecutor() throws Exception {
        var pool = new RequestTaskPool();
        try {
            var checked = new CompletableFuture<Void>();
            pool.execute(
                    () -> {
                        assertThrows(IllegalStateException.class, pool::awaitIdle);
                        assertThrows(IllegalStateException.class, pool::close);
                        checked.complete(null);
                    });
            checked.get(2, TimeUnit.SECONDS);
        } finally {
            pool.close();
        }
    }
}
