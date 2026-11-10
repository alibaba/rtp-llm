package org.flexlb.balance.scheduler;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Verifies serialized cleanup, required response ordering, and rejection of queued close operations.
 */
class RequestCloseQueueTest {
    @Test
    void cleanupCoalescingCannotDiscardAnAcceptedResponse() {
        var queue = new RequestCloseQueue();
        var owner = new Object();
        var runner = new AtomicReference<Runnable>();
        var events = new ArrayList<String>();
        queue.submit(owner, () -> events.add("cleanup"), runner::set, true, null);
        queue.submit(owner, () -> events.add("response"), runner::set, false, null);
        queue.submit(owner, () -> events.add("duplicate cleanup"), runner::set, true, null);
        runner.get().run();
        queue.awaitIdle();
        assertEquals(List.of("cleanup", "response"), events);
    }

    @Test
    void drainWaitsForInlineCleanupAndRejectsWaitingOnItself() throws Exception {
        var queue = new RequestCloseQueue();
        var entered = new CountDownLatch(1);
        var release = new CountDownLatch(1);
        var caller =
                java.util.concurrent.CompletableFuture.runAsync(
                        () ->
                                queue.submit(
                                        new Object(),
                                        () -> {
                                            assertThrows(
                                                    IllegalStateException.class, queue::awaitIdle);
                                            entered.countDown();
                                            RequestProtocolTestSupport.await(release);
                                        },
                                        Runnable::run,
                                        true,
                                        null));
        try {
            assertTrue(entered.await(2, TimeUnit.SECONDS));
            var drained = java.util.concurrent.CompletableFuture.runAsync(queue::awaitIdle);
            org.junit.jupiter.api.Assertions.assertFalse(drained.isDone());
            release.countDown();
            caller.get(2, TimeUnit.SECONDS);
            drained.get(2, TimeUnit.SECONDS);
        } finally {
            release.countDown();
            caller.get(2, TimeUnit.SECONDS);
        }
    }

    @Test
    void duplicateRunnerCannotPassTheExecutingCleanup() throws Exception {
        var queue = new RequestCloseQueue();
        var owner = new Object();
        var entered = new CountDownLatch(1);
        var release = new CountDownLatch(1);
        var next = new CountDownLatch(1);
        var runner = new AtomicReference<Runnable>();
        var workers = Executors.newFixedThreadPool(2);
        try {
            queue.submit(
                    owner,
                    () -> {
                        entered.countDown();
                        RequestProtocolTestSupport.await(release);
                    },
                    task -> {
                        runner.set(task);
                        workers.execute(task);
                    },
                    true,
                    null);
            assertTrue(entered.await(2, TimeUnit.SECONDS));
            queue.submit(owner, next::countDown, workers::execute, true, null);
            workers.submit(runner.get()).get(2, TimeUnit.SECONDS);
            assertEquals(1, next.getCount());
            release.countDown();
            assertTrue(next.await(2, TimeUnit.SECONDS));
        } finally {
            release.countDown();
            workers.shutdown();
            assertTrue(workers.awaitTermination(2, TimeUnit.SECONDS));
        }
    }

    @Test
    void failedStartRejectsEveryQueuedOperationAndAllowsRetry() {
        var queue = new RequestCloseQueue();
        var owner = new Object();
        var failure = new IllegalStateException("executor rejected");
        var rejected = new AtomicInteger();
        var ran = new AtomicInteger();
        assertSame(
                failure,
                assertThrows(
                        IllegalStateException.class,
                        () ->
                                queue.submit(
                                        owner,
                                        ran::incrementAndGet,
                                        task -> {
                                            queue.submit(
                                                    owner,
                                                    ran::incrementAndGet,
                                                    Runnable::run,
                                                    false,
                                                    cause -> {
                                                        assertSame(failure, cause);
                                                        rejected.incrementAndGet();
                                                    });
                                            throw failure;
                                        },
                                        true,
                                        cause -> {
                                            assertSame(failure, cause);
                                            rejected.incrementAndGet();
                                        })));
        assertEquals(2, rejected.get());
        assertEquals(0, ran.get());
        queue.submit(owner, ran::incrementAndGet, Runnable::run, true, null);
        assertEquals(1, ran.get());
    }

    @Test
    void responsesRemainOrderedWhileCleanupNotificationsCoalesce() {
        var queue = new RequestCloseQueue();
        var owner = new Object();
        var runner = new AtomicReference<Runnable>();
        var effects = new ArrayList<String>();
        queue.submit(owner, () -> effects.add("cleanup"), runner::set, true, null);
        queue.submit(owner, () -> effects.add("response-1"), runner::set, false, null);
        queue.submit(owner, () -> effects.add("response-2"), runner::set, false, null);
        queue.submit(owner, () -> effects.add("duplicate-cleanup"), runner::set, true, null);
        runner.get().run();
        assertEquals(List.of("cleanup", "response-1", "response-2"), effects);
        queue.submit(owner, () -> effects.add("next-cleanup"), Runnable::run, true, null);
        assertEquals(4, effects.size());
        assertEquals(
                0,
                ((java.util.Map<?, ?>)
                                org.springframework.test.util.ReflectionTestUtils.getField(
                                        queue, "lanes"))
                        .size());
    }
}
