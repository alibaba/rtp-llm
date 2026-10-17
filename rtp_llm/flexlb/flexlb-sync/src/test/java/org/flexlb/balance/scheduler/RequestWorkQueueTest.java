package org.flexlb.balance.scheduler;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.*;

class RequestWorkQueueTest {
    @Test
    @Timeout(10)
    void duplicateRunnerCannotExecuteTheNextSettlementConcurrently() throws Exception {
        var queue = new RequestWorkQueue();
        var context = RequestProtocolTestSupport.context(SchedulingTestConfig.batchConfig(), 7101L);
        var entered = new CountDownLatch(1);
        var release = new CountDownLatch(1);
        var next = new CountDownLatch(1);
        var driver = new AtomicReference<Runnable>();
        var workers = Executors.newFixedThreadPool(2);
        try {
            assertTrue(queue.execute(context, () -> {
                entered.countDown();
                RequestProtocolTestSupport.await(release);
            }, fail -> fail(fail), task -> { driver.set(task); workers.execute(task); }));
            assertTrue(entered.await(5, TimeUnit.SECONDS));
            assertTrue(queue.execute(context, next::countDown, fail -> fail(fail), workers::execute));
            workers.submit(driver.get()).get(5, TimeUnit.SECONDS);
            assertEquals(1, next.getCount(), "a fallback runner cannot pass the executing settlement");
            release.countDown();
            assertTrue(next.await(5, TimeUnit.SECONDS));
        } finally {
            release.countDown();
            workers.shutdownNow();
            assertTrue(workers.awaitTermination(5, TimeUnit.SECONDS));
        }
    }

    @Test
    void failedStartRejectsCountedWorkAndAllowsTheNextNotification() {
        var queue = new RequestWorkQueue();
        var context = RequestProtocolTestSupport.context(SchedulingTestConfig.batchConfig(), 7102L);
        var failure = new IllegalStateException("all executors rejected");
        var rejected = new AtomicInteger();
        var ran = new AtomicInteger();
        assertSame(failure, assertThrows(IllegalStateException.class,
                () -> queue.execute(context, ran::incrementAndGet, cause -> {
                    assertSame(failure, cause);
                    rejected.incrementAndGet();
                }, work -> { throw failure; })));
        assertEquals(0, ran.get());
        assertEquals(1, rejected.get());
        assertTrue(queue.execute(context, ran::incrementAndGet, cause -> fail(cause), Runnable::run));
        assertEquals(1, ran.get(), "failed handoff cannot strand the request lane");
    }

    @Test
    void pendingNotificationsCoalesceAndAnIdleLaneDoesNotRetainRequestIdentity() {
        var queue = new RequestWorkQueue();
        var context = RequestProtocolTestSupport.context(SchedulingTestConfig.batchConfig(), 7103L);
        var driver = new AtomicReference<Runnable>();
        var ran = new AtomicInteger();
        assertTrue(queue.execute(context, ran::incrementAndGet, cause -> fail(cause), driver::set));
        for (int i = 0; i < 100; i++) {
            assertFalse(queue.execute(context, ran::incrementAndGet, cause -> fail(cause), driver::set));
        }
        driver.get().run();
        assertEquals(1, ran.get());
        assertEquals(0, ((java.util.Map<?, ?>) org.springframework.test.util.ReflectionTestUtils.getField(queue, "lanes")).size());
        assertTrue(queue.execute(context, ran::incrementAndGet, cause -> fail(cause), Runnable::run));
        assertEquals(2, ran.get());
    }
}
