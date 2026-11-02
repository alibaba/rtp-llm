package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.junit.jupiter.api.Test;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.LockSupport;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class SchedulerLoopTest {
    private abstract static class Policy implements RequestScheduler {
        public void onRequest(RequestContext request) {}
        public void onRequestChanged(RequestContext request) {}
        public void onEndpointChanged(WorkerEndpoint endpoint) {}
        public void close() {}
    }

    @Test
    void customSchedulerDeadlineRunsWithoutAnotherNotification() throws Exception {
        var calls = new AtomicInteger();
        var deadline = new AtomicLong();
        var reached = new CountDownLatch(1);
        var stopped = new CountDownLatch(1);
        var policy = new Policy() {
            public long schedule(long now) {
                if (calls.incrementAndGet() == 1) {
                    deadline.set(now + TimeUnit.MILLISECONDS.toNanos(20));
                    return deadline.get();
                }
                if (now >= deadline.get()) reached.countDown();
                return now < deadline.get() ? deadline.get() : Long.MAX_VALUE;
            }
        };
        var loop = new SchedulerLoop(Thread.ofPlatform().daemon(), policy, stopped::countDown);
        loop.start();
        try {
            assertTrue(reached.await(2, TimeUnit.SECONDS));
        } finally {
            loop.stop();
            loop.awaitStopped();
        }
        assertEquals(0, stopped.getCount());
    }

    @Test
    void notificationConsumedInsideDecisionStillCausesAnotherDecision() throws Exception {
        var calls = new AtomicInteger();
        var entered = new CountDownLatch(1);
        var notified = new CountDownLatch(1);
        var nextDecision = new CountDownLatch(1);
        var policy = new Policy() {
            public long schedule(long now) {
                if (calls.incrementAndGet() == 1) {
                    entered.countDown();
                    // Model a blocking operation consuming the notification's park permit.
                    RequestProtocolTestSupport.await(notified);
                    LockSupport.parkNanos(TimeUnit.MILLISECONDS.toNanos(1));
                } else nextDecision.countDown();
                return Long.MAX_VALUE;
            }
        };
        var loop = new SchedulerLoop(Thread.ofPlatform().daemon(), policy, () -> {});
        loop.start();
        try {
            assertTrue(entered.await(2, TimeUnit.SECONDS));
            loop.wake();
            notified.countDown();
            assertTrue(nextDecision.await(2, TimeUnit.SECONDS));
        } finally {
            notified.countDown();
            loop.stop();
            loop.awaitStopped();
        }
    }
}
