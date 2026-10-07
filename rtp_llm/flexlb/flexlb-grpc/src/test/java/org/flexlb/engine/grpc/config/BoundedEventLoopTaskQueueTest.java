package org.flexlb.engine.grpc.config;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Queue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.RejectedExecutionException;
import org.junit.jupiter.api.Test;

/**
 * Regression for the case55 master-fault root cause (2026-10-08, diagnostic
 * execution 92649): the gRPC client event-loop task queue was UNBOUNDED
 * (PlatformDependent::newMpscQueue), so under 512-1024-thread ramp pressure
 * the queue grew without limit, its tasks retained CompositeByteBuf payloads
 * until the JVM exhausted the Java heap (45.6 GB dump) and deep composite
 * release recursion overflowed the stack.
 *
 * <p>The fix bounds the per-event-loop task queue; saturation surfaces as a
 * rejected task instead of unbounded retention.
 */
class BoundedEventLoopTaskQueueTest {

    private static final int CAP = ChannelConfiguration.GRPC_CLIENT_EVENT_LOOP_TASK_QUEUE_CAPACITY;

    @Test
    void capacityIsBounded() {
        assertTrue(CAP > 0, "task-queue capacity must be positive");
        // The bound is exactly the documented constant (10_000).
        assertEquals(10_000, CAP);
    }

    @Test
    void queueFactoryBoundsNettyProvidedMaxPending() {
        // The factory contract: newTaskQueue(int maxPendingTasks) must never
        // exceed the configured cap even when Netty asks for more.
        Queue<Runnable> q =
                ChannelConfiguration.boundedTaskQueue(Integer.MAX_VALUE);
        assertEquals(CAP, ((LinkedBlockingQueue<?>) q).remainingCapacity() + q.size());

        // Below the cap, Netty's requested bound is respected.
        Queue<Runnable> small = ChannelConfiguration.boundedTaskQueue(64);
        assertEquals(64, ((LinkedBlockingQueue<?>) small).remainingCapacity() + small.size());
    }

    @Test
    void boundedQueueRejectsBeyondCapacity() {
        // Saturation must be observable (RejectedExecutionException from the
        // queue's offer contract / executor rejection), not silent retention.
        LinkedBlockingQueue<Runnable> q = ChannelConfiguration.boundedTaskQueue(2);
        Runnable noop = () -> {};
        assertTrue(q.offer(noop));
        assertTrue(q.offer(noop));
        // A bounded LinkedBlockingQueue returns false (or throws for add()).
        assertTrue(!q.offer(noop));
        assertThrows(IllegalStateException.class, () -> q.add(noop));
    }
}
