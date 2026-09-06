package org.flexlb.balance.scheduler;

import org.junit.jupiter.api.Test;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class RequestLifecycleTest {

    @Test
    void normalBatchDeliveryTransitionsToCompleted() {
        RequestLifecycle lifecycle = new RequestLifecycle("1");

        assertFalse(lifecycle.hasDeliveryClaim());
        lifecycle.startBatchEnqueue(101L);
        assertTrue(lifecycle.hasDeliveryClaim());
        assertEquals(RequestLifecycleState.DISPATCHING, lifecycle.snapshot().state());
        assertEquals(DeliveryClaimKind.BATCH_ENQUEUE, lifecycle.snapshot().deliveryClaimKind());
        assertEquals(RequestLifecycleState.ACKNOWLEDGED,
                lifecycle.markDeliveryConfirmed().state());
        RequestLifecycleSnapshot completed = lifecycle.complete("decode finished");

        assertEquals(RequestLifecycleState.COMPLETED, completed.state());
        assertEquals(101L, completed.batchId());
        assertTrue(completed.state().isTerminal());
    }

    @Test
    void deadlineTransitionsDirectlyToTimedOutAndCannotBeOverwritten() {
        RequestLifecycle lifecycle = new RequestLifecycle("3");
        lifecycle.startBatchEnqueue(103L);

        RequestLifecycleSnapshot timedOut = lifecycle.timeout("deadline exceeded");
        assertEquals(RequestLifecycleState.TIMED_OUT, timedOut.state());
        assertEquals(RequestLifecycleState.TIMED_OUT,
                lifecycle.markDeliveryConfirmed().state());
        assertEquals(RequestLifecycleState.TIMED_OUT, lifecycle.fail("late failure").state());
        assertEquals(RequestLifecycleState.TIMED_OUT, lifecycle.complete("late completion").state());
    }

    @Test
    void routeDecisionDeliveryAcquiresRequestScopedDeliveryClaim() {
        RequestLifecycle lifecycle = new RequestLifecycle("4");

        lifecycle.startRouteDecisionDelivery();
        RequestLifecycleSnapshot acknowledged = lifecycle.markDeliveryConfirmed();

        assertTrue(lifecycle.hasDeliveryClaim());
        assertEquals(RequestLifecycleState.ACKNOWLEDGED, acknowledged.state());
        assertEquals(DeliveryClaimKind.ROUTE_DECISION, acknowledged.deliveryClaimKind());
        assertEquals(0L, acknowledged.batchId());
        assertEquals("route decision delivered", acknowledged.detail());
    }

    @Test
    void deliveryClaimKindAndBatchOwnershipCannotChangeAfterClaim() {
        RequestLifecycle batchLifecycle = new RequestLifecycle("5");
        batchLifecycle.startBatchEnqueue(105L);
        batchLifecycle.startBatchEnqueue(105L);

        assertThrows(IllegalStateException.class, batchLifecycle::startRouteDecisionDelivery);
        assertThrows(IllegalStateException.class, () -> batchLifecycle.startBatchEnqueue(106L));
        assertEquals(DeliveryClaimKind.BATCH_ENQUEUE, batchLifecycle.snapshot().deliveryClaimKind());
        assertEquals(105L, batchLifecycle.snapshot().batchId());

        RequestLifecycle routeLifecycle = new RequestLifecycle("6");
        routeLifecycle.startRouteDecisionDelivery();
        routeLifecycle.startRouteDecisionDelivery();

        assertThrows(IllegalStateException.class,
                () -> routeLifecycle.startBatchEnqueue(106L));
        assertEquals(DeliveryClaimKind.ROUTE_DECISION, routeLifecycle.snapshot().deliveryClaimKind());
        assertEquals(0L, routeLifecycle.snapshot().batchId());
    }

    @Test
    void rejectedLateDeliveryDoesNotLeavePartialClaim() {
        RequestLifecycle lifecycle = new RequestLifecycle("7");
        lifecycle.timeout("expired in queue");

        assertThrows(IllegalStateException.class, lifecycle::startRouteDecisionDelivery);
        assertThrows(IllegalStateException.class, () -> lifecycle.startBatchEnqueue(107L));

        RequestLifecycleSnapshot snapshot = lifecycle.snapshot();
        assertEquals(RequestLifecycleState.TIMED_OUT, snapshot.state());
        assertEquals(DeliveryClaimKind.NONE, snapshot.deliveryClaimKind());
        assertEquals(0L, snapshot.batchId());
        assertFalse(lifecycle.hasDeliveryClaim());
    }

    @Test
    void batchEnqueueTimestampRequiresBatchClaimAndIsFirstWriteWins() throws Exception {
        RequestLifecycle lifecycle = new RequestLifecycle("8");
        assertThrows(IllegalStateException.class, lifecycle::markBatchEnqueueStarted);

        lifecycle.startRouteDecisionDelivery();
        assertThrows(IllegalStateException.class, lifecycle::markBatchEnqueueStarted);

        RequestLifecycle batchLifecycle = new RequestLifecycle("81");
        batchLifecycle.startBatchEnqueue(108L);
        batchLifecycle.markBatchEnqueueStarted();
        long firstTimestamp = batchLifecycle.getBatchEnqueueStartedAtMs();
        TimeUnit.MILLISECONDS.sleep(2);
        batchLifecycle.markBatchEnqueueStarted();

        assertEquals(firstTimestamp, batchLifecycle.getBatchEnqueueStartedAtMs());
    }

    @Test
    void concurrentIncompatibleDeliveryClaimsHaveOneWinner() throws Exception {
        RequestLifecycle lifecycle = new RequestLifecycle("9");
        CountDownLatch start = new CountDownLatch(1);
        AtomicInteger successes = new AtomicInteger();
        AtomicInteger rejected = new AtomicInteger();
        ExecutorService executor = Executors.newFixedThreadPool(2);
        try {
            Future<?> batch = executor.submit(() -> attempt(start,
                    () -> lifecycle.startBatchEnqueue(109L), successes, rejected));
            Future<?> route = executor.submit(() -> attempt(start,
                    lifecycle::startRouteDecisionDelivery, successes, rejected));

            start.countDown();
            batch.get(5, TimeUnit.SECONDS);
            route.get(5, TimeUnit.SECONDS);
        } finally {
            executor.shutdownNow();
        }

        RequestLifecycleSnapshot snapshot = lifecycle.snapshot();
        assertEquals(1, successes.get());
        assertEquals(1, rejected.get());
        assertTrue(snapshot.deliveryClaimKind().isClaimed());
        assertEquals(snapshot.deliveryClaimKind() == DeliveryClaimKind.BATCH_ENQUEUE ? 109L : 0L,
                snapshot.batchId());
    }

    @Test
    void timeoutRaceCannotProduceAClaimWithoutDeliveryStateWinningFirst() throws Exception {
        RequestLifecycle lifecycle = new RequestLifecycle("10");
        CountDownLatch start = new CountDownLatch(1);
        ExecutorService executor = Executors.newFixedThreadPool(2);
        try {
            Future<?> delivery = executor.submit(() -> {
                await(start);
                try {
                    lifecycle.startRouteDecisionDelivery();
                } catch (IllegalStateException expectedWhenTimeoutWins) {
                    // The timeout owns the terminal state and no claim is published.
                }
            });
            Future<?> timeout = executor.submit(() -> {
                await(start);
                lifecycle.timeout("deadline race");
            });

            start.countDown();
            delivery.get(5, TimeUnit.SECONDS);
            timeout.get(5, TimeUnit.SECONDS);
        } finally {
            executor.shutdownNow();
        }

        RequestLifecycleSnapshot snapshot = lifecycle.snapshot();
        assertEquals(RequestLifecycleState.TIMED_OUT, snapshot.state());
        assertEquals(0L, snapshot.batchId());
        if (snapshot.deliveryClaimKind() == DeliveryClaimKind.NONE) {
            assertFalse(lifecycle.hasDeliveryClaim());
        } else {
            assertEquals(DeliveryClaimKind.ROUTE_DECISION, snapshot.deliveryClaimKind());
            assertTrue(lifecycle.hasDeliveryClaim());
        }
    }

    private static void attempt(CountDownLatch start,
                                Runnable action,
                                AtomicInteger successes,
                                AtomicInteger rejected) {
        await(start);
        try {
            action.run();
            successes.incrementAndGet();
        } catch (IllegalStateException expected) {
            rejected.incrementAndGet();
        }
    }

    private static void await(CountDownLatch latch) {
        try {
            if (!latch.await(5, TimeUnit.SECONDS)) {
                throw new AssertionError("timed out waiting for test latch");
            }
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
            throw new AssertionError("test interrupted", interrupted);
        }
    }

}
