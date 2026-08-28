package org.flexlb.balance.scheduler;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.Arrays;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class RequestLifecycleTest {

    @ParameterizedTest(name = "{0} + {1} -> {2}")
    @MethodSource("lifecycleTransitionCases")
    void everyReducerHasAnExplicitResultFromEveryLifecycleState(
            RequestLifecycleState initial,
            LifecycleEvent event,
            ExpectedTransition expected) {
        RequestLifecycle lifecycle = lifecycleIn(initial);
        RequestLifecycleSnapshot before = lifecycle.snapshot();

        if (expected.rejected()) {
            assertThrows(IllegalStateException.class,
                    () -> event.apply(lifecycle));
        } else {
            assertEquals(expected.state(), event.apply(lifecycle).state());
        }

        RequestLifecycleSnapshot after = lifecycle.snapshot();
        assertEquals(expected.state(), after.state());
        if (initial.isTerminal()) {
            assertEquals(before.detail(), after.detail(),
                    "a late reducer must not overwrite the terminal cause");
            assertEquals(before.deliveryClaimKind(), after.deliveryClaimKind());
            assertEquals(before.batchId(), after.batchId());
        }
    }

    @Test
    void normalBatchDeliveryTransitionsToCompleted() {
        RequestLifecycle lifecycle = new RequestLifecycle(1L);

        assertFalse(hasDeliveryClaim(lifecycle));
        lifecycle.startBatchEnqueue(101L);
        assertTrue(hasDeliveryClaim(lifecycle));
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
    void routeDecisionDeliveryAcquiresRequestScopedDeliveryClaim() {
        RequestLifecycle lifecycle = new RequestLifecycle(4L);

        lifecycle.startRouteDecisionDelivery();
        RequestLifecycleSnapshot acknowledged = lifecycle.markDeliveryConfirmed();

        assertTrue(hasDeliveryClaim(lifecycle));
        assertEquals(RequestLifecycleState.ACKNOWLEDGED, acknowledged.state());
        assertEquals(DeliveryClaimKind.ROUTE_DECISION, acknowledged.deliveryClaimKind());
        assertEquals(0L, acknowledged.batchId());
        assertEquals("route decision delivered", acknowledged.detail());
    }

    @Test
    void deliveryClaimKindAndBatchOwnershipCannotChangeAfterClaim() {
        RequestLifecycle batchLifecycle = new RequestLifecycle(5L);
        batchLifecycle.startBatchEnqueue(105L);
        batchLifecycle.startBatchEnqueue(105L);

        assertThrows(IllegalStateException.class, batchLifecycle::startRouteDecisionDelivery);
        assertThrows(IllegalStateException.class, () -> batchLifecycle.startBatchEnqueue(106L));
        assertEquals(DeliveryClaimKind.BATCH_ENQUEUE, batchLifecycle.snapshot().deliveryClaimKind());
        assertEquals(105L, batchLifecycle.snapshot().batchId());

        RequestLifecycle routeLifecycle = new RequestLifecycle(6L);
        routeLifecycle.startRouteDecisionDelivery();
        routeLifecycle.startRouteDecisionDelivery();

        assertThrows(IllegalStateException.class,
                () -> routeLifecycle.startBatchEnqueue(106L));
        assertEquals(DeliveryClaimKind.ROUTE_DECISION, routeLifecycle.snapshot().deliveryClaimKind());
        assertEquals(0L, routeLifecycle.snapshot().batchId());
    }

    @Test
    void rejectedLateDeliveryDoesNotLeavePartialClaim() {
        RequestLifecycle lifecycle = new RequestLifecycle(7L);
        lifecycle.timeout("expired in queue");

        assertThrows(IllegalStateException.class, lifecycle::startRouteDecisionDelivery);
        assertThrows(IllegalStateException.class, () -> lifecycle.startBatchEnqueue(107L));

        RequestLifecycleSnapshot snapshot = lifecycle.snapshot();
        assertEquals(RequestLifecycleState.TIMED_OUT, snapshot.state());
        assertEquals(DeliveryClaimKind.NONE, snapshot.deliveryClaimKind());
        assertEquals(0L, snapshot.batchId());
        assertFalse(hasDeliveryClaim(lifecycle));
    }

    @Test
    void batchEnqueueTimestampRequiresBatchClaimAndIsFirstWriteWins() throws Exception {
        RequestLifecycle lifecycle = new RequestLifecycle(8L);
        assertThrows(IllegalStateException.class, lifecycle::markBatchEnqueueStarted);

        lifecycle.startRouteDecisionDelivery();
        assertThrows(IllegalStateException.class, lifecycle::markBatchEnqueueStarted);

        RequestLifecycle batchLifecycle = new RequestLifecycle(81L);
        batchLifecycle.startBatchEnqueue(108L);
        batchLifecycle.markBatchEnqueueStarted();
        long firstTimestamp = batchLifecycle.getBatchEnqueueStartedAtMs();
        TimeUnit.MILLISECONDS.sleep(2);
        batchLifecycle.markBatchEnqueueStarted();

        assertEquals(firstTimestamp, batchLifecycle.getBatchEnqueueStartedAtMs());
    }

    @Test
    void concurrentIncompatibleDeliveryClaimsHaveOneWinner() throws Exception {
        RequestLifecycle lifecycle = new RequestLifecycle(9L);
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
        RequestLifecycle lifecycle = new RequestLifecycle(10L);
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
            assertFalse(hasDeliveryClaim(lifecycle));
        } else {
            assertEquals(DeliveryClaimKind.ROUTE_DECISION, snapshot.deliveryClaimKind());
            assertTrue(hasDeliveryClaim(lifecycle));
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

    private static boolean hasDeliveryClaim(RequestLifecycle lifecycle) {
        return lifecycle.snapshot().deliveryClaimKind().isClaimed();
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

    private static Stream<Arguments> lifecycleTransitionCases() {
        return Arrays.stream(RequestLifecycleState.values())
                .flatMap(initial -> Arrays.stream(LifecycleEvent.values())
                        .map(event -> Arguments.of(
                                initial, event, expected(initial, event))));
    }

    private static ExpectedTransition expected(
            RequestLifecycleState initial,
            LifecycleEvent event) {
        if (initial.isTerminal()) {
            return ExpectedTransition.accepted(initial);
        }
        return switch (initial) {
            case QUEUED -> switch (event) {
                case CONFIRM_DELIVERY, COMPLETE -> ExpectedTransition.rejected(initial);
                case REQUEST_CANCEL -> ExpectedTransition.accepted(
                        RequestLifecycleState.CANCEL_REQUESTED);
                case CANCEL -> ExpectedTransition.accepted(RequestLifecycleState.CANCELLED);
                case TIMEOUT -> ExpectedTransition.accepted(RequestLifecycleState.TIMED_OUT);
                case FAIL -> ExpectedTransition.accepted(RequestLifecycleState.FAILED);
            };
            case DISPATCHING -> switch (event) {
                case CONFIRM_DELIVERY -> ExpectedTransition.accepted(
                        RequestLifecycleState.ACKNOWLEDGED);
                case REQUEST_CANCEL -> ExpectedTransition.accepted(
                        RequestLifecycleState.CANCEL_REQUESTED);
                case CANCEL -> ExpectedTransition.accepted(RequestLifecycleState.CANCELLED);
                case TIMEOUT -> ExpectedTransition.accepted(RequestLifecycleState.TIMED_OUT);
                case FAIL -> ExpectedTransition.accepted(RequestLifecycleState.FAILED);
                case COMPLETE -> ExpectedTransition.accepted(RequestLifecycleState.COMPLETED);
            };
            case ACKNOWLEDGED -> switch (event) {
                case CONFIRM_DELIVERY -> ExpectedTransition.accepted(
                        RequestLifecycleState.ACKNOWLEDGED);
                case REQUEST_CANCEL -> ExpectedTransition.accepted(
                        RequestLifecycleState.CANCEL_REQUESTED);
                case CANCEL -> ExpectedTransition.accepted(RequestLifecycleState.CANCELLED);
                case TIMEOUT -> ExpectedTransition.accepted(RequestLifecycleState.TIMED_OUT);
                case FAIL -> ExpectedTransition.accepted(RequestLifecycleState.FAILED);
                case COMPLETE -> ExpectedTransition.accepted(RequestLifecycleState.COMPLETED);
            };
            case CANCEL_REQUESTED -> switch (event) {
                case CONFIRM_DELIVERY, REQUEST_CANCEL -> ExpectedTransition.accepted(
                        RequestLifecycleState.CANCEL_REQUESTED);
                case CANCEL -> ExpectedTransition.accepted(RequestLifecycleState.CANCELLED);
                case TIMEOUT -> ExpectedTransition.accepted(RequestLifecycleState.TIMED_OUT);
                case FAIL -> ExpectedTransition.accepted(RequestLifecycleState.FAILED);
                case COMPLETE -> ExpectedTransition.accepted(RequestLifecycleState.COMPLETED);
            };
            case CANCELLED, TIMED_OUT, FAILED, COMPLETED ->
                    throw new AssertionError("terminal state handled above");
        };
    }

    private static RequestLifecycle lifecycleIn(RequestLifecycleState state) {
        RequestLifecycle lifecycle = new RequestLifecycle(1000L + state.ordinal());
        switch (state) {
            case QUEUED -> {
            }
            case DISPATCHING -> lifecycle.startRouteDecisionDelivery();
            case ACKNOWLEDGED -> {
                lifecycle.startRouteDecisionDelivery();
                lifecycle.markDeliveryConfirmed();
            }
            case CANCEL_REQUESTED -> lifecycle.requestCancel("initial cancel requested");
            case CANCELLED -> lifecycle.cancel("initial cancelled");
            case TIMED_OUT -> lifecycle.timeout("initial timeout");
            case FAILED -> lifecycle.fail("initial failure");
            case COMPLETED -> {
                lifecycle.startRouteDecisionDelivery();
                lifecycle.complete("initial completion");
            }
        }
        assertEquals(state, lifecycle.snapshot().state());
        return lifecycle;
    }

    private enum LifecycleEvent {
        CONFIRM_DELIVERY {
            @Override
            RequestLifecycleSnapshot apply(RequestLifecycle lifecycle) {
                return lifecycle.markDeliveryConfirmed();
            }
        },
        REQUEST_CANCEL {
            @Override
            RequestLifecycleSnapshot apply(RequestLifecycle lifecycle) {
                return lifecycle.requestCancel("event cancel requested");
            }
        },
        CANCEL {
            @Override
            RequestLifecycleSnapshot apply(RequestLifecycle lifecycle) {
                return lifecycle.cancel("event cancelled");
            }
        },
        TIMEOUT {
            @Override
            RequestLifecycleSnapshot apply(RequestLifecycle lifecycle) {
                return lifecycle.timeout("event timeout");
            }
        },
        FAIL {
            @Override
            RequestLifecycleSnapshot apply(RequestLifecycle lifecycle) {
                return lifecycle.fail("event failure");
            }
        },
        COMPLETE {
            @Override
            RequestLifecycleSnapshot apply(RequestLifecycle lifecycle) {
                return lifecycle.complete("event completion");
            }
        };

        abstract RequestLifecycleSnapshot apply(RequestLifecycle lifecycle);
    }

    private record ExpectedTransition(
            RequestLifecycleState state,
            boolean rejected) {

        private static ExpectedTransition accepted(RequestLifecycleState state) {
            return new ExpectedTransition(state, false);
        }

        private static ExpectedTransition rejected(RequestLifecycleState state) {
            return new ExpectedTransition(state, true);
        }
    }

}
