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
            RequestState.Phase initial,
            LifecycleEvent event,
            ExpectedTransition expected) {
        RequestState lifecycle = lifecycleIn(initial);
        RequestState.Snapshot before = lifecycle.snapshot();

        if (expected.rejected()) {
            assertThrows(IllegalStateException.class,
                    () -> event.apply(lifecycle));
        } else {
            assertEquals(expected.state(), event.apply(lifecycle).state());
        }

        RequestState.Snapshot after = lifecycle.snapshot();
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
        RequestState lifecycle = new RequestState(1L);

        assertFalse(hasDeliveryClaim(lifecycle));
        lifecycle.startBatchEnqueue(101L);
        assertTrue(hasDeliveryClaim(lifecycle));
        assertEquals(RequestState.Phase.DISPATCHING, lifecycle.snapshot().state());
        assertEquals(DeliveryClaimKind.BATCH_ENQUEUE, lifecycle.snapshot().deliveryClaimKind());
        assertEquals(RequestState.Phase.ACKNOWLEDGED,
                lifecycle.markDeliveryConfirmed().state());
        RequestState.Snapshot completed = lifecycle.complete("decode finished");

        assertEquals(RequestState.Phase.COMPLETED, completed.state());
        assertEquals(101L, completed.batchId());
        assertTrue(completed.state().isTerminal());
    }

    @Test
    void routeDecisionDeliveryAcquiresRequestScopedDeliveryClaim() {
        RequestState lifecycle = new RequestState(4L);

        lifecycle.startRouteDecisionDelivery();
        RequestState.Snapshot acknowledged = lifecycle.markDeliveryConfirmed();

        assertTrue(hasDeliveryClaim(lifecycle));
        assertEquals(RequestState.Phase.ACKNOWLEDGED, acknowledged.state());
        assertEquals(DeliveryClaimKind.ROUTE_DECISION, acknowledged.deliveryClaimKind());
        assertEquals(0L, acknowledged.batchId());
        assertEquals("route decision delivered", acknowledged.detail());
    }

    @Test
    void deliveryClaimKindAndBatchOwnershipCannotChangeAfterClaim() {
        RequestState batchLifecycle = new RequestState(5L);
        batchLifecycle.startBatchEnqueue(105L);
        batchLifecycle.startBatchEnqueue(105L);

        assertThrows(IllegalStateException.class, batchLifecycle::startRouteDecisionDelivery);
        assertThrows(IllegalStateException.class, () -> batchLifecycle.startBatchEnqueue(106L));
        assertEquals(DeliveryClaimKind.BATCH_ENQUEUE, batchLifecycle.snapshot().deliveryClaimKind());
        assertEquals(105L, batchLifecycle.snapshot().batchId());

        RequestState routeLifecycle = new RequestState(6L);
        routeLifecycle.startRouteDecisionDelivery();
        routeLifecycle.startRouteDecisionDelivery();

        assertThrows(IllegalStateException.class,
                () -> routeLifecycle.startBatchEnqueue(106L));
        assertEquals(DeliveryClaimKind.ROUTE_DECISION, routeLifecycle.snapshot().deliveryClaimKind());
        assertEquals(0L, routeLifecycle.snapshot().batchId());
    }

    @Test
    void rejectedLateDeliveryDoesNotLeavePartialClaim() {
        RequestState lifecycle = new RequestState(7L);
        lifecycle.timeout("expired in queue");

        assertThrows(IllegalStateException.class, lifecycle::startRouteDecisionDelivery);
        assertThrows(IllegalStateException.class, () -> lifecycle.startBatchEnqueue(107L));

        RequestState.Snapshot snapshot = lifecycle.snapshot();
        assertEquals(RequestState.Phase.TIMED_OUT, snapshot.state());
        assertEquals(DeliveryClaimKind.NONE, snapshot.deliveryClaimKind());
        assertEquals(0L, snapshot.batchId());
        assertFalse(hasDeliveryClaim(lifecycle));
    }

    @Test
    void batchEnqueueTimestampRequiresBatchClaimAndIsFirstWriteWins() throws Exception {
        RequestState lifecycle = new RequestState(8L);
        assertThrows(IllegalStateException.class, lifecycle::markBatchEnqueueStarted);

        lifecycle.startRouteDecisionDelivery();
        assertThrows(IllegalStateException.class, lifecycle::markBatchEnqueueStarted);

        RequestState batchLifecycle = new RequestState(81L);
        batchLifecycle.startBatchEnqueue(108L);
        batchLifecycle.markBatchEnqueueStarted();
        long firstTimestamp = batchLifecycle.getBatchEnqueueStartedAtMs();
        TimeUnit.MILLISECONDS.sleep(2);
        batchLifecycle.markBatchEnqueueStarted();

        assertEquals(firstTimestamp, batchLifecycle.getBatchEnqueueStartedAtMs());
    }

    @Test
    void concurrentIncompatibleDeliveryClaimsHaveOneWinner() throws Exception {
        RequestState lifecycle = new RequestState(9L);
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

        RequestState.Snapshot snapshot = lifecycle.snapshot();
        assertEquals(1, successes.get());
        assertEquals(1, rejected.get());
        assertTrue(snapshot.deliveryClaimKind().isClaimed());
        assertEquals(snapshot.deliveryClaimKind() == DeliveryClaimKind.BATCH_ENQUEUE ? 109L : 0L,
                snapshot.batchId());
    }

    @Test
    void timeoutRaceCannotProduceAClaimWithoutDeliveryStateWinningFirst() throws Exception {
        RequestState lifecycle = new RequestState(10L);
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

        RequestState.Snapshot snapshot = lifecycle.snapshot();
        assertEquals(RequestState.Phase.TIMED_OUT, snapshot.state());
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

    private static boolean hasDeliveryClaim(RequestState lifecycle) {
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
        return Arrays.stream(RequestState.Phase.values())
                .flatMap(initial -> Arrays.stream(LifecycleEvent.values())
                        .map(event -> Arguments.of(
                                initial, event, expected(initial, event))));
    }

    private static ExpectedTransition expected(
            RequestState.Phase initial,
            LifecycleEvent event) {
        if (initial.isTerminal()) {
            return ExpectedTransition.accepted(initial);
        }
        return switch (initial) {
            case QUEUED -> switch (event) {
                case CONFIRM_DELIVERY, COMPLETE -> ExpectedTransition.rejected(initial);
                case REQUEST_CANCEL -> ExpectedTransition.accepted(
                        RequestState.Phase.CANCEL_REQUESTED);
                case CANCEL -> ExpectedTransition.accepted(RequestState.Phase.CANCELLED);
                case TIMEOUT -> ExpectedTransition.accepted(RequestState.Phase.TIMED_OUT);
                case FAIL -> ExpectedTransition.accepted(RequestState.Phase.FAILED);
            };
            case DISPATCHING -> switch (event) {
                case CONFIRM_DELIVERY -> ExpectedTransition.accepted(
                        RequestState.Phase.ACKNOWLEDGED);
                case REQUEST_CANCEL -> ExpectedTransition.accepted(
                        RequestState.Phase.CANCEL_REQUESTED);
                case CANCEL -> ExpectedTransition.accepted(RequestState.Phase.CANCELLED);
                case TIMEOUT -> ExpectedTransition.accepted(RequestState.Phase.TIMED_OUT);
                case FAIL -> ExpectedTransition.accepted(RequestState.Phase.FAILED);
                case COMPLETE -> ExpectedTransition.accepted(RequestState.Phase.COMPLETED);
            };
            case ACKNOWLEDGED -> switch (event) {
                case CONFIRM_DELIVERY -> ExpectedTransition.accepted(
                        RequestState.Phase.ACKNOWLEDGED);
                case REQUEST_CANCEL -> ExpectedTransition.accepted(
                        RequestState.Phase.CANCEL_REQUESTED);
                case CANCEL -> ExpectedTransition.accepted(RequestState.Phase.CANCELLED);
                case TIMEOUT -> ExpectedTransition.accepted(RequestState.Phase.TIMED_OUT);
                case FAIL -> ExpectedTransition.accepted(RequestState.Phase.FAILED);
                case COMPLETE -> ExpectedTransition.accepted(RequestState.Phase.COMPLETED);
            };
            case CANCEL_REQUESTED -> switch (event) {
                case CONFIRM_DELIVERY, REQUEST_CANCEL -> ExpectedTransition.accepted(
                        RequestState.Phase.CANCEL_REQUESTED);
                case CANCEL -> ExpectedTransition.accepted(RequestState.Phase.CANCELLED);
                case TIMEOUT -> ExpectedTransition.accepted(RequestState.Phase.TIMED_OUT);
                case FAIL -> ExpectedTransition.accepted(RequestState.Phase.FAILED);
                case COMPLETE -> ExpectedTransition.accepted(RequestState.Phase.COMPLETED);
            };
            case CANCELLED, TIMED_OUT, FAILED, COMPLETED ->
                    throw new AssertionError("terminal state handled above");
        };
    }

    private static RequestState lifecycleIn(RequestState.Phase state) {
        RequestState lifecycle = new RequestState(1000L + state.ordinal());
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
            RequestState.Snapshot apply(RequestState lifecycle) {
                return lifecycle.markDeliveryConfirmed();
            }
        },
        REQUEST_CANCEL {
            @Override
            RequestState.Snapshot apply(RequestState lifecycle) {
                return lifecycle.requestCancel("event cancel requested");
            }
        },
        CANCEL {
            @Override
            RequestState.Snapshot apply(RequestState lifecycle) {
                return lifecycle.cancel("event cancelled");
            }
        },
        TIMEOUT {
            @Override
            RequestState.Snapshot apply(RequestState lifecycle) {
                return lifecycle.timeout("event timeout");
            }
        },
        FAIL {
            @Override
            RequestState.Snapshot apply(RequestState lifecycle) {
                return lifecycle.fail("event failure");
            }
        },
        COMPLETE {
            @Override
            RequestState.Snapshot apply(RequestState lifecycle) {
                return lifecycle.complete("event completion");
            }
        };

        abstract RequestState.Snapshot apply(RequestState lifecycle);
    }

    private record ExpectedTransition(
            RequestState.Phase state,
            boolean rejected) {

        private static ExpectedTransition accepted(RequestState.Phase state) {
            return new ExpectedTransition(state, false);
        }

        private static ExpectedTransition rejected(RequestState.Phase state) {
            return new ExpectedTransition(state, true);
        }
    }

}
