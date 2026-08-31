package org.flexlb.balance.scheduler;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Cancel-path and terminal-idempotency contracts for {@link RequestState}.
 *
 * <p>{@code RequestLifecycleTest} already covers the single-step reducer matrix
 * (every state × every event). This suite covers the compositions it does not:
 * the two-step {@code requestCancel → cancel} detail propagation, the three
 * ordinary terminals reachable from {@code CANCEL_REQUESTED}, the
 * {@code markDeliveryConfirmed} no-op while a cancel is pending, terminal
 * absorbing behavior across every reducer, and delivery-claim retention through
 * cancellation. These are the request-lifecycle exception/cancel links.
 */
@DisplayName("RequestState cancel-path and terminal idempotency")
class RequestLifecycleCancelPathTest {

    private static RequestState lifecycle() {
        return new RequestState(1L);
    }

    // ═══════════════════════════════════════════════════════════════════════
    @Nested
    @DisplayName("Cancellation")
    class Cancellation {

        @Test
        void cancelFromQueuedEndsCancelledWithDetail() {
            RequestState rl = lifecycle();
            RequestState.Snapshot snap = rl.cancel("user aborted");
            assertEquals(RequestState.Phase.CANCELLED, snap.state());
            assertEquals("user aborted", snap.detail());
        }

        @Test
        void requestCancelThenCancelPropagatesLatestDetail() {
            RequestState rl = lifecycle();
            RequestState.Snapshot requested = rl.requestCancel("frontend requested");
            assertEquals(RequestState.Phase.CANCEL_REQUESTED, requested.state());
            assertEquals("frontend requested", requested.detail());

            RequestState.Snapshot cancelled = rl.cancel("engine acked cancel");
            assertEquals(RequestState.Phase.CANCELLED, cancelled.state());
            assertEquals("engine acked cancel", cancelled.detail());
        }

        @Test
        void cancelRetainsAnExistingBatchDeliveryClaim() {
            RequestState rl = lifecycle();
            rl.startBatchEnqueue(7L); // DISPATCHING + BATCH_ENQUEUE, batchId 7
            RequestState.Snapshot snap = rl.cancel("x");
            assertEquals(RequestState.Phase.CANCELLED, snap.state());
            assertEquals(DeliveryClaimKind.BATCH_ENQUEUE, snap.deliveryClaimKind());
            assertEquals(7L, snap.batchId(),
                    "cancellation must not drop the batch ownership record");
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    @Nested
    @DisplayName("Ordinary terminals from CANCEL_REQUESTED")
    class TerminalFromCancelRequested {

        @Test
        void failFromCancelRequestedIsAllowed() {
            RequestState rl = lifecycle();
            rl.requestCancel("c");
            RequestState.Snapshot snap = rl.fail("engine error");
            assertEquals(RequestState.Phase.FAILED, snap.state());
            assertEquals("engine error", snap.detail());
        }

        @Test
        void completeFromCancelRequestedIsAllowed() {
            // A completion can still win over a pending cancel (cancel raced a
            // successful delivery).
            RequestState rl = lifecycle();
            rl.requestCancel("c");
            RequestState.Snapshot snap = rl.complete("delivered before cancel");
            assertEquals(RequestState.Phase.COMPLETED, snap.state());
        }

        @Test
        void timeoutFromCancelRequestedIsAllowed() {
            RequestState rl = lifecycle();
            rl.requestCancel("c");
            RequestState.Snapshot snap = rl.timeout("slo expired");
            assertEquals(RequestState.Phase.TIMED_OUT, snap.state());
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    @Nested
    @DisplayName("markDeliveryConfirmed while cancel is pending")
    class ConfirmWhileCancelPending {

        @Test
        void confirmInCancelRequestedIsANoOpAndKeepsTheClaim() {
            RequestState rl = lifecycle();
            rl.startBatchEnqueue(5L);       // DISPATCHING + BATCH_ENQUEUE
            rl.requestCancel("cancel pending");

            RequestState.Snapshot snap = rl.markDeliveryConfirmed();
            // Confirming a delivery for a request already asked to cancel must
            // not resurrect it to ACKNOWLEDGED.
            assertEquals(RequestState.Phase.CANCEL_REQUESTED, snap.state());
            assertEquals(DeliveryClaimKind.BATCH_ENQUEUE, snap.deliveryClaimKind());
            assertEquals(5L, snap.batchId());
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    @Nested
    @DisplayName("Terminal is absorbing across every reducer")
    class TerminalAbsorbing {

        @Test
        void completedStateIgnoresAllLaterReducersAndKeepsDetail() {
            RequestState rl = lifecycle();
            rl.startBatchEnqueue(3L);
            rl.markDeliveryConfirmed();
            RequestState.Snapshot completed = rl.complete("done");
            assertEquals(RequestState.Phase.COMPLETED, completed.state());

            // Every later reducer returns the frozen COMPLETED snapshot.
            assertEquals(RequestState.Phase.COMPLETED, rl.fail("late fail").state());
            assertEquals(RequestState.Phase.COMPLETED, rl.timeout("late timeout").state());
            assertEquals(RequestState.Phase.COMPLETED, rl.requestCancel("late cancel").state());
            assertEquals(RequestState.Phase.COMPLETED, rl.cancel("late cancel").state());
            assertEquals("done", rl.snapshot().detail(),
                    "a terminal detail is never overwritten by a later reducer");
        }

        @Test
        void cancelledStateIsAbsorbing() {
            RequestState rl = lifecycle();
            rl.cancel("aborted");
            assertEquals(RequestState.Phase.CANCELLED, rl.fail("x").state());
            assertEquals(RequestState.Phase.CANCELLED, rl.complete("x").state());
            assertEquals("aborted", rl.snapshot().detail());
        }
    }
}
