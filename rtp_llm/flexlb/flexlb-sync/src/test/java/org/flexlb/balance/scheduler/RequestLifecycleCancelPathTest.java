package org.flexlb.balance.scheduler;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Cancel-path and terminal-idempotency contracts for {@link RequestLifecycle}.
 *
 * <p>{@code RequestLifecycleTest} already covers the single-step reducer matrix
 * (every state × every event). This suite covers the compositions it does not:
 * the two-step {@code requestCancel → cancel} detail propagation, the three
 * ordinary terminals reachable from {@code CANCEL_REQUESTED}, the
 * {@code markDeliveryConfirmed} no-op while a cancel is pending, terminal
 * absorbing behavior across every reducer, and delivery-claim retention through
 * cancellation. These are the request-lifecycle exception/cancel links.
 */
@DisplayName("RequestLifecycle cancel-path and terminal idempotency")
class RequestLifecycleCancelPathTest {

    private static RequestLifecycle lifecycle() {
        return new RequestLifecycle(1L);
    }

    // ═══════════════════════════════════════════════════════════════════════
    @Nested
    @DisplayName("Cancellation")
    class Cancellation {

        @Test
        void cancelFromQueuedEndsCancelledWithDetail() {
            RequestLifecycle rl = lifecycle();
            RequestLifecycleSnapshot snap = rl.cancel("user aborted");
            assertEquals(RequestLifecycleState.CANCELLED, snap.state());
            assertEquals("user aborted", snap.detail());
        }

        @Test
        void requestCancelThenCancelPropagatesLatestDetail() {
            RequestLifecycle rl = lifecycle();
            RequestLifecycleSnapshot requested = rl.requestCancel("frontend requested");
            assertEquals(RequestLifecycleState.CANCEL_REQUESTED, requested.state());
            assertEquals("frontend requested", requested.detail());

            RequestLifecycleSnapshot cancelled = rl.cancel("engine acked cancel");
            assertEquals(RequestLifecycleState.CANCELLED, cancelled.state());
            assertEquals("engine acked cancel", cancelled.detail());
        }

        @Test
        void cancelRetainsAnExistingBatchDeliveryClaim() {
            RequestLifecycle rl = lifecycle();
            rl.startBatchEnqueue(7L); // DISPATCHING + BATCH_ENQUEUE, batchId 7
            RequestLifecycleSnapshot snap = rl.cancel("x");
            assertEquals(RequestLifecycleState.CANCELLED, snap.state());
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
            RequestLifecycle rl = lifecycle();
            rl.requestCancel("c");
            RequestLifecycleSnapshot snap = rl.fail("engine error");
            assertEquals(RequestLifecycleState.FAILED, snap.state());
            assertEquals("engine error", snap.detail());
        }

        @Test
        void completeFromCancelRequestedIsAllowed() {
            // A completion can still win over a pending cancel (cancel raced a
            // successful delivery).
            RequestLifecycle rl = lifecycle();
            rl.requestCancel("c");
            RequestLifecycleSnapshot snap = rl.complete("delivered before cancel");
            assertEquals(RequestLifecycleState.COMPLETED, snap.state());
        }

        @Test
        void timeoutFromCancelRequestedIsAllowed() {
            RequestLifecycle rl = lifecycle();
            rl.requestCancel("c");
            RequestLifecycleSnapshot snap = rl.timeout("slo expired");
            assertEquals(RequestLifecycleState.TIMED_OUT, snap.state());
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    @Nested
    @DisplayName("markDeliveryConfirmed while cancel is pending")
    class ConfirmWhileCancelPending {

        @Test
        void confirmInCancelRequestedIsANoOpAndKeepsTheClaim() {
            RequestLifecycle rl = lifecycle();
            rl.startBatchEnqueue(5L);       // DISPATCHING + BATCH_ENQUEUE
            rl.requestCancel("cancel pending");

            RequestLifecycleSnapshot snap = rl.markDeliveryConfirmed();
            // Confirming a delivery for a request already asked to cancel must
            // not resurrect it to ACKNOWLEDGED.
            assertEquals(RequestLifecycleState.CANCEL_REQUESTED, snap.state());
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
            RequestLifecycle rl = lifecycle();
            rl.startBatchEnqueue(3L);
            rl.markDeliveryConfirmed();
            RequestLifecycleSnapshot completed = rl.complete("done");
            assertEquals(RequestLifecycleState.COMPLETED, completed.state());

            // Every later reducer returns the frozen COMPLETED snapshot.
            assertEquals(RequestLifecycleState.COMPLETED, rl.fail("late fail").state());
            assertEquals(RequestLifecycleState.COMPLETED, rl.timeout("late timeout").state());
            assertEquals(RequestLifecycleState.COMPLETED, rl.requestCancel("late cancel").state());
            assertEquals(RequestLifecycleState.COMPLETED, rl.cancel("late cancel").state());
            assertEquals("done", rl.snapshot().detail(),
                    "a terminal detail is never overwritten by a later reducer");
        }

        @Test
        void cancelledStateIsAbsorbing() {
            RequestLifecycle rl = lifecycle();
            rl.cancel("aborted");
            assertEquals(RequestLifecycleState.CANCELLED, rl.fail("x").state());
            assertEquals(RequestLifecycleState.CANCELLED, rl.complete("x").state());
            assertEquals("aborted", rl.snapshot().detail());
        }
    }
}
