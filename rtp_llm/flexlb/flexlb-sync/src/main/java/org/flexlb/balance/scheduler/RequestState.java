package org.flexlb.balance.scheduler;

import java.util.Objects;

import static com.google.common.base.Preconditions.checkArgument;

/** Immutable public view of one canonical request generation. */
public record RequestState(
        long requestId,
        Phase state,
        DeliveryClaimKind deliveryClaimKind,
        long batchId,
        long createdAtMs,
        long updatedAtMs,
        String detail) {

    public RequestState {
        Objects.requireNonNull(state, "state");
        Objects.requireNonNull(deliveryClaimKind, "deliveryClaimKind");
        Objects.requireNonNull(detail, "detail");
        checkArgument(deliveryClaimKind != DeliveryClaimKind.BATCH_ENQUEUE || batchId > 0L,
                "batch enqueue delivery requires a positive batchId");
        checkArgument(deliveryClaimKind == DeliveryClaimKind.BATCH_ENQUEUE || batchId == 0L,
                "only batch enqueue delivery may carry a batchId");
    }

    /** Public reporting projection; internal control facts never escape as executable commands. */
    public static RequestState from(ContextResult.Snapshot snapshot) {
        if (snapshot.terminalRecord() != null) { return snapshot.terminalRecord(); }
        if (snapshot.key() == null) { return null; }
        var delivery = snapshot.delivery();
        var cancellation = snapshot.cancellation();
        boolean cancellingRemote = snapshot.outcome() != null
                && snapshot.outcome().kind() == ContextResult.EndKind.FRONTEND && cancellation != null
                && snapshot.phase() == ContextResult.Phase.ENDING && delivery != null
                && delivery.identity().kind() == DeliveryClaimKind.BATCH_ENQUEUE
                && !snapshot.deliveryReleasable() && !snapshot.close().expired();
        Phase phase = cancellingRemote ? Phase.CANCEL_REQUESTED
                : snapshot.outcome() != null ? snapshot.outcome().outcome().phase()
                : cancellation != null || snapshot.preemption() != null && snapshot.preemption().cancelAcknowledged()
                ? Phase.CANCEL_REQUESTED : delivery != null && delivery.acknowledged() ? Phase.ACKNOWLEDGED
                : delivery != null ? Phase.DISPATCHING : Phase.QUEUED;
        String detail = snapshot.outcome() != null ? snapshot.outcome().detail()
                : cancellation != null ? cancellation.detail()
                : snapshot.engine().visibilityOverdue() || delivery != null && delivery.send() == ContextResult.SendState.UNKNOWN
                        && !snapshot.engine().decodeAccepted()
                        && !(snapshot.engine().prefillObserved() && snapshot.engine().prefillCompletedAtMs() == 0L)
                        ? "SUSPECTED_LOST: request not visible before its deadline"
                : delivery != null && delivery.acknowledged() ? "delivery acknowledged" : "queued";
        return new RequestState(snapshot.key().requestId(), phase,
                delivery == null ? DeliveryClaimKind.NONE : delivery.identity().kind(),
                delivery == null ? 0L : delivery.identity().batchId(), snapshot.createdAtMs(), snapshot.updatedAtMs(),
                detail == null ? "" : detail);
    }

    /** An expected batch ID of zero accepts any batch, including route delivery. */
    public boolean matchesBatch(long expectedBatchId) {
        return expectedBatchId == 0L || batchId == expectedBatchId;
    }

    public enum Phase {
        QUEUED,
        DISPATCHING,
        ACKNOWLEDGED,
        CANCEL_REQUESTED,
        CANCELLED,
        TIMED_OUT,
        FAILED,
        COMPLETED;

        public boolean isTerminal() {
            return this == CANCELLED || this == TIMED_OUT
                    || this == FAILED || this == COMPLETED;
        }
    }
}
