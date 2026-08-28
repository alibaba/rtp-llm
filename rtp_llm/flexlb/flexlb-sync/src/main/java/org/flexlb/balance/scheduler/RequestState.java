package org.flexlb.balance.scheduler;

import java.util.Objects;

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
        if (deliveryClaimKind == DeliveryClaimKind.BATCH_ENQUEUE
                && batchId <= 0L) {
            throw new IllegalArgumentException(
                    "batch enqueue delivery requires a positive batchId");
        }
        if (deliveryClaimKind != DeliveryClaimKind.BATCH_ENQUEUE
                && batchId != 0L) {
            throw new IllegalArgumentException(
                    "only batch enqueue delivery may carry a batchId");
        }
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

        boolean canTransitionTo(Phase next) {
            if (this == next) {
                return true;
            }
            return switch (this) {
                case QUEUED -> next == DISPATCHING
                        || next == CANCEL_REQUESTED
                        || next == TIMED_OUT
                        || next == FAILED;
                case DISPATCHING -> next == ACKNOWLEDGED
                        || next == CANCEL_REQUESTED
                        || next == TIMED_OUT
                        || next == FAILED
                        || next == COMPLETED;
                case ACKNOWLEDGED -> next == CANCEL_REQUESTED
                        || next == TIMED_OUT
                        || next == FAILED
                        || next == COMPLETED;
                case CANCEL_REQUESTED -> next == CANCELLED
                        || next == TIMED_OUT
                        || next == FAILED
                        || next == COMPLETED;
                case CANCELLED, TIMED_OUT, FAILED, COMPLETED -> false;
            };
        }

        public boolean isTerminal() {
            return this == CANCELLED || this == TIMED_OUT
                    || this == FAILED || this == COMPLETED;
        }
    }
}
