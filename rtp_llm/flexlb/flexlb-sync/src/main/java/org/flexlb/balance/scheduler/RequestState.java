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
