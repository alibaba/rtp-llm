package org.flexlb.balance.scheduler;

import java.util.Objects;

/** Immutable lifecycle view returned by request-state and reconciliation APIs. */
public record RequestLifecycleSnapshot(long requestId,
                                       RequestLifecycleState state,
                                       DeliveryClaimKind deliveryClaimKind,
                                       long batchId,
                                       long createdAtMs,
                                       long updatedAtMs,
                                       String detail) {

    public RequestLifecycleSnapshot {
        Objects.requireNonNull(state, "state");
        Objects.requireNonNull(deliveryClaimKind, "deliveryClaimKind");
        Objects.requireNonNull(detail, "detail");
        if (deliveryClaimKind == DeliveryClaimKind.BATCH_ENQUEUE && batchId <= 0) {
            throw new IllegalArgumentException(
                    "batch enqueue delivery requires a positive batchId");
        }
        if (deliveryClaimKind != DeliveryClaimKind.BATCH_ENQUEUE && batchId != 0) {
            throw new IllegalArgumentException(
                    "only batch enqueue delivery may carry a batchId");
        }
    }

    /**
     * Compatibility constructor for callers that predate delivery-aware
     * snapshots. A positive batch id implies a batch enqueue claim.
     */
    public RequestLifecycleSnapshot(long requestId,
                                    RequestLifecycleState state,
                                    long batchId,
                                    long createdAtMs,
                                    long updatedAtMs,
                                    String detail) {
        this(requestId, state,
                batchId > 0 ? DeliveryClaimKind.BATCH_ENQUEUE : DeliveryClaimKind.NONE,
                batchId, createdAtMs, updatedAtMs, detail);
    }
}
