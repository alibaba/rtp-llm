package org.flexlb.balance.scheduler;


/** Immutable lifecycle view returned by request-state and reconciliation APIs. */
public record RequestLifecycleSnapshot(long requestId,
                                       RequestLifecycleState state,
                                       DeliveryClaimKind deliveryClaimKind,
                                       long batchId,
                                       long createdAtMs,
                                       long updatedAtMs,
                                       String detail) {

    public RequestLifecycleSnapshot {
        assert state != null : "missing lifecycle state";
        assert deliveryClaimKind != null : "missing delivery claim kind";
        assert detail != null : "missing lifecycle detail";
        if (deliveryClaimKind == DeliveryClaimKind.BATCH_ENQUEUE && batchId <= 0) {
            throw new IllegalArgumentException(
                    "batch enqueue delivery requires a positive batchId");
        }
        if (deliveryClaimKind != DeliveryClaimKind.BATCH_ENQUEUE && batchId != 0) {
            throw new IllegalArgumentException(
                    "only batch enqueue delivery may carry a batchId");
        }
    }
}
