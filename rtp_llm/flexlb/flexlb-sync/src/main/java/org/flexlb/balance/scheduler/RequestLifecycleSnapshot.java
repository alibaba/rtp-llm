package org.flexlb.balance.scheduler;

/** Immutable lifecycle view returned by cancellation and reconciliation APIs. */
public record RequestLifecycleSnapshot(long requestId,
                                       RequestLifecycleState state,
                                       long batchId,
                                       long createdAtMs,
                                       long updatedAtMs,
                                       String detail) {
}
