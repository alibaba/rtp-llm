package org.flexlb.balance.scheduler;

/**
 * Identifies the point-of-no-return claim acquired for a request.
 *
 * <p>The concrete delivery strategy is fixed once per worker. This value only
 * records whether the corresponding external delivery has actually started.</p>
 */
public enum DeliveryClaimKind {
    /** The request is still wholly owned by the scheduler. */
    NONE,

    /** An EnqueueBatch operation may have made the request visible to the engine. */
    BATCH_ENQUEUE,

    /** The route decision may have been returned to the caller. */
    ROUTE_DECISION;

    public boolean isClaimed() {
        return this != NONE;
    }
}
