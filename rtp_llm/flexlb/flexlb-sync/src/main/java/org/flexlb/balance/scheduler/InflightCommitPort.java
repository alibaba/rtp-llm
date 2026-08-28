package org.flexlb.balance.scheduler;

/** Narrow exact-slot commit capability used by route placement. */
interface InflightCommitPort {

    enum RouteCommitResult {
        PUBLISHED,
        REQUEST_CLOSED,
        ACCEPTANCE_LIMIT_REACHED,
        PUBLICATION_REJECTED
    }

    /**
     * Publish the exact item into the ACTIVE endpoint queue.
     *
     * <p>The lifecycle binds the exact item into its slot, then invokes this
     * operation exactly once without holding the request-slot monitor.
     * Returning {@code true} means queue publication crossed its point of no
     * return. Returning {@code false}, or throwing, means no ACTIVE queue
     * ownership remains and all publication-local rollback is complete. An
     * implementation must not return {@code false} or throw after crossing
     * that point of no return.
     */
    @FunctionalInterface
    interface ActivePublication {
        boolean publish();
    }

    /** Bind the Decode acceptance guard in the same transaction as the item. */
    RouteCommitResult commitRoute(
            BatchItem item,
            boolean priorityAdmission,
            int acceptanceLimit,
            long acceptanceTimeoutMs,
            ActivePublication publication);
}
