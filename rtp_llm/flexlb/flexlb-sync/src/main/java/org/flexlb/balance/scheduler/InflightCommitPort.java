package org.flexlb.balance.scheduler;

import org.flexlb.balance.admission.AdmissionMutation;

/** Narrow exact-slot commit capability used by route placement. */
interface InflightCommitPort {

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

    /**
     * Bind and publish one exact request generation.
     *
     * <p>The caller must retain that generation's admission mutation until
     * this method returns. The mutation defers cancellation and endpoint
     * terminal facts while the slot and queue sides are resolved separately.
     */
    boolean commitInflight(
            BatchItem item,
            boolean priorityAdmission,
            AdmissionMutation exactMutation,
            ActivePublication publication);
}
