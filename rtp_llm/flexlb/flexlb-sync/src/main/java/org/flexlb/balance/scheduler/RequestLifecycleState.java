package org.flexlb.balance.scheduler;

/** Scheduler-owned request states. Terminal states never transition again. */
public enum RequestLifecycleState {
    QUEUED,
    /** A delivery claim is active; the external outcome is not yet confirmed. */
    DISPATCHING,
    /** Delivery is confirmed: EnqueueBatch was acknowledged or a route decision was published. */
    ACKNOWLEDGED,
    CANCEL_REQUESTED,
    CANCELLED,
    TIMED_OUT,
    FAILED,
    COMPLETED;

    public boolean isTerminal() {
        return this == CANCELLED || this == TIMED_OUT || this == FAILED || this == COMPLETED;
    }
}
