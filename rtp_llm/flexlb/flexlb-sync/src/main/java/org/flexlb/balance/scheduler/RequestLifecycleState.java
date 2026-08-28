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

    boolean canTransitionTo(RequestLifecycleState next) {
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
        return this == CANCELLED || this == TIMED_OUT || this == FAILED || this == COMPLETED;
    }
}
