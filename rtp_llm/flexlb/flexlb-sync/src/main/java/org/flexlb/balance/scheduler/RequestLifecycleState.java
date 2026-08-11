package org.flexlb.balance.scheduler;

/** Scheduler-owned request states. Terminal states never transition again. */
public enum RequestLifecycleState {
    QUEUED,
    DISPATCHING,
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
