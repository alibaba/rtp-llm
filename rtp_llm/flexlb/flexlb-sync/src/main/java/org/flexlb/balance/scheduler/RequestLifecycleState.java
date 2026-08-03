package org.flexlb.balance.scheduler;

/**
 * Scheduler-owned request states. Terminal states never transition again.
 *
 * @deprecated v2 调度器重构：被 InflightItem/AbstractScheduler/InflightStore 替代。
 */
@Deprecated
public enum RequestLifecycleState {
    QUEUED,
    DISPATCHING,
    ACKNOWLEDGED,
    TIMED_OUT,
    FAILED,
    COMPLETED;

    public boolean isTerminal() {
        return this == TIMED_OUT || this == FAILED || this == COMPLETED;
    }
}
