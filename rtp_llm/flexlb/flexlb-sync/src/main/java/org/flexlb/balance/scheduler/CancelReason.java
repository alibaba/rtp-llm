package org.flexlb.balance.scheduler;

/** First cause asking the scheduling master to cancel a request. */
public enum CancelReason {
    CLIENT_CANCELLED("request cancelled by client"),
    DEADLINE_EXCEEDED("request deadline exceeded"),
    SHUTDOWN("request cancelled during shutdown"),
    PRIORITY_PREEMPTED("preempted by a higher-priority request");

    private final String message;

    CancelReason(String message) {
        this.message = message;
    }

    public String getMessage() {
        return message;
    }
}
