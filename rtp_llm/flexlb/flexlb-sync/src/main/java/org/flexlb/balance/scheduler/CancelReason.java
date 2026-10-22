package org.flexlb.balance.scheduler;

/** First cause asking the scheduling master to cancel a request. */
public enum CancelReason {
    CLIENT_CANCELLED("request cancelled by client"),
    DEADLINE_EXCEEDED("request deadline exceeded");

    private final String message;

    CancelReason(String message) {
        this.message = message;
    }

    public String getMessage() {
        return message;
    }
}
