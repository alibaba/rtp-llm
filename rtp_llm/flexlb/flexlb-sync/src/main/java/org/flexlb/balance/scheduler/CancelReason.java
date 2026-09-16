package org.flexlb.balance.scheduler;

/** First cause asking the scheduling master to cancel a request. */
public enum CancelReason {
    CLIENT_CANCELLED,
    DEADLINE_EXCEEDED
}
