package org.flexlb.balance.scheduler;

/** Cause used to distinguish explicit cancellation from deadline expiry. */
public enum CancelReason {
    CLIENT_CANCELLED,
    DEADLINE_EXCEEDED
}
