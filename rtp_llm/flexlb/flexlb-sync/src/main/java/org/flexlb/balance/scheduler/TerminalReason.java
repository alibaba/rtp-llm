package org.flexlb.balance.scheduler;

import java.util.concurrent.CancellationException;

/**
 * Reason why an inflight item transitioned to a terminal state.
 *
 * <p>Used by {@link InflightItem} to record the terminal cause and by
 * {@link InflightStore} TTL eviction to distinguish tombstone age.
 */
public enum TerminalReason {
    CANCELLED,
    FAILED,
    TIMED_OUT,
    COMPLETED;

    /**
     * Convert this reason to an appropriate exception without a cause.
     */
    public RuntimeException toException() {
        return switch (this) {
            case CANCELLED -> new CancellationException("Request cancelled");
            case FAILED -> new RuntimeException("Request failed");
            case TIMED_OUT -> new RuntimeException("Request timed out");
            case COMPLETED -> new IllegalStateException("Should not convert COMPLETED to exception");
        };
    }

    /**
     * Convert this reason to an appropriate exception with a cause chain.
     */
    public RuntimeException toException(Throwable cause) {
        return switch (this) {
            case CANCELLED -> new CancellationException("Request cancelled: " + cause.getMessage());
            case FAILED -> new RuntimeException("Request failed: " + cause.getMessage(), cause);
            case TIMED_OUT -> new RuntimeException("Request timed out: " + cause.getMessage());
            case COMPLETED -> new IllegalStateException("Should not convert COMPLETED to exception");
        };
    }
}
