package org.flexlb.balance.scheduler;

/**
 * Atomic state enum for inflight request lifecycle.
 *
 * <p>Replaces the former {@code AtomicBoolean terminated} +
 * {@code volatile TerminalReason} pair with a single
 * {@code AtomicReference<InflightState>}, eliminating the
 * intermediate window between the two writes.
 */
public enum InflightState {
    RUNNING,
    COMPLETED,
    FAILED,
    CANCELLED,
    TIMED_OUT;

    public boolean isTerminal() {
        return this != RUNNING;
    }
}
