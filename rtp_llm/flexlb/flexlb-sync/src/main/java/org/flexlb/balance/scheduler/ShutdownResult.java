package org.flexlb.balance.scheduler;

/** Outcome of one scheduler shutdown request. */
public record ShutdownResult(ShutdownMode requested, ShutdownMode completed, boolean drained) {
    public boolean forced() {
        return requested == ShutdownMode.DRAIN && completed == ShutdownMode.IMMEDIATE;
    }
}
