package org.flexlb.balance.strategy;

/** Execution-feedback capacity for the outstanding uncached work ledger. */
public final class PrefillComputeCapacity {
    private final long serviceWindowMs;
    private volatile double tokensPerMs;

    public PrefillComputeCapacity(long serviceWindowMs) {
        if (serviceWindowMs <= 0) { throw new IllegalArgumentException("positive service window required"); }
        this.serviceWindowMs = serviceWindowMs;
    }

    public synchronized void observe(long computeTokens, long executionMs) {
        if (computeTokens <= 0 || executionMs <= 0) { return; }
        double sample = (double) computeTokens / executionMs;
        double weight = Math.min(1.0, (double) executionMs / serviceWindowMs);
        tokensPerMs = tokensPerMs == 0 ? sample : tokensPerMs + weight * (sample - tokensPerMs);
    }

    /** Zero means no positive execution sample is available yet. */
    public long estimatedMs(double computeTokens) {
        double rate = tokensPerMs;
        return rate == 0 || computeTokens <= 0 ? 0 : (long) Math.ceil(computeTokens / rate);
    }
}
