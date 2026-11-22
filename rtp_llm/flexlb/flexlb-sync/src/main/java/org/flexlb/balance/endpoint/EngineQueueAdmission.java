package org.flexlb.balance.endpoint;

/** Generation-local pressure from committed Engine waiting-stream observations. */
public final class EngineQueueAdmission {
    public enum State { UNKNOWN, AVAILABLE, CONGESTED }

    private final long threshold;
    private final long durationMs;
    private final boolean recoverCostOrderedQueue;
    private boolean draining;
    private long emptySinceMs = -1L;
    private volatile long dispatchHeadroom = Long.MAX_VALUE;
    private long aboveSinceMs = -1L;
    private volatile State state = State.UNKNOWN;
    private volatile long waiting = -1L;
    private long lastObservedMs = Long.MIN_VALUE;

    private long rescueFullSinceMs = -1L;
    private boolean rescuePressure;
    private final long rescueMaxAgeMs;

    boolean freshPressure(boolean source, long nowMs, long observedAfterMs) {
        return lastObservedMs > observedAfterMs && nowMs - lastObservedMs <= rescueMaxAgeMs
                && (source ? rescuePressure : waiting == 0L);
    }

    EngineQueueAdmission(long threshold, long durationMs) {
        this(threshold, durationMs, 1000L);
    }

    EngineQueueAdmission(long threshold, long durationMs, long rescueMaxAgeMs) {
        this(threshold, durationMs, rescueMaxAgeMs, false);
    }

    EngineQueueAdmission(long threshold, long durationMs, long rescueMaxAgeMs, boolean recoverCostOrderedQueue) {
        this.recoverCostOrderedQueue = recoverCostOrderedQueue && threshold > 1L;
        this.rescueMaxAgeMs = rescueMaxAgeMs;
        this.threshold = threshold;
        this.durationMs = durationMs;
    }

    // Endpoint status ownership serializes observations; readers consume one published state.
    void observe(long waiting, long nowMs) {
        boolean stale = lastObservedMs == Long.MIN_VALUE || nowMs - lastObservedMs > rescueMaxAgeMs;
        if (stale || waiting < 0L) {
            draining = false;
            emptySinceMs = -1L;
        }
        if (stale || waiting < 0L || waiting <= threshold - Math.ceilDiv(threshold, 4L)) {
            rescuePressure = false;
            rescueFullSinceMs = -1L;
        }
        if (waiting > 0 && waiting >= threshold) {
            if (rescueFullSinceMs < 0L) { rescueFullSinceMs = nowMs; }
            if (nowMs - rescueFullSinceMs >= durationMs) { rescuePressure = true; }
        } else if (!rescuePressure) {
            rescueFullSinceMs = -1L;
        }
        long dispatchLimit = threshold;
        if (recoverCostOrderedQueue) {
            // A single full FIFO already contains work admitted against older cache observations.
            // Alternating samples on either side of the limit must not restart its drain timer.
            if (waiting >= threshold) {
                draining = true;
            }
            if (draining && waiting == 0L) {
                if (emptySinceMs < 0L) { emptySinceMs = nowMs; }
                if (nowMs - emptySinceMs >= durationMs) { draining = false; }
            } else {
                emptySinceMs = -1L;
            }
            // While the FIFO drains, keep the next request available at the Worker.
            // The Master retains the rest and can reprice it as prefix fills complete.
            if (draining) { dispatchLimit = 1L; }
        }
        dispatchHeadroom = waiting < 0L ? Long.MAX_VALUE : dispatchLimit - waiting;
        lastObservedMs = nowMs;
        this.waiting = waiting;
        if (waiting < 0L) {
            aboveSinceMs = -1L;
            state = State.UNKNOWN;
        } else if (waiting <= threshold) {
            aboveSinceMs = -1L;
            state = State.AVAILABLE;
        } else {
            if (aboveSinceMs < 0L) { aboveSinceMs = nowMs; }
            state = nowMs - aboveSinceMs >= durationMs ? State.CONGESTED : State.AVAILABLE;
        }
    }

    boolean dispatchBlocked() { return dispatchBlocked(0L); }

    /** Reserve waiting headroom until Engine feedback accounts for a local send. */
    boolean dispatchBlocked(long unobservedRequests) {
        return unobservedRequests > dispatchHeadroom;
    }

    public State state() { return state; }
}
