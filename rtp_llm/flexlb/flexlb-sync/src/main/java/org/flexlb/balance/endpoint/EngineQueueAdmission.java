package org.flexlb.balance.endpoint;

/** Generation-local pressure from committed Engine waiting-stream observations. */
public final class EngineQueueAdmission {
    public enum State { UNKNOWN, AVAILABLE, CONGESTED }

    private final long threshold;
    private final long durationMs;
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
        this.rescueMaxAgeMs = rescueMaxAgeMs;
        this.threshold = threshold;
        this.durationMs = durationMs;
    }

    // Endpoint status ownership serializes observations; readers consume one published state.
    void observe(long waiting, long nowMs) {
        if (lastObservedMs == Long.MIN_VALUE || nowMs - lastObservedMs > rescueMaxAgeMs
                || waiting <= threshold - Math.ceilDiv(threshold, 4L)) {
            rescuePressure = false;
            rescueFullSinceMs = -1L;
        }
        if (waiting > 0 && waiting >= threshold) {
            if (rescueFullSinceMs < 0L) { rescueFullSinceMs = nowMs; }
            if (nowMs - rescueFullSinceMs >= durationMs) { rescuePressure = true; }
        } else if (!rescuePressure) {
            rescueFullSinceMs = -1L;
        }
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
        long observedWaiting = waiting;
        return observedWaiting >= 0L && observedWaiting > threshold - unobservedRequests;
    }

    public State state() { return state; }
}
