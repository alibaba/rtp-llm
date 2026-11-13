package org.flexlb.balance.endpoint;

/** Generation-local pressure from committed Engine waiting-stream observations. */
public final class EngineQueueAdmission {
    public enum State { UNKNOWN, AVAILABLE, CONGESTED }

    private final long threshold;
    private final long durationMs;
    private long aboveSinceMs = -1L;
    private volatile State state = State.UNKNOWN;
    private volatile long waiting = -1L;

    EngineQueueAdmission(long threshold, long durationMs) {
        this.threshold = threshold;
        this.durationMs = durationMs;
    }

    // Endpoint status ownership serializes observations; readers consume one published state.
    void observe(long waiting, long nowMs) {
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
    boolean dispatchBlocked(long unobservedRequests) { return dispatchBlocked(unobservedRequests, threshold); }

    boolean dispatchBlocked(long unobservedRequests, long watermark) {
        long observedWaiting = waiting;
        return observedWaiting >= 0L && observedWaiting > watermark - unobservedRequests;
    }

    public State state() { return state; }
}
