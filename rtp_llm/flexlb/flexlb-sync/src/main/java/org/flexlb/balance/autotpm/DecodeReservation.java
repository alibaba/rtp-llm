package org.flexlb.balance.autotpm;

/**
 * Tracks a single request's decode-side reservation state.
 *
 * <p>Created when a request is admitted and routed to a decode endpoint.
 * The {@link DecodeAdmissionTracker} maintains a per-endpoint map of
 * these reservations to track capacity usage and find eviction candidates.
 *
 * <p>Fields are immutable except {@link #state}, which progresses through
 * the {@link DecodeAdmissionState} lifecycle as the Engine reports status.
 */
public final class DecodeReservation {

    private final long requestId;
    private final int priority;
    private final long deadlineMs;
    private final long kvTokensRequired;
    private final String decodeEndpointKey;
    private final long createdAtMs;
    private volatile DecodeAdmissionState state;

    public DecodeReservation(long requestId, int priority, long deadlineMs,
                             long kvTokensRequired, String decodeEndpointKey) {
        this(requestId, priority, deadlineMs, kvTokensRequired,
                decodeEndpointKey, System.currentTimeMillis(),
                DecodeAdmissionState.RESERVED_NOT_ACCEPTED);
    }

    public DecodeReservation(long requestId, int priority, long deadlineMs,
                             long kvTokensRequired, String decodeEndpointKey,
                             long createdAtMs, DecodeAdmissionState state) {
        this.requestId = requestId;
        this.priority = priority;
        this.deadlineMs = deadlineMs;
        this.kvTokensRequired = kvTokensRequired;
        this.decodeEndpointKey = decodeEndpointKey;
        this.createdAtMs = createdAtMs;
        this.state = state;
    }

    public long requestId() {
        return requestId;
    }

    public int priority() {
        return priority;
    }

    public long deadlineMs() {
        return deadlineMs;
    }

    /** KV cache tokens this reservation requires (prompt seqLen). */
    public long kvTokensRequired() {
        return kvTokensRequired;
    }

    /** Endpoint key (ip:port) this reservation is on. */
    public String decodeEndpointKey() {
        return decodeEndpointKey;
    }

    public long createdAtMs() {
        return createdAtMs;
    }

    public DecodeAdmissionState state() {
        return state;
    }

    public void setState(DecodeAdmissionState state) {
        this.state = state;
    }

    /**
     * @return {@code true} if this reservation is eligible for eviction
     *         (state is not RUNNING).
     */
    public boolean isEvictable() {
        return state.isEvictable();
    }

    @Override
    public String toString() {
        return "DecodeReservation{reqId=" + requestId + ", pri=" + priority
                + ", kv=" + kvTokensRequired + ", state=" + state
                + ", ep=" + decodeEndpointKey + "}";
    }
}
