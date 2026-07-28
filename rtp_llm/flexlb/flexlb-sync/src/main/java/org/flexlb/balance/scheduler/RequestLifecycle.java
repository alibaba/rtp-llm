package org.flexlb.balance.scheduler;

/**
 * Serialized request lifecycle. All mutations are synchronized so dispatch,
 * timeout and worker-status callbacks observe one transition order.
 */
final class RequestLifecycle {

    private final long requestId;
    private final long createdAtMs;
    private RequestLifecycleState state;
    private long updatedAtMs;
    private String detail = "queued";
    private long batchId;
    private long dispatchedAtMs;

    RequestLifecycle(long requestId) {
        this(requestId, RequestLifecycleState.QUEUED);
    }

    RequestLifecycle(long requestId, RequestLifecycleState initialState) {
        this.requestId = requestId;
        this.createdAtMs = System.currentTimeMillis();
        this.updatedAtMs = createdAtMs;
        this.state = initialState;
        this.detail = initialState == RequestLifecycleState.ROUTING ? "routing" : "queued";
    }

    synchronized void queued() {
        transition(RequestLifecycleState.QUEUED, "queued");
    }

    synchronized void startDispatch(long assignedBatchId) {
        if (assignedBatchId <= 0) {
            throw new IllegalArgumentException("batchId must be positive");
        }
        if (batchId != 0 && batchId != assignedBatchId) {
            throw new IllegalStateException("request already belongs to batch " + batchId);
        }
        if (batchId == 0) {
            batchId = assignedBatchId;
        }
        transition(RequestLifecycleState.DISPATCHING, "dispatch started");
    }

    /**
     * Record the timestamp when the request is dispatched to the engine via gRPC.
     * Used together with {@link #getDispatchedAtMs()} to compute dispatch-to-ACK latency.
     */
    synchronized void markDispatched() {
        dispatchedAtMs = System.currentTimeMillis();
    }

    /**
     * @return the dispatch timestamp set by {@link #markDispatched()}, or 0 if not yet dispatched.
     */
    synchronized long getDispatchedAtMs() {
        return dispatchedAtMs;
    }

    synchronized RequestLifecycleSnapshot acknowledge() {
        if (state.isTerminal()) {
            return snapshot();
        }
        return transition(RequestLifecycleState.ACKNOWLEDGED, "engine acknowledged batch");
    }

    synchronized RequestLifecycleSnapshot timeout(String message) {
        if (state.isTerminal()) {
            return snapshot();
        }
        return transition(RequestLifecycleState.TIMED_OUT, message);
    }

    synchronized RequestLifecycleSnapshot fail(String message) {
        if (state.isTerminal()) {
            return snapshot();
        }
        return transition(RequestLifecycleState.FAILED, message);
    }

    synchronized RequestLifecycleSnapshot complete(String message) {
        if (state.isTerminal()) {
            return snapshot();
        }
        return transition(RequestLifecycleState.COMPLETED, message);
    }

    synchronized boolean isTerminal() {
        return state.isTerminal();
    }

    synchronized RequestLifecycleSnapshot snapshot() {
        return new RequestLifecycleSnapshot(requestId, state, batchId,
                createdAtMs, updatedAtMs, detail);
    }

    private RequestLifecycleSnapshot transition(RequestLifecycleState next, String message) {
        if (state == next) {
            return snapshot();
        }
        if (!allows(state, next)) {
            throw new IllegalStateException("invalid request lifecycle transition " + state + " -> " + next);
        }
        state = next;
        detail = message == null ? "" : message;
        updatedAtMs = System.currentTimeMillis();
        return snapshot();
    }

    private static boolean allows(RequestLifecycleState from, RequestLifecycleState to) {
        return switch (from) {
            case ROUTING -> to == RequestLifecycleState.QUEUED || isAbort(to);
            case QUEUED -> to == RequestLifecycleState.DISPATCHING || isAbort(to);
            case DISPATCHING -> to == RequestLifecycleState.ACKNOWLEDGED || to.isTerminal();
            case ACKNOWLEDGED -> to.isTerminal();
            default -> false;
        };
    }

    private static boolean isAbort(RequestLifecycleState state) {
        return state == RequestLifecycleState.TIMED_OUT
                || state == RequestLifecycleState.FAILED;
    }
}
