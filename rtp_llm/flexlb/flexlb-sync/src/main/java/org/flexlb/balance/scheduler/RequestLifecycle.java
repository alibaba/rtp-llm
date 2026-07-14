package org.flexlb.balance.scheduler;

import java.util.EnumSet;
import java.util.Map;

/**
 * Serialized request lifecycle. All mutations are synchronized so dispatch,
 * cancellation, timeout and worker-status callbacks observe one transition order.
 */
final class RequestLifecycle {

    private static final Map<RequestLifecycleState, EnumSet<RequestLifecycleState>> ALLOWED = Map.of(
            RequestLifecycleState.QUEUED, EnumSet.of(
                    RequestLifecycleState.DISPATCHING,
                    RequestLifecycleState.CANCELLED,
                    RequestLifecycleState.TIMED_OUT,
                    RequestLifecycleState.FAILED),
            RequestLifecycleState.DISPATCHING, EnumSet.of(
                    RequestLifecycleState.ACKNOWLEDGED,
                    RequestLifecycleState.CANCEL_REQUESTED,
                    RequestLifecycleState.TIMED_OUT,
                    RequestLifecycleState.FAILED,
                    RequestLifecycleState.COMPLETED),
            RequestLifecycleState.ACKNOWLEDGED, EnumSet.of(
                    RequestLifecycleState.CANCEL_REQUESTED,
                    RequestLifecycleState.TIMED_OUT,
                    RequestLifecycleState.FAILED,
                    RequestLifecycleState.COMPLETED),
            RequestLifecycleState.CANCEL_REQUESTED, EnumSet.of(
                    RequestLifecycleState.CANCELLED,
                    RequestLifecycleState.TIMED_OUT),
            RequestLifecycleState.CANCELLED, EnumSet.noneOf(RequestLifecycleState.class),
            RequestLifecycleState.TIMED_OUT, EnumSet.noneOf(RequestLifecycleState.class),
            RequestLifecycleState.FAILED, EnumSet.noneOf(RequestLifecycleState.class),
            RequestLifecycleState.COMPLETED, EnumSet.noneOf(RequestLifecycleState.class));

    private final long requestId;
    private final long createdAtMs;
    private RequestLifecycleState state = RequestLifecycleState.QUEUED;
    private long updatedAtMs;
    private String detail = "queued";
    private long batchId;

    RequestLifecycle(long requestId) {
        this.requestId = requestId;
        this.createdAtMs = System.currentTimeMillis();
        this.updatedAtMs = createdAtMs;
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

    synchronized RequestLifecycleSnapshot acknowledge() {
        if (state == RequestLifecycleState.CANCEL_REQUESTED || state.isTerminal()) {
            return snapshot();
        }
        return transition(RequestLifecycleState.ACKNOWLEDGED, "engine acknowledged batch");
    }

    synchronized RequestLifecycleSnapshot requestCancel(CancelReason reason) {
        if (state.isTerminal() || state == RequestLifecycleState.CANCEL_REQUESTED) {
            return snapshot();
        }
        if (reason == CancelReason.DEADLINE_EXCEEDED) {
            return transition(RequestLifecycleState.TIMED_OUT, "schedule deadline exceeded");
        }
        RequestLifecycleState next = state == RequestLifecycleState.DISPATCHING
                || state == RequestLifecycleState.ACKNOWLEDGED
                ? RequestLifecycleState.CANCEL_REQUESTED
                : RequestLifecycleState.CANCELLED;
        return transition(next, "cancelled by client");
    }

    synchronized RequestLifecycleSnapshot finishCancellation() {
        if (state == RequestLifecycleState.CANCEL_REQUESTED) {
            return transition(RequestLifecycleState.CANCELLED, "engine cancellation reconciled");
        }
        return snapshot();
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
        if (state == RequestLifecycleState.CANCEL_REQUESTED) {
            return transition(RequestLifecycleState.CANCELLED, "cancel completed while dispatch failed");
        }
        return transition(RequestLifecycleState.FAILED, message);
    }

    synchronized RequestLifecycleSnapshot complete(String message) {
        if (state.isTerminal()) {
            return snapshot();
        }
        if (state == RequestLifecycleState.CANCEL_REQUESTED) {
            return transition(RequestLifecycleState.CANCELLED, "cancel won completion race");
        }
        return transition(RequestLifecycleState.COMPLETED, message);
    }

    synchronized boolean isCancellationRequested() {
        return state == RequestLifecycleState.CANCEL_REQUESTED || state == RequestLifecycleState.CANCELLED;
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
        if (!ALLOWED.get(state).contains(next)) {
            throw new IllegalStateException("invalid request lifecycle transition " + state + " -> " + next);
        }
        state = next;
        detail = message == null ? "" : message;
        updatedAtMs = System.currentTimeMillis();
        return snapshot();
    }
}
