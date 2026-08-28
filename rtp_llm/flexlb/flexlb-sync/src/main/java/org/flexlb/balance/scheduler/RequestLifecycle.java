package org.flexlb.balance.scheduler;

/**
 * Serialized lifecycle transition kernel. In production the canonical
 * RequestSlot inherits these fields, so every synchronized method locks that
 * exact slot rather than a delegated lifecycle object.
 */
class RequestLifecycle {

    private final long requestId;
    private final long createdAtMs;
    private RequestLifecycleState state = RequestLifecycleState.QUEUED;
    private long updatedAtMs;
    private String detail = "queued";
    private DeliveryClaimKind deliveryClaimKind = DeliveryClaimKind.NONE;
    private long batchId;
    private long batchEnqueueStartedAtMs;

    RequestLifecycle(long requestId) {
        this.requestId = requestId;
        this.createdAtMs = System.currentTimeMillis();
        this.updatedAtMs = createdAtMs;
    }

    /**
     * Acquire the point-of-no-return claim for an EnqueueBatch delivery.
     *
     * <p>The claim is idempotent for the same batch and immutable afterwards.</p>
     */
    synchronized void startBatchEnqueue(long assignedBatchId) {
        if (assignedBatchId <= 0) {
            throw new IllegalArgumentException("batchId must be positive");
        }
        requireCompatibleDelivery(DeliveryClaimKind.BATCH_ENQUEUE, assignedBatchId);
        ensureTransitionAllowed(RequestLifecycleState.DISPATCHING);
        if (deliveryClaimKind == DeliveryClaimKind.NONE) {
            deliveryClaimKind = DeliveryClaimKind.BATCH_ENQUEUE;
            batchId = assignedBatchId;
        }
        transition(RequestLifecycleState.DISPATCHING, "batch enqueue started");
    }

    /**
     * Acquire the point-of-no-return claim before publishing a route decision
     * to the caller. Route-decision deliveries deliberately carry no batch
     * id: accounting and terminal reconciliation remain request scoped.
     */
    synchronized void startRouteDecisionDelivery() {
        requireCompatibleDelivery(DeliveryClaimKind.ROUTE_DECISION, 0);
        ensureTransitionAllowed(RequestLifecycleState.DISPATCHING);
        if (deliveryClaimKind == DeliveryClaimKind.NONE) {
            deliveryClaimKind = DeliveryClaimKind.ROUTE_DECISION;
        }
        transition(RequestLifecycleState.DISPATCHING, "route decision delivery started");
    }

    /**
     * Record the first EnqueueBatch send timestamp. Route-decision delivery
     * deliberately does not use this batch transport metric.
     */
    synchronized void markBatchEnqueueStarted() {
        if (deliveryClaimKind != DeliveryClaimKind.BATCH_ENQUEUE) {
            throw new IllegalStateException(
                    "batch enqueue timestamp requires a batch delivery claim");
        }
        if (batchEnqueueStartedAtMs == 0) {
            batchEnqueueStartedAtMs = System.currentTimeMillis();
            afterLifecycleMutation();
        }
    }

    /**
     * @return the timestamp set by {@link #markBatchEnqueueStarted()}, or 0 if not yet sent.
     */
    synchronized long getBatchEnqueueStartedAtMs() {
        return batchEnqueueStartedAtMs;
    }

    synchronized RequestLifecycleSnapshot markDeliveryConfirmed() {
        if (state.isTerminal() || state == RequestLifecycleState.CANCEL_REQUESTED) {
            return snapshot();
        }
        String confirmationDetail = switch (deliveryClaimKind) {
            case BATCH_ENQUEUE -> "batch enqueue acknowledged";
            case ROUTE_DECISION -> "route decision delivered";
            case NONE -> throw new IllegalStateException(
                    "cannot confirm delivery without a delivery claim");
        };
        return transition(RequestLifecycleState.ACKNOWLEDGED, confirmationDetail);
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

    synchronized RequestLifecycleSnapshot requestCancel(String message) {
        if (state.isTerminal()) {
            return snapshot();
        }
        return transition(RequestLifecycleState.CANCEL_REQUESTED, message);
    }

    synchronized RequestLifecycleSnapshot cancel(String message) {
        if (state.isTerminal()) {
            return snapshot();
        }
        if (state != RequestLifecycleState.CANCEL_REQUESTED) {
            transition(RequestLifecycleState.CANCEL_REQUESTED, message);
        }
        return transition(RequestLifecycleState.CANCELLED, message);
    }

    synchronized RequestLifecycleSnapshot snapshot() {
        return new RequestLifecycleSnapshot(requestId, state, deliveryClaimKind, batchId,
                createdAtMs, updatedAtMs, detail);
    }

    /** Aggregate hook invoked while the exact lifecycle lock is still held. */
    void afterLifecycleMutation() {
        // Standalone lifecycle instances have no aggregate invariants.
    }

    private void requireCompatibleDelivery(DeliveryClaimKind requestedKind, long requestedBatchId) {
        if (deliveryClaimKind == DeliveryClaimKind.NONE) {
            return;
        }
        if (deliveryClaimKind != requestedKind) {
            throw new IllegalStateException(
                    "request already has a " + deliveryClaimKind + " delivery claim");
        }
        if (deliveryClaimKind == DeliveryClaimKind.BATCH_ENQUEUE && batchId != requestedBatchId) {
            throw new IllegalStateException("request already belongs to batch " + batchId);
        }
    }

    private void ensureTransitionAllowed(RequestLifecycleState next) {
        if (!state.canTransitionTo(next)) {
            throw new IllegalStateException("invalid request lifecycle transition " + state + " -> " + next);
        }
    }

    private RequestLifecycleSnapshot transition(RequestLifecycleState next, String message) {
        if (state == next) {
            return snapshot();
        }
        ensureTransitionAllowed(next);
        state = next;
        detail = message == null ? "" : message;
        updatedAtMs = System.currentTimeMillis();
        afterLifecycleMutation();
        return snapshot();
    }
}
