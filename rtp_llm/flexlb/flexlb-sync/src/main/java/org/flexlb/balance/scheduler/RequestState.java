package org.flexlb.balance.scheduler;

/**
 * Serialized lifecycle transition kernel. In production the canonical
 * RequestSlot inherits these fields, so every synchronized method locks that
 * exact slot rather than a delegated lifecycle object.
 */
public class RequestState {

    public enum Phase {
        QUEUED,
        DISPATCHING,
        ACKNOWLEDGED,
        CANCEL_REQUESTED,
        CANCELLED,
        TIMED_OUT,
        FAILED,
        COMPLETED;

        boolean canTransitionTo(Phase next) {
            if (this == next) {
                return true;
            }
            return switch (this) {
                case QUEUED -> next == DISPATCHING
                        || next == CANCEL_REQUESTED
                        || next == TIMED_OUT
                        || next == FAILED;
                case DISPATCHING -> next == ACKNOWLEDGED
                        || next == CANCEL_REQUESTED
                        || next == TIMED_OUT
                        || next == FAILED
                        || next == COMPLETED;
                case ACKNOWLEDGED -> next == CANCEL_REQUESTED
                        || next == TIMED_OUT
                        || next == FAILED
                        || next == COMPLETED;
                case CANCEL_REQUESTED -> next == CANCELLED
                        || next == TIMED_OUT
                        || next == FAILED
                        || next == COMPLETED;
                case CANCELLED, TIMED_OUT, FAILED, COMPLETED -> false;
            };
        }

        public boolean isTerminal() {
            return this == CANCELLED || this == TIMED_OUT
                    || this == FAILED || this == COMPLETED;
        }
    }

    public record Snapshot(
            long requestId,
            Phase state,
            DeliveryClaimKind deliveryClaimKind,
            long batchId,
            long createdAtMs,
            long updatedAtMs,
            String detail) {
        public Snapshot {
            java.util.Objects.requireNonNull(state, "state");
            java.util.Objects.requireNonNull(
                    deliveryClaimKind, "deliveryClaimKind");
            java.util.Objects.requireNonNull(detail, "detail");
            if (deliveryClaimKind == DeliveryClaimKind.BATCH_ENQUEUE
                    && batchId <= 0) {
                throw new IllegalArgumentException(
                        "batch enqueue delivery requires a positive batchId");
            }
            if (deliveryClaimKind != DeliveryClaimKind.BATCH_ENQUEUE
                    && batchId != 0) {
                throw new IllegalArgumentException(
                        "only batch enqueue delivery may carry a batchId");
            }
        }
    }

    private final long requestId;
    private final long createdAtMs;
    private Phase state = Phase.QUEUED;
    private long updatedAtMs;
    private String detail = "queued";
    private DeliveryClaimKind deliveryClaimKind = DeliveryClaimKind.NONE;
    private long batchId;
    private long batchEnqueueStartedAtMs;

    RequestState(long requestId) {
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
        ensureTransitionAllowed(Phase.DISPATCHING);
        if (deliveryClaimKind == DeliveryClaimKind.NONE) {
            deliveryClaimKind = DeliveryClaimKind.BATCH_ENQUEUE;
            batchId = assignedBatchId;
        }
        transition(Phase.DISPATCHING, "batch enqueue started");
    }

    /**
     * Acquire the point-of-no-return claim before publishing a route decision
     * to the caller. Route-decision deliveries deliberately carry no batch
     * id: accounting and terminal reconciliation remain request scoped.
     */
    synchronized void startRouteDecisionDelivery() {
        requireCompatibleDelivery(DeliveryClaimKind.ROUTE_DECISION, 0);
        ensureTransitionAllowed(Phase.DISPATCHING);
        if (deliveryClaimKind == DeliveryClaimKind.NONE) {
            deliveryClaimKind = DeliveryClaimKind.ROUTE_DECISION;
        }
        transition(Phase.DISPATCHING, "route decision delivery started");
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

    synchronized Snapshot markDeliveryConfirmed() {
        if (state.isTerminal() || state == Phase.CANCEL_REQUESTED) {
            return snapshot();
        }
        String confirmationDetail = switch (deliveryClaimKind) {
            case BATCH_ENQUEUE -> "batch enqueue acknowledged";
            case ROUTE_DECISION -> "route decision delivered";
            case NONE -> throw new IllegalStateException(
                    "cannot confirm delivery without a delivery claim");
        };
        return transition(Phase.ACKNOWLEDGED, confirmationDetail);
    }

    synchronized Snapshot timeout(String message) {
        if (state.isTerminal()) {
            return snapshot();
        }
        return transition(Phase.TIMED_OUT, message);
    }

    synchronized Snapshot fail(String message) {
        if (state.isTerminal()) {
            return snapshot();
        }
        return transition(Phase.FAILED, message);
    }

    synchronized Snapshot complete(String message) {
        if (state.isTerminal()) {
            return snapshot();
        }
        return transition(Phase.COMPLETED, message);
    }

    synchronized Snapshot requestCancel(String message) {
        if (state.isTerminal()) {
            return snapshot();
        }
        return transition(Phase.CANCEL_REQUESTED, message);
    }

    synchronized Snapshot cancel(String message) {
        if (state.isTerminal()) {
            return snapshot();
        }
        if (state != Phase.CANCEL_REQUESTED) {
            transition(Phase.CANCEL_REQUESTED, message);
        }
        return transition(Phase.CANCELLED, message);
    }

    synchronized Snapshot snapshot() {
        return new Snapshot(requestId, state, deliveryClaimKind, batchId,
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

    private void ensureTransitionAllowed(Phase next) {
        if (!state.canTransitionTo(next)) {
            throw new IllegalStateException("invalid request lifecycle transition " + state + " -> " + next);
        }
    }

    private Snapshot transition(Phase next, String message) {
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
