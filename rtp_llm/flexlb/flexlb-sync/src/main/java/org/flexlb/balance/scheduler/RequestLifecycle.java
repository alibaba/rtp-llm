package org.flexlb.balance.scheduler;

import java.util.EnumSet;
import java.util.Map;

/**
 * Serialized request lifecycle. All mutations are synchronized so delivery,
 * timeout and worker-status callbacks observe one transition order.
 */
final class RequestLifecycle {

    private static final Map<RequestLifecycleState, EnumSet<RequestLifecycleState>> ALLOWED = Map.of(
            RequestLifecycleState.QUEUED, EnumSet.of(
                    RequestLifecycleState.DISPATCHING,
                    RequestLifecycleState.CANCEL_REQUESTED,
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
                    RequestLifecycleState.TIMED_OUT,
                    RequestLifecycleState.FAILED,
                    RequestLifecycleState.COMPLETED),
            RequestLifecycleState.CANCELLED, EnumSet.noneOf(RequestLifecycleState.class),
            RequestLifecycleState.TIMED_OUT, EnumSet.noneOf(RequestLifecycleState.class),
            RequestLifecycleState.FAILED, EnumSet.noneOf(RequestLifecycleState.class),
            RequestLifecycleState.COMPLETED, EnumSet.noneOf(RequestLifecycleState.class));

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

    synchronized boolean isTerminal() {
        return state.isTerminal();
    }

    /**
     * Whether delivery has crossed the point where an external caller or
     * engine may observe the request. A claim is acquired before either an
     * EnqueueBatch invocation or a route decision becomes visible.
     */
    synchronized boolean hasDeliveryClaim() {
        return deliveryClaimKind.isClaimed();
    }

    synchronized RequestLifecycleSnapshot snapshot() {
        return new RequestLifecycleSnapshot(requestId, state, deliveryClaimKind, batchId,
                createdAtMs, updatedAtMs, detail);
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
        if (state != next && !ALLOWED.get(state).contains(next)) {
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
        return snapshot();
    }
}
