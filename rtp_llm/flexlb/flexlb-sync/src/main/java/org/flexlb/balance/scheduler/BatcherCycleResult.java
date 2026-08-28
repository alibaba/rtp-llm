package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.delivery.DeliveryMetadata;

import java.util.List;
import java.util.Objects;

/** Result of one worker decision cycle. */
record BatcherCycleResult(
        Status status,
        List<ScheduledRequest> items,
        DeliveryMetadata metadata,
        ScheduledRequest request,
        CapacityBoundary unavailable,
        long queueVersion,
        long schedulingInputVersion,
        long wakeAtMs,
        SchedulingWaitReason reason) {

    static final BatcherCycleResult NO_ACTION = simple(Status.NO_ACTION);
    static final BatcherCycleResult QUEUE_CHANGED = simple(Status.QUEUE_CHANGED);

    BatcherCycleResult {
        Objects.requireNonNull(status, "status");
        items = List.copyOf(items);
        boolean admitted = status == Status.ADMITTED;
        boolean capacityBlocked = status == Status.CAPACITY_BLOCKED;
        boolean awaiting = status == Status.AWAITING_SCHEDULING_CHANGE;
        if (admitted != (!items.isEmpty() && metadata != null)
                || capacityBlocked != (request != null && unavailable != null)
                || awaiting != (request != null && reason != null)
                || (!admitted && (!items.isEmpty() || metadata != null))
                || (!capacityBlocked && unavailable != null)
                || (!capacityBlocked && !awaiting && request != null)
                || (!awaiting && reason != null)) {
            throw new IllegalArgumentException(
                    "worker-cycle status requires its exact payload");
        }
    }

    static BatcherCycleResult admitted(
            List<ScheduledRequest> items,
            DeliveryMetadata metadata) {
        return new BatcherCycleResult(Status.ADMITTED, items,
                Objects.requireNonNull(metadata, "metadata"),
                null, null, 0L, 0L, 0L, null);
    }

    static BatcherCycleResult capacityBlocked(
            ScheduledRequest item,
            CapacityBoundary unavailable) {
        return new BatcherCycleResult(Status.CAPACITY_BLOCKED, List.of(), null,
                Objects.requireNonNull(item, "item"),
                Objects.requireNonNull(unavailable, "unavailable"),
                0L, 0L, 0L, null);
    }

    static BatcherCycleResult awaitingSchedulingChange(
            ScheduledRequest head,
            long queueVersion,
            long schedulingInputVersion,
            long wakeAtMs,
            SchedulingWaitReason reason) {
        return new BatcherCycleResult(Status.AWAITING_SCHEDULING_CHANGE,
                List.of(), null, Objects.requireNonNull(head, "head"), null,
                queueVersion, schedulingInputVersion, wakeAtMs,
                Objects.requireNonNull(reason, "reason"));
    }

    ScheduledRequest item() {
        return request;
    }

    ScheduledRequest head() {
        return request;
    }

    private static BatcherCycleResult simple(Status status) {
        return new BatcherCycleResult(
                status, List.of(), null, null, null, 0L, 0L, 0L, null);
    }

    enum Status {
        NO_ACTION,
        QUEUE_CHANGED,
        ADMITTED,
        CAPACITY_BLOCKED,
        AWAITING_SCHEDULING_CHANGE
    }

    enum SchedulingWaitReason {
        COLLECTION_WINDOW,
        PREFILL_KV_CAPACITY
    }
}
