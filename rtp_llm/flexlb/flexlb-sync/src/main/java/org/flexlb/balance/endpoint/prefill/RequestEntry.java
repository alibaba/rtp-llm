package org.flexlb.balance.endpoint.prefill;

import org.flexlb.balance.scheduler.ScheduledRequest;

/**
 * 一个请求在 Prefill 上唯一的可变记录。由 PrefillState 的锁保护。
 */
final class RequestEntry {
    final String requestId;
    QueueMembership queueMembership;
    ScheduledRequest activeItem;
    SingleRequestProgress singleRequestProgress;
    BatchProgress batchProgress;
    /**
     * Canonical callback identity after admission commits.
     */
    ScheduledRequest committedItem;
    Reservation reservation;

    RequestEntry(ScheduledRequest item) {
        this.requestId = item.requestId();
        this.queueMembership = QueueMembership.WAITING;
        this.activeItem = item;
    }

    boolean isActive() {
        return activeItem != null;
    }

    boolean activeIdentity(ScheduledRequest item) {
        return activeItem != null && activeItem == item && queueMembership != QueueMembership.STOP_DETACHED;
    }

    void commitIndividual(PrefillState.RouteReservation lease, long nowMs) {
        if (!isActive() || reservation != lease) {
            throw new IllegalStateException("request is not an ACTIVE route request_id=" + requestId);
        }
        singleRequestProgress = new SingleRequestProgress(lease.predictedWorkMs(), nowMs);
        committedItem = activeItem;
        activeItem = null;
        queueMembership = QueueMembership.UNINDEXED;
    }

    void commitBatch(BatchProgress work) {
        if (!isActive()) {
            throw new IllegalStateException("request is not an ACTIVE batch member request_id=" + requestId);
        }
        batchProgress = work;
        committedItem = activeItem;
        reservation = null;
        activeItem = null;
        queueMembership = QueueMembership.UNINDEXED;
    }
}
