package org.flexlb.balance.endpoint.prefill;

import org.flexlb.balance.endpoint.prefill.batch.BatchProgress;
import org.flexlb.balance.endpoint.prefill.single.SingleRequestProgress;
import org.flexlb.balance.endpoint.prefill.single.SingleReservation;
import org.flexlb.balance.scheduler.ScheduledRequest;

/**
 * 一个请求在 Prefill 上唯一的可变记录。由 PrefillState 的锁保护。
 */
public final class RequestEntry {
    public final String requestId;
    public QueueMembership queueMembership;
    public ScheduledRequest activeItem;
    public SingleRequestProgress singleRequestProgress;
    public BatchProgress batchProgress;
    /**
     * Canonical callback identity after admission commits.
     */
    public ScheduledRequest committedItem;
    public Reservation reservation;

    public RequestEntry(ScheduledRequest item) {
        this.requestId = item.requestId();
        this.queueMembership = QueueMembership.WAITING;
        this.activeItem = item;
    }

    public boolean isActive() {
        return activeItem != null;
    }

    public boolean activeIdentity(ScheduledRequest item) {
        return activeItem != null && activeItem == item && queueMembership != QueueMembership.STOP_DETACHED;
    }

    public void commitIndividual(SingleReservation lease, long nowMs) {
        if (!isActive() || reservation != lease) {
            throw new IllegalStateException("request is not an ACTIVE route request_id=" + requestId);
        }
        singleRequestProgress = new SingleRequestProgress(lease.predictedWorkMs(), nowMs);
        committedItem = activeItem;
        activeItem = null;
        queueMembership = QueueMembership.UNINDEXED;
    }

    public void commitBatch(BatchProgress work) {
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
