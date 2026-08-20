package org.flexlb.balance.scheduler;

import java.util.List;

/**
 * Receives request-group decisions from a worker's scheduling queue.
 * <p>
 * Each method corresponds to a decision made during the queue's run loop:
 * <ul>
 *   <li>{@link #onExpired} — head item's deadline has passed, must be dropped</li>
 *   <li>{@link #onDecisionGroupReady} — a logical group is ready for its configured delivery mode</li>
 *   <li>{@link #onOfferFailure} — a new item could not be enqueued (batcher stopped or queue full)</li>
 *   <li>{@link #onDeliveryFailure} — a staged item could not complete delivery</li>
 * </ul>
 */
public interface DecisionGroupHandler {

    /**
     * Called when the head request has expired.
     * The scheduler removes it from inflight, rolls back the route, and fails the future.
     */
    void onExpired(BatchItem head);

    /**
     * Called when the grouping policy has released a logical request group.
     * A normal return consumes members not explicitly resolved through the
     * scheduler's pending-delivery API; throwing restores members whose
     * delivery ownership has not been claimed.
     */
    void onDecisionGroupReady(List<BatchItem> items, DecisionGroupMetadata metadata);

    /**
     * Called when {@link WorkerBatcher#offer} fails — batcher is stopped or queue is full.
     *
     * @param item  the item that could not be enqueued
     * @param error non-null if the batcher is stopped; null if the queue is full
     */
    void onOfferFailure(BatchItem item, Throwable error);

    /**
     * Called after an item has left the scheduling queue but delivery cannot
     * complete. Unlike {@link #onOfferFailure}, the handler must treat this as
     * a delivery failure and release any ownership already acquired for it.
     *
     * @param item  the staged or claimed item whose delivery failed
     * @param error the failure that prevented delivery from completing
     */
    void onDeliveryFailure(BatchItem item, Throwable error);
}
