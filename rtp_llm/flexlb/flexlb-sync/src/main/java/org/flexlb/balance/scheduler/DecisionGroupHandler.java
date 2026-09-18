package org.flexlb.balance.scheduler;

import java.util.List;

/**
 * Receives request-group decisions from a worker's scheduling queue.
 * <p>
 * Reports expiration, grouping, and delivery failure for admitted requests:
 * <ul>
 *   <li>{@link #onExpired} — head item's deadline has passed, must be dropped</li>
 *   <li>{@link #onDecisionGroupReady} — a logical group is ready for its configured delivery mode</li>
 *   <li>{@link #onDeliveryFailure} — an admitted item could not complete delivery</li>
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
     * Called when an admitted request cannot complete delivery, including queued
     * work drained during shutdown. Releases any ownership acquired by the request.
     * Immediate admission rejection is returned by {@link WorkerBatcher#tryOffer}.
     *
     * @param item  the queued, staged, or claimed item whose delivery failed
     * @param error the failure that prevented delivery from completing
     */
    void onDeliveryFailure(BatchItem item, Throwable error);
}
