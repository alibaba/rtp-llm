package org.flexlb.balance.scheduler;

import java.util.List;

/**
 * Callback interface for {@link WorkerBatcher} to notify the scheduler of batching decisions.
 * <p>
 * Each method corresponds to a decision the batcher makes during its run loop:
 * <ul>
 *   <li>{@link #onExpired} — head item's deadline has passed, must be dropped</li>
 *   <li>{@link #onBatchReady} — a batch has been assembled and is ready for gRPC dispatch</li>
 *   <li>{@link #onOfferFailure} — a new item could not be enqueued (batcher stopped or queue full)</li>
 * </ul>
 */
public interface BatchDecisionHandler {

    /**
     * Called when the head item's SLO deadline has expired.
     * The scheduler removes it from inflight, rolls back the route, and fails the future.
     */
    void onExpired(BatchItem head);

    /**
     * Called when the batcher has assembled a batch ready for dispatch.
     */
    void onBatchReady(List<BatchItem> items, DispatchMeta meta);

    /**
     * Called when {@link WorkerBatcher#offer} fails — batcher is stopped or queue is full.
     *
     * @param item  the item that could not be enqueued
     * @param error non-null if the batcher is stopped; null if the queue is full
     */
    void onOfferFailure(BatchItem item, Throwable error);

    /**
     * Called when the head item's queue deadline has been exceeded.
     *
     * <p>Unlike {@link #onExpired}, this callback signals the scheduler that
     * the request should be returned for retry/fail decision rather than
     * silently dropped. Used by {@link PriorityDeadlineBatcherAlgorithm}.
     *
     * <p>Default implementation delegates to {@link #onExpired} for backward
     * compatibility with algorithms that do not distinguish deadline expiry
     * from SLO expiry.
     */
    default void onDeadlineExceeded(BatchItem item) {
        onExpired(item);
    }
}
