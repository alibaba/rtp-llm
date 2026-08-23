package org.flexlb.balance.scheduler;

/**
 * Receives terminal queue events and capacity-admitted decision groups.
 * <p>
 * Each method corresponds to an ownership transition in the queue run loop:
 * <ul>
 *   <li>{@link #onExpired} — head item's deadline has passed, must be dropped</li>
 *   <li>{@link #onDecisionGroupAdmitted} — every admitted member owns its
 *       required hard capacity</li>
 *   <li>{@link #onOfferFailure} — a new item could not be enqueued (batcher stopped or queue full)</li>
 *   <li>{@link #onDeliveryFailure} — an admitted request cannot complete delivery</li>
 * </ul>
 */
public interface DecisionGroupHandler {

    /**
     * Called when the head request has expired.
     * The scheduler settles its endpoint and request ownership, then fails the future.
     */
    void onExpired(BatchItem head);

    /**
     * Called only for the final ordered prefix whose mode-specific hard
     * capacity has already been reserved: endpoint slots and, for BATCH, one
     * accepted local dispatcher task. The handler performs no capacity check.
     *
     * <p>Every live member must be claimed and resolved before this method
     * returns. A normal return with unresolved members, or an exception, is an
     * invariant failure: every member still owned by the batcher is terminated
     * through {@link #onDeliveryFailure} and is never retried.
     */
    void onDecisionGroupAdmitted(
            AdmittedDecisionGroup group,
            DecisionGroupMetadata metadata);

    /**
     * Called when {@link WorkerBatcher#offer} fails — batcher is stopped or queue is full.
     *
     * @param item  the item that could not be enqueued
     * @param error the reason the item could not enter the queue
     */
    void onOfferFailure(BatchItem item, Throwable error);

    /**
     * Called after an item has left the scheduling queue but delivery cannot
     * complete. Unlike {@link #onOfferFailure}, this is terminal settlement:
     * release acquired ownership, fail the future, and never requeue the item.
     *
     * @param item  the admitted or callback-owned item whose delivery failed
     * @param error the failure that prevented delivery from completing
     */
    void onDeliveryFailure(BatchItem item, Throwable error);
}
