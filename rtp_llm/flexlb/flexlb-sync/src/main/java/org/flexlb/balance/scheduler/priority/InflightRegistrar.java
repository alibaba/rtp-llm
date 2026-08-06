package org.flexlb.balance.scheduler.priority;

import org.flexlb.balance.scheduler.BatchItem;

/**
 * Registers Auto-TPM admitted requests into the batch scheduler's inflight
 * tracking so that dispatch, completion, TTL cleanup and rollback treat them
 * exactly like legacy-path requests.
 *
 * <p>Implemented by {@code FlexlbBatchScheduler}; expressed as an interface to
 * avoid a circular bean dependency with {@code PriorityAdmissionScheduler}.
 */
public interface InflightRegistrar {

    /**
     * Register the item as inflight.
     *
     * @return false when the request id is already inflight or terminal
     *         (duplicate) — the item was NOT registered
     */
    boolean registerInflight(BatchItem item);

    /** Remove a previously registered item (offer failed, plan aborted). */
    void unregisterInflight(BatchItem item);

    /**
     * Drive an evicted victim to its terminal state (design doc 9.5/17.3):
     * release its decode reservation, complete its future with
     * {@code PRIORITY_PREEMPTED} and tombstone the request id. Reserved for
     * victims the engine has already accepted (contract 5.3). Idempotent —
     * repeated calls (or races with other terminal paths) take effect once.
     */
    void finishPreempted(BatchItem victim, String detail);

    /**
     * {@link #finishPreempted} addressed by request id, for victims whose
     * {@code BatchItem} is not at hand (design doc 11.5). No-op when the id
     * is not inflight — the victim already reached a terminal state;
     * idempotent like {@code finishPreempted}.
     */
    void finishPreemptedById(long requestId, String detail);

    /**
     * Drive a yielded victim — one the engine never saw (prefill queue
     * eviction or decode reserved-only eviction, contract 5.3) — to its
     * terminal state: same idempotent release/tombstone chain as
     * {@link #finishPreempted}, but the client-visible terminal is the
     * retryable {@code NO_AVAILABLE_WORKER} with the yield reason.
     */
    void finishYielded(BatchItem victim, String detail);

    /**
     * {@link #finishYielded} addressed by request id, for decode
     * reserved-only victims whose {@code BatchItem} is not at hand. No-op
     * when the id is not inflight; idempotent like {@code finishYielded}.
     */
    void finishYieldedById(long requestId, String detail);

    /**
     * Mark an inflight accepted-eviction victim as CANCEL_REQUESTED (Phase 5)
     * so a later engine-reported CANCELLED completion is attributed to
     * {@code PRIORITY_PREEMPTED} instead of a generic worker failure. Default
     * no-op keeps non-scheduler implementations source-compatible.
     *
     * @return true when the id was inflight and the mark was recorded
     */
    default boolean markCancelRequested(long requestId, String detail) {
        return false;
    }
}
