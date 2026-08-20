package org.flexlb.balance.scheduler.priority;

/**
 * Read-only view of one request queued in a prefill batcher, captured by
 * {@code PrefillQueueManager.snapshot()} for eviction planning.
 *
 * @param requestId      unique request id
 * @param priority       normalized priority (30/40/50/60/70)
 * @param arrivalTimeMs  batcher enqueue timestamp (epoch ms)
 * @param seqLen         prompt sequence length in tokens
 * @param cacheHitTokens cache-hit tokens on the assigned prefill endpoint
 * @param state          scheduling stage; queued items are always
 *                       {@link #PREFILL_QUEUED}
 */
public record QueuedRequestSnapshot(
        long requestId,
        int priority,
        long arrivalTimeMs,
        long seqLen,
        long cacheHitTokens,
        String state) {

    public static final String PREFILL_QUEUED = "PREFILL_QUEUED";
}
