package org.flexlb.balance.scheduler.priority;

import org.flexlb.enums.DecodeTaskPhase;

/**
 * Read-only view of one decode shadow reservation, captured by
 * {@link DecodeEndpointSnapshot#capture} for eviction planning.
 *
 * @param requestId        unique request id
 * @param priority         normalized priority (30/40/50/60/70)
 * @param phase            decode admission phase; Phase 4 only ever sees
 *                         {@code RESERVED_NOT_ACCEPTED}
 * @param kvTokens         releasable hard KV reservation (= prompt seqLen)
 * @param expectedKvTokens conservative KV estimate (seqLen + maxNewTokens)
 * @param deadlineMs       admission deadline (epoch ms); 0 = unset
 * @param queued           reserved entry still parked in a prefill queue
 *                         (N2 queued phase): evicting it frees KV but no
 *                         engine concurrency slot (review P1-3)
 */
public record DecodeRequestSnapshot(
        long requestId,
        int priority,
        DecodeTaskPhase phase,
        long kvTokens,
        long expectedKvTokens,
        long deadlineMs,
        boolean queued) {

    /** Compatibility constructor: non-queued entry (pre-P1-3 call sites). */
    public DecodeRequestSnapshot(long requestId, int priority, DecodeTaskPhase phase,
                                 long kvTokens, long expectedKvTokens, long deadlineMs) {
        this(requestId, priority, phase, kvTokens, expectedKvTokens, deadlineMs, false);
    }
}
