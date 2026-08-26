package org.flexlb.balance.eviction;

import org.flexlb.enums.DecodeTaskPhase;

/**
 * Read-only view of one decode shadow reservation, captured by
 * {@link DecodeEndpointSnapshot#capture} for eviction planning.
 *
 * @param requestId        unique request id
 * @param priority         normalized priority (30/40/50/60/70)
 * @param phase            Decode admission phase: Master-local planning sees
 *                         queued/unconfirmed reservations; Engine Cancel
 *                         planning also sees accepted and running tasks
 * @param kvTokens         releasable hard KV reservation (= prompt seqLen)
 * @param expectedKvTokens conservative KV estimate (seqLen + maxNewTokens)
 * @param priorityKnown    true only when this Master supplied the priority;
 *                         foreign confirmed tasks must never be guessed from
 *                         the default priority value
 * @param queued           reserved entry still parked in a prefill queue
 *                         (N2 queued phase): evicting it frees KV but no
 *                         engine concurrency slot (review P1-3)
 * @param reservationToken exact endpoint-local identity of a reserved shadow;
 *                         zero for confirmed Engine-owned observations
 */
public record DecodeRequestSnapshot(
        long requestId,
        int priority,
        DecodeTaskPhase phase,
        long kvTokens,
        long expectedKvTokens,
        boolean priorityKnown,
        boolean queued,
        long reservationToken) {

}
