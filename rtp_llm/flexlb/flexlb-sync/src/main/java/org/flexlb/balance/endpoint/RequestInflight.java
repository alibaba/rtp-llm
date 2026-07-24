package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.InflightEvictor;

/**
 * Tracks a single inflight decode request's KV reservation.
 *
 * @param kvTokens          hard KV demand — the prompt's seqLen, used for
 *                          hard-capacity filtering (ensures the prompt itself fits)
 * @param expectedKvTokens  conservative KV estimate — seqLen + maxNewTokens,
 *                          used for scoring / load balancing to account for
 *                          generation-phase KV growth
 * @param createdAtMs       epoch-millis when this entry was created
 */
record RequestInflight(
        long kvTokens,
        long expectedKvTokens,
        long createdAtMs
) implements InflightEvictor.TtlTracked {
    RequestInflight(long kvTokens, long expectedKvTokens) {
        this(kvTokens, expectedKvTokens, System.currentTimeMillis());
    }
}
