package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.InflightEvictor;

/**
 * Tracks a single inflight decode request's KV reservation.
 *
 * @param requestId         engine request ID
 * @param kvTokens          hard KV demand — the prompt's seqLen, used for
 *                          hard-capacity filtering (ensures the prompt itself fits)
 * @param expectedKvTokens  conservative KV estimate — seqLen + maxNewTokens,
 *                          used for scoring / load balancing to account for
 *                          generation-phase KV growth
 * @param createdAtMs       epoch-millis when this entry was created
 */
public record RequestInflight(
        long requestId,
        long kvTokens,
        long expectedKvTokens,
        long createdAtMs
) implements InflightEvictor.TtlTracked {
    public RequestInflight(long requestId, long kvTokens, long expectedKvTokens) {
        this(requestId, kvTokens, expectedKvTokens, System.currentTimeMillis());
    }
}
