package org.flexlb.balance.eviction.model;

/**
 * Immutable per-request descriptor used by the Auto-TPM priority scheduler.
 *
 * <p>Captures everything the admission pipeline needs to place a request:
 * identity, normalized priority and KV demand.
 *
 * @param requestId        unique request id
 * @param priority         normalized priority (30/40/50/60/70, higher = more important)
 * @param seqLen           prompt sequence length in tokens
 * @param maxNewTokens     generation budget in tokens
 * @param arrivalTimeMs    server-side arrival timestamp (epoch ms)
 * @param hardKvTokens     minimum KV demand (= seqLen, prompt must fit)
 * @param expectedKvTokens expected KV demand: min(seqLen + maxNewTokens, decode KV total)
 */
public record PriorityRequestEnvelope(
        long requestId,
        int priority,
        long seqLen,
        long maxNewTokens,
        long arrivalTimeMs,
        long hardKvTokens,
        long expectedKvTokens) {
}
