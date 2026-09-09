package org.flexlb.balance.eviction.model;

/**
 * Immutable per-request descriptor used by priority preemption planning.
 *
 * <p>Captures everything the admission pipeline needs to place a request:
 * identity, normalized priority and hard KV demand.
 *
 * @param requestId        unique request id
 * @param priority         normalized priority in [1, 100] (higher = more important)
 * @param hardKvTokens     minimum KV demand (= seqLen, prompt must fit)
 */
public record PriorityRequestEnvelope(
        long requestId,
        int priority,
        long hardKvTokens) {
}
