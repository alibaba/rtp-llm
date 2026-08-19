package org.flexlb.balance.endpoint;

/**
 * Per-exit eviction counts from one endpoint-ledger eviction sweep
 * ({@link PrefillEndpoint#evictExpiredBatchesByReason} /
 * {@link DecodeEndpoint#evictExpiredRequestsByReason}).
 *
 * <p>Splits the former single-number eviction return by exit leg so the
 * {@code app.flexlb.inflight.ttl.expired.qps} endpoint series can carry
 * {@code reason} = {@code all_terminal | age_capped | hard_age_cap | ttl}
 * instead of folding every exit into {@code reason=ttl}. Reason values align
 * with the scheduler-side series ({@code ttl}, {@code hard_age_cap}) and the
 * per-exit log events ({@code inflight_batch_age_capped},
 * {@code inflight_hard_age_eviction}, all-terminal settle).
 *
 * @param allTerminal batches released because every member's scheduler-side
 *                    future is already terminal (all-terminal release fix)
 * @param ageCapped   batches force-settled by the progress-aware batch-level
 *                    age cap (F-F)
 * @param hardAgeCap  entries force-released by the guarded hard age cap
 *                    overriding fences and observation keep-alives
 * @param ttl         entries evicted by the normal unobserved TTL
 */
public record EvictionBreakdown(int allTerminal, int ageCapped, int hardAgeCap, int ttl) {

    /** Total evictions across all exits — the former single-number return. */
    public int total() {
        return allTerminal + ageCapped + hardAgeCap + ttl;
    }
}
