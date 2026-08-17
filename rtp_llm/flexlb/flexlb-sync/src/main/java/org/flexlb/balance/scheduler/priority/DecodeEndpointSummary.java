package org.flexlb.balance.scheduler.priority;

import org.flexlb.balance.endpoint.DecodeEndpoint;

/**
 * Lightweight aggregate view of a decode endpoint — O(1) capture, zero list
 * allocation, no admission lock (O(1) snapshot redesign §3.2).
 *
 * <p>Every field is read from an already incrementally-maintained atomic /
 * volatile counter on the endpoint, so the aggregates can never drift from
 * the mutation state: {@code reserve()/release()/calibrate()} and the
 * eviction commits update those counters inside their own lock or atomic
 * scope. The fields are weakly consistent <i>with each other</i> (no lock is
 * held across the reads), which matches the pre-existing contract of the KV /
 * load fields on {@link DecodeEndpointSnapshot} — the lockfree commit path
 * never validates snapshot state, and the eviction path re-validates victims
 * live at commit time.
 *
 * <p>The normal placement path consumes only these scalars; eviction /
 * failure-classification paths upgrade to the full per-entry
 * {@link DecodeEndpointSnapshot} via {@link #toFullSnapshot()} only when they
 * actually need the reserved/accepted/running lists.
 *
 * @param endpoint           the live endpoint (used for the lazy upgrade)
 * @param endpointId         endpoint key ("ip:httpPort")
 * @param admissionVersion   admission version at capture time (informational;
 *                           commit-time guards read the live endpoint)
 * @param realKvAvailable    engine-reported available KV minus local hard reservations
 * @param realKvTotal        engine-reported total KV capacity
 * @param totalLoad          confirmed running + local inflight request count
 * @param engineLoad         engine-facing load (same measure as the N2 gate)
 * @param concurrencyLimit   configured decode concurrency limit (0 = unlimited)
 * @param hardKvReserved     sum of reserved hard KV tokens (shadow accounting)
 * @param expectedKvReserved sum of reserved expected KV tokens (shadow accounting)
 */
public record DecodeEndpointSummary(
        DecodeEndpoint endpoint,
        String endpointId,
        long admissionVersion,
        long realKvAvailable,
        long realKvTotal,
        int totalLoad,
        int engineLoad,
        long concurrencyLimit,
        long hardKvReserved,
        long expectedKvReserved) {

    /**
     * O(1) capture: reads only volatile/atomic aggregates — no admission
     * lock, no per-entry list copies.
     */
    public static DecodeEndpointSummary capture(DecodeEndpoint endpoint, long concurrencyLimit) {
        return new DecodeEndpointSummary(
                endpoint,
                endpoint.ipPort(),
                endpoint.admissionVersion(),
                endpoint.realKvAvailable(),
                endpoint.realKvTotal(),
                endpoint.getTotalLoad(),
                endpoint.getEngineLoad(),
                concurrencyLimit,
                endpoint.inflightHardKvReserved(),
                endpoint.inflightExpectedKvReserved());
    }

    /**
     * Upgrade to the full per-entry snapshot (briefly holds the endpoint
     * admission lock). Call only when eviction planning or failure
     * classification actually needs the reserved/accepted/running lists.
     */
    public DecodeEndpointSnapshot toFullSnapshot() {
        return DecodeEndpointSnapshot.capture(endpoint, concurrencyLimit);
    }

    /** Slot deficit on the aggregates (mirrors {@code EvictionPlanner.slotDeficit}). */
    public long slotDeficit() {
        return concurrencyLimit > 0 ? Math.max(0, engineLoad + 1 - concurrencyLimit) : 0;
    }

    /** KV deficit on the aggregates (mirrors {@code EvictionPlanner.kvDeficit}). */
    public long kvDeficit(long incomingHardKvTokens) {
        return (realKvTotal > 0 && realKvAvailable < incomingHardKvTokens)
                ? incomingHardKvTokens - realKvAvailable : 0;
    }
}
