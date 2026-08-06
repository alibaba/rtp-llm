package org.flexlb.balance.scheduler.priority;

import java.util.Comparator;
import java.util.List;

/**
 * Pure planning result for one decode reserved-only eviction on one endpoint
 * (design doc 11-13). Produced by {@link EvictionPlanner#planDecode}; carries
 * no live endpoint reference and has zero side effects.
 *
 * <p>Victims are always {@code RESERVED_NOT_ACCEPTED} shadow reservations of
 * strictly lower priority than the incoming request (design doc 3.3);
 * confirmed (accepted/running) requests never appear.
 *
 * @param endpointId       decode endpoint key ("ip:httpPort")
 * @param admissionVersion admission version the plan was built against
 * @param victims          selected victims in eviction order
 * @param evictionCase     {@link #CASE_SLOT} / {@link #CASE_KV} / {@link #CASE_SLOT_AND_KV}
 * @param totalCost        h-weighted total cost; a combined plan sums its two
 *                         already-weighted parts and is never re-multiplied
 * @param freedKvTokens    sum of the victims' releasable hard KV tokens
 * @param cost             structured {@link PlanCost} for 7.2 comparison
 */
public record DecodeEvictionProposal(
        String endpointId,
        long admissionVersion,
        List<DecodeRequestSnapshot> victims,
        String evictionCase,
        long totalCost,
        long freedKvTokens,
        PlanCost cost) {

    /** Concurrency slots exhausted (design doc 11). */
    public static final String CASE_SLOT = "decode_slot_full";
    /** Real KV available below the incoming hard demand (design doc 12). */
    public static final String CASE_KV = "decode_kv_full";
    /** Both deficits at once (design doc 13). */
    public static final String CASE_SLOT_AND_KV = "decode_slot_and_kv_full";

    /**
     * Cross-endpoint plan preference (smaller = better): {@link PlanCost#ORDER}
     * then endpointId for determinism.
     */
    public static final Comparator<DecodeEvictionProposal> ORDER = Comparator
            .comparing(DecodeEvictionProposal::cost, PlanCost.ORDER)
            .thenComparing(DecodeEvictionProposal::endpointId);
}
