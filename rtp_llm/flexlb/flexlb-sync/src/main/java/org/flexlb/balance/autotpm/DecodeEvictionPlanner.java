package org.flexlb.balance.autotpm;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Pure function: {@link DecodeAdmissionTracker} state + incoming request → {@link DecodeEvictionPlan}.
 *
 * <p>Stateless and side-effect free — performs no tracker mutation, no CAS.
 * The caller ({@link DecodeEvictionCommitter}) is responsible for committing the plan.
 *
 * <h2>Hard rule</h2>
 * A victim is eligible only when {@code victim.priority < incoming.priority}.
 * Same-priority requests are never evicted. This is enforced by
 * {@link DecodeAdmissionTracker#findSlotEvictionCandidates} and
 * {@link DecodeAdmissionTracker#findKvEvictionCandidates}.
 *
 * <h2>MVP rule</h2>
 * Only {@link DecodeAdmissionState#RESERVED_NOT_ACCEPTED} and
 * {@link DecodeAdmissionState#ACCEPTED_NOT_RUNNING} are evictable.
 * {@link DecodeAdmissionState#RUNNING} is never evicted.
 *
 * <h2>Three planning modes</h2>
 * <ol>
 *   <li><b>Slot-only:</b> D slot is full. Find lower-priority victims that free
 *       the needed slots. Sort: priority lowest first, same priority earlier
 *       stage first.</li>
 *   <li><b>KV-only:</b> D KV is insufficient. Find lower-priority victims that
 *       release the most KV. Sort: priority lowest first, same priority more KV
 *       released first, same priority earlier stage first.</li>
 *   <li><b>Combined:</b> Both slot+KV insufficient. Try slot-only plan — if it
 *       also satisfies KV, use it. Try kv-only plan — if it also satisfies slots,
 *       use it. Otherwise combine plans with victim DEDUP (no victim evicted
 *       twice).</li>
 * </ol>
 */
public final class DecodeEvictionPlanner {

    /**
     * Plan decode eviction when D slot is full.
     *
     * @param tracker          the admission tracker (source of truth)
     * @param endpointKey      ip:port of the decode endpoint
     * @param incomingPriority priority of the incoming request
     * @param neededSlots      number of slots the incoming needs (typically 1)
     * @param maxVictims       max victims per decision (from config)
     * @return eviction plan, or empty plan if no eligible victims
     */
    public DecodeEvictionPlan planSlotEviction(DecodeAdmissionTracker tracker,
                                               String endpointKey,
                                               int incomingPriority,
                                               int neededSlots,
                                               int maxVictims) {
        List<DecodeReservation> candidates = tracker.findSlotEvictionCandidates(
                endpointKey, incomingPriority, neededSlots);
        return buildPlan(candidates, endpointKey, maxVictims);
    }

    /**
     * Plan decode eviction when D KV is insufficient.
     *
     * @param tracker          the admission tracker (source of truth)
     * @param endpointKey      ip:port of the decode endpoint
     * @param incomingPriority priority of the incoming request
     * @param neededKv         KV tokens the incoming needs
     * @param maxVictims       max victims per decision (from config)
     * @return eviction plan, or empty plan if no eligible victims
     */
    public DecodeEvictionPlan planKvEviction(DecodeAdmissionTracker tracker,
                                             String endpointKey,
                                             int incomingPriority,
                                             long neededKv,
                                             int maxVictims) {
        List<DecodeReservation> candidates = tracker.findKvEvictionCandidates(
                endpointKey, incomingPriority, neededKv);
        return buildPlan(candidates, endpointKey, maxVictims);
    }

    /**
     * Plan combined eviction when both slot+KV insufficient.
     *
     * <p>Strategy:
     * <ol>
     *   <li>Try slot-only plan. If it also satisfies KV, use it (fewer victims, simpler).</li>
     *   <li>Try kv-only plan. If it also satisfies slots, use it.</li>
     *   <li>Otherwise combine both plans with victim DEDUP: take all unique victims
     *       from both plans. The combined plan may exceed maxVictims (the union of
     *       two plans each capped at maxVictims can be up to 2*maxVictims), but
     *       since the same victim won't be evicted twice, the actual count is bounded.</li>
     * </ol>
     *
     * @param tracker          the admission tracker
     * @param endpointKey      ip:port of the decode endpoint
     * @param incomingPriority priority of the incoming request
     * @param neededSlots      slots the incoming needs
     * @param neededKv         KV tokens the incoming needs
     * @param maxVictims       max victims per decision
     * @return eviction plan, or empty plan if no eligible victims
     */
    public DecodeEvictionPlan planCombinedEviction(DecodeAdmissionTracker tracker,
                                                   String endpointKey,
                                                   int incomingPriority,
                                                   int neededSlots,
                                                   long neededKv,
                                                   int maxVictims) {
        // 1. Try slot-only plan
        DecodeEvictionPlan slotPlan = planSlotEviction(
                tracker, endpointKey, incomingPriority, neededSlots, maxVictims);
        if (!slotPlan.isEmpty() && slotPlan.satisfiesSlots(neededSlots)
                && slotPlan.satisfiesKv(neededKv)) {
            // Slot-only plan satisfies both — use it (simplest)
            return slotPlan;
        }

        // 2. Try kv-only plan
        DecodeEvictionPlan kvPlan = planKvEviction(
                tracker, endpointKey, incomingPriority, neededKv, maxVictims);
        if (!kvPlan.isEmpty() && kvPlan.satisfiesSlots(neededSlots)
                && kvPlan.satisfiesKv(neededKv)) {
            // KV-only plan satisfies both — use it
            return kvPlan;
        }

        // 3. Combine with victim DEDUP
        if (slotPlan.isEmpty() && kvPlan.isEmpty()) {
            return emptyPlan(endpointKey);
        }

        // Merge victims, dedup by requestId
        Map<Long, DecodeReservation> deduped = new LinkedHashMap<>();
        for (DecodeReservation r : slotPlan.victims()) {
            deduped.putIfAbsent(r.requestId(), r);
        }
        for (DecodeReservation r : kvPlan.victims()) {
            deduped.putIfAbsent(r.requestId(), r);
        }

        List<DecodeReservation> merged = new ArrayList<>(deduped.values());
        return buildPlan(merged, endpointKey, merged.size()); // no cap on combined
    }

    // ==================== Internal: build plan from candidates ====================

    private DecodeEvictionPlan buildPlan(List<DecodeReservation> candidates,
                                         String endpointKey,
                                         int maxVictims) {
        if (candidates == null || candidates.isEmpty()) {
            return emptyPlan(endpointKey);
        }

        int selectCount = Math.min(Math.max(0, maxVictims), candidates.size());
        List<DecodeReservation> selected = new ArrayList<>(candidates.subList(0, selectCount));

        // Compute structured cost
        long priorityCost = 0L;
        int stageCost = 0;
        long resourceCost = 0L;
        long tieBreak = 0L;
        int slotsFreed = 0;
        long kvFreed = 0L;

        for (DecodeReservation v : selected) {
            priorityCost += PriorityCostFunction.f(v.priority());
            stageCost += PriorityCostFunction.g(v.state().toVictimStage());
            resourceCost += v.kvTokensRequired();
            tieBreak += v.requestId();
            slotsFreed += 1; // each victim frees one slot
            kvFreed += v.kvTokensRequired();
        }

        PlanCost cost = new PlanCost(priorityCost, stageCost, resourceCost,
                selected.size(), 0.0, tieBreak);

        return new DecodeEvictionPlan(selected, cost, slotsFreed, kvFreed, endpointKey);
    }

    private static DecodeEvictionPlan emptyPlan(String endpointKey) {
        return new DecodeEvictionPlan(Collections.emptyList(),
                new PlanCost(0L, 0, 0L, 0, 0.0, 0L),
                0, 0L, endpointKey);
    }
}
