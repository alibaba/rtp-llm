package org.flexlb.balance.scheduler.priority;

import org.flexlb.config.FlexlbConfig;
import org.flexlb.enums.DecodeTaskPhase;
import org.flexlb.util.PriorityNormalizer;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Pure planner for prefill-queue eviction (design doc 9.1-9.4). Zero side
 * effects: it only inspects {@link PrefillQueueSnapshot}s and returns the
 * cheapest feasible {@link PrefillEvictionProposal}, or {@code null} with
 * per-endpoint reasons recorded into {@code failures}.
 *
 * <p>Absolute rule (design doc 3.3): only strictly lower-priority queued
 * requests are eviction candidates; equal priority never yields. The bounded
 * cache benefit (see {@link PriorityCostFunction#boundedCacheBenefit}) cannot
 * reverse that boundary.
 */
public final class EvictionPlanner {

    /**
     * Safety valve: max victims a single eviction plan may select (task43 收编：
     * 原 autoTpmMaxVictimsPerPlan 配置，无需对外暴露，取原默认值）.
     */
    static final int MAX_VICTIMS_PER_PLAN = 8;

    /**
     * Candidate preference (design doc 9.3, first = evicted first):
     * priority asc → deadline desc (more slack first) → arrival desc
     * (newest first) → requestId asc (deterministic).
     */
    static final Comparator<QueuedRequestSnapshot> CANDIDATE_ORDER = Comparator
            .comparingInt(QueuedRequestSnapshot::priority)
            .thenComparing(QueuedRequestSnapshot::deadlineMs, Comparator.reverseOrder())
            .thenComparing(QueuedRequestSnapshot::arrivalTimeMs, Comparator.reverseOrder())
            .thenComparingLong(QueuedRequestSnapshot::requestId);

    private EvictionPlanner() {
    }

    /** A decode plan never mixes Master-local removal with Engine Cancel. */
    private enum VictimOwnership {
        MASTER_LOCAL,
        ENGINE_CANCEL
    }

    /**
     * Plan the cheapest prefill-queue eviction that frees enough room for the
     * incoming request across the given queues.
     *
     * @param envelope       incoming request descriptor
     * @param queues         candidate queue snapshots (production passes the
     *                       router-selected endpoint's queue only)
     * @param cacheHitTokens incoming cache-hit tokens per endpointId (may be empty)
     * @param config         live config (cache benefit cap)
     * @param failures       out-param: per-endpoint infeasibility reason
     * @return the best proposal by {@link PrefillEvictionProposal#ORDER}, or
     *         {@code null} when no endpoint has a feasible plan
     */
    public static PrefillEvictionProposal planPrefillQueue(PriorityRequestEnvelope envelope,
                                                           List<PrefillQueueSnapshot> queues,
                                                           Map<String, Long> cacheHitTokens,
                                                           FlexlbConfig config,
                                                           Map<String, String> failures) {
        PrefillEvictionProposal best = null;
        for (PrefillQueueSnapshot queue : queues) {
            PrefillEvictionProposal proposal = planOne(envelope, queue,
                    cacheHitTokens.getOrDefault(queue.endpointId(), 0L), config, failures);
            if (proposal != null
                    && (best == null || PrefillEvictionProposal.ORDER.compare(proposal, best) < 0)) {
                best = proposal;
            }
        }
        return best;
    }

    private static PrefillEvictionProposal planOne(PriorityRequestEnvelope envelope,
                                                   PrefillQueueSnapshot queue,
                                                   long endpointCacheHitTokens,
                                                   FlexlbConfig config,
                                                   Map<String, String> failures) {
        int hardLimit = queue.queueCapacity();
        if (hardLimit <= 0) {
            failures.put(queue.endpointId(), "queue_unbounded");
            return null;
        }

        // 9.1: how many slots must be freed for the incoming request to fit.
        int queueDeficit = queue.items().size() + 1 - hardLimit;
        if (queueDeficit <= 0) {
            failures.put(queue.endpointId(), "queue_not_full");
            return null;
        }
        int maxVictims = MAX_VICTIMS_PER_PLAN;
        if (maxVictims > 0 && queueDeficit > maxVictims) {
            failures.put(queue.endpointId(), "deficit_exceeds_max_victims");
            return null;
        }

        // 9.2: only PREFILL_QUEUED requests of strictly lower priority.
        // Task40: no-priority items (legacy path) are never selected as
        // victims — filter them out before the < comparison, which would
        // otherwise always match them.
        List<QueuedRequestSnapshot> candidates = new ArrayList<>();
        for (QueuedRequestSnapshot item : queue.items()) {
            if (QueuedRequestSnapshot.PREFILL_QUEUED.equals(item.state())
                    && PriorityNormalizer.hasPriority(item.priority())
                    && item.priority() < envelope.priority()) {
                candidates.add(item);
            }
        }
        if (candidates.size() < queueDeficit) {
            failures.put(queue.endpointId(), "insufficient_lower_priority_candidates");
            return null;
        }

        candidates.sort(CANDIDATE_ORDER);
        List<QueuedRequestSnapshot> victims = List.copyOf(candidates.subList(0, queueDeficit));

        // 9.3: netCost = rawCost - boundedCacheBenefit.
        long rawCost = 0;
        int minVictimPriority = Integer.MAX_VALUE;
        long latestVictimDeadline = Long.MIN_VALUE;
        long tieBreak = Long.MAX_VALUE;
        for (QueuedRequestSnapshot victim : victims) {
            rawCost += PriorityCostFunction.f(victim.priority());
            minVictimPriority = Math.min(minVictimPriority, victim.priority());
            latestVictimDeadline = Math.max(latestVictimDeadline, victim.deadlineMs());
            tieBreak = Math.min(tieBreak, victim.requestId());
        }
        long boundedBenefit = PriorityCostFunction.boundedCacheBenefit(
                endpointCacheHitTokens, config.getAutoTpmPlanCacheHitBenefitCap());
        long netCost = rawCost - boundedBenefit;

        PlanCost cost = new PlanCost(minVictimPriority, netCost, victims.size(),
                boundedBenefit, latestVictimDeadline, tieBreak);
        return new PrefillEvictionProposal(queue.endpointId(), queue.queueVersion(), victims,
                rawCost, endpointCacheHitTokens, boundedBenefit, netCost, cost);
    }

    // ==================== Decode reserved-only eviction (design doc 11-13) ====================

    /**
     * Candidate preference for slot eviction (design doc 11.3, first = evicted
     * first): priority asc → stage asc (reserved before accepted, Phase 5) →
     * deadline desc (more slack first) → requestId asc.
     */
    static final Comparator<DecodeRequestSnapshot> DECODE_SLOT_ORDER = Comparator
            .comparingInt(DecodeRequestSnapshot::priority)
            .thenComparingInt(v -> v.phase().ordinal())
            .thenComparing(DecodeRequestSnapshot::deadlineMs, Comparator.reverseOrder())
            .thenComparingLong(DecodeRequestSnapshot::requestId);

    /**
     * Candidate preference for KV eviction (design doc 12.4): priority asc →
     * stage asc (reserved before accepted, Phase 5) → kvBucket desc (bigger
     * releases first, fewer victims) → deadline desc → requestId asc.
     */
    static final Comparator<DecodeRequestSnapshot> DECODE_KV_ORDER = Comparator
            .comparingInt(DecodeRequestSnapshot::priority)
            .thenComparingInt(v -> v.phase().ordinal())
            .thenComparing(v -> PriorityCostFunction.kvBucket(v.kvTokens()), Comparator.reverseOrder())
            .thenComparing(DecodeRequestSnapshot::deadlineMs, Comparator.reverseOrder())
            .thenComparingLong(DecodeRequestSnapshot::requestId);

    /**
     * Reserved-only decode planning (Phase 4 signature): equivalent to
     * {@link #planDecode(PriorityRequestEnvelope, List, FlexlbConfig,
     * EngineCancelChannel, Map)} with no cancel channel, so engine-confirmed
     * layers are never considered. Kept for callers that must not preempt confirmed
     * requests (e.g. deadline-rescue placement).
     */
    public static DecodeEvictionProposal planDecode(PriorityRequestEnvelope envelope,
                                                    List<DecodeEndpointSnapshot> decodes,
                                                    FlexlbConfig config,
                                                    Map<String, String> failures) {
        return planDecode(envelope, decodes, config, null, failures);
    }

    /**
     * Plan the cheapest decode eviction that clears the incoming request's
     * slot and/or KV deficit across the given endpoints.
     *
     * <p>Candidates are the strictly lower-priority reserved entries, plus —
     * only when {@code autoTpmDecodeAcceptedEvictEnabled} is set AND the
     * endpoint's engine supports the Cancel RPC — the strictly lower-priority
     * engine-confirmed accepted/running entries (Phase 5). Running entries use
     * a larger stage cost, so an otherwise equivalent accepted-not-running
     * victim is preferred.
     *
     * @param envelope incoming request descriptor (priority + hardKvTokens)
     * @param decodes  candidate decode endpoint snapshots
     * @param config   live config (victim cap + engine-confirmed-evict gate)
     * @param channel  engine cancel channel for the per-endpoint support gate;
     *                 {@code null} disables confirmed layers entirely
     * @param failures out-param: per-endpoint infeasibility reason
     * @return the best proposal by {@link DecodeEvictionProposal#ORDER}, or
     *         {@code null} when no endpoint has a feasible plan
     */
    public static DecodeEvictionProposal planDecode(PriorityRequestEnvelope envelope,
                                                    List<DecodeEndpointSnapshot> decodes,
                                                    FlexlbConfig config,
                                                    EngineCancelChannel channel,
                                                    Map<String, String> failures) {
        DecodeEvictionProposal best = null;
        for (DecodeEndpointSnapshot ep : decodes) {
            DecodeEvictionProposal proposal = planDecodeOne(envelope, ep, config, channel, failures);
            if (proposal != null
                    && (best == null || DecodeEvictionProposal.ORDER.compare(proposal, best) < 0)) {
                best = proposal;
            }
        }
        return best;
    }

    /**
     * Eviction case of one endpoint for the incoming request, or {@code null}
     * when its capacity is sufficient. Shared with the scheduler for
     * infeasible-plan metric labeling.
     */
    public static String decodeEvictionCase(PriorityRequestEnvelope envelope, DecodeEndpointSnapshot ep) {
        boolean slot = slotDeficit(ep) > 0;
        boolean kv = kvDeficit(envelope, ep) > 0;
        if (slot && kv) {
            return DecodeEvictionProposal.CASE_SLOT_AND_KV;
        }
        if (slot) {
            return DecodeEvictionProposal.CASE_SLOT;
        }
        return kv ? DecodeEvictionProposal.CASE_KV : null;
    }

    /**
     * 11.1: slots to free so engineLoad + 1 fits the limit (0 = unlimited).
     * P1-3: measured against the engine-facing load — the same measure the
     * N2 concurrency gate uses — so queued-phase reservations neither create
     * a phantom deficit nor hide a real one.
     */
    static long slotDeficit(DecodeEndpointSnapshot ep) {
        long limit = ep.concurrencyLimit();
        return limit > 0 ? Math.max(0, ep.engineLoad() + 1 - limit) : 0;
    }

    /**
     * 12.1: hard KV tokens to free so the prompt fits, mirroring the decode
     * strategy's hard filter ({@code totalKv > 0 && available < seqLen}).
     */
    static long kvDeficit(PriorityRequestEnvelope envelope, DecodeEndpointSnapshot ep) {
        if (ep.realKvTotal() > 0 && ep.realKvAvailable() < envelope.hardKvTokens()) {
            return envelope.hardKvTokens() - ep.realKvAvailable();
        }
        return 0;
    }

    private static DecodeEvictionProposal planDecodeOne(PriorityRequestEnvelope envelope,
                                                        DecodeEndpointSnapshot ep,
                                                        FlexlbConfig config,
                                                        EngineCancelChannel channel,
                                                        Map<String, String> failures) {
        long slotDeficit = slotDeficit(ep);
        long kvDeficit = kvDeficit(envelope, ep);
        if (slotDeficit <= 0 && kvDeficit <= 0) {
            failures.put(ep.endpointId(), "decode_capacity_sufficient");
            return null;
        }
        boolean engineCancelEnabled = config.isAutoTpmDecodeAcceptedEvictEnabled()
                && channel != null && channel.isSupported(ep.endpoint());
        int maxVictims = MAX_VICTIMS_PER_PLAN;

        DecodeEvictionProposal local = planDecodeOneOwnership(envelope, ep,
                slotDeficit, kvDeficit, maxVictims, VictimOwnership.MASTER_LOCAL, failures);
        DecodeEvictionProposal engine = engineCancelEnabled
                ? planDecodeOneOwnership(envelope, ep, slotDeficit, kvDeficit,
                        maxVictims, VictimOwnership.ENGINE_CANCEL, failures)
                : null;
        if (local == null) {
            return engine;
        }
        if (engine == null) {
            return local;
        }
        return DecodeEvictionProposal.ORDER.compare(local, engine) <= 0 ? local : engine;
    }

    private static DecodeEvictionProposal planDecodeOneOwnership(
            PriorityRequestEnvelope envelope,
            DecodeEndpointSnapshot ep,
            long slotDeficit,
            long kvDeficit,
            int maxVictims,
            VictimOwnership ownership,
            Map<String, String> failures) {
        if (slotDeficit > 0 && kvDeficit > 0) {
            return planDecodeCombined(envelope, ep, slotDeficit, kvDeficit,
                    maxVictims, ownership, failures);
        }
        DecodeVictimSet set = slotDeficit > 0
                ? selectSlotVictims(envelope, ep, slotDeficit, Set.of(), maxVictims, ownership)
                : selectKvVictims(envelope, ep, kvDeficit, Set.of(), maxVictims, ownership);
        if (!set.ok()) {
            failures.put(ep.endpointId(), set.failReason());
            return null;
        }
        String evictionCase = slotDeficit > 0
                ? DecodeEvictionProposal.CASE_SLOT
                : DecodeEvictionProposal.CASE_KV;
        return buildDecodeProposal(ep, evictionCase, set.victims(), set.weightedCost(), set.freedKvTokens());
    }

    /**
     * 13: both deficits at once. A single-dimension plan whose side effect
     * already clears the other deficit wins (cheapest, keeps its own case
     * tag); otherwise build a two-part plan ordered by scarcity, victims
     * deduplicated via exclusion, {@code totalCost} = sum of the two already
     * h-weighted parts (never re-multiplied).
     */
    private static DecodeEvictionProposal planDecodeCombined(PriorityRequestEnvelope envelope,
                                                             DecodeEndpointSnapshot ep,
                                                             long slotDeficit,
                                                             long kvDeficit,
                                                             int maxVictims,
                                                             VictimOwnership ownership,
                                                             Map<String, String> failures) {
        // 13.2: slot-only incidentally freeing enough KV / kv-only incidentally
        // freeing enough slots — prefer the cheaper single-case plan.
        DecodeVictimSet slotOnly = selectSlotVictims(envelope, ep, slotDeficit, Set.of(),
                maxVictims, ownership);
        DecodeEvictionProposal slotSide = slotOnly.ok() && slotOnly.freedKvTokens() >= kvDeficit
                ? buildDecodeProposal(ep, DecodeEvictionProposal.CASE_SLOT,
                        slotOnly.victims(), slotOnly.weightedCost(), slotOnly.freedKvTokens())
                : null;
        DecodeVictimSet kvOnly = selectKvVictims(envelope, ep, kvDeficit, Set.of(),
                maxVictims, ownership);
        // P1-3: only non-queued victims free an engine slot — queued ones
        // never counted against the engine load in the first place.
        DecodeEvictionProposal kvSide = kvOnly.ok() && nonQueuedCount(kvOnly.victims()) >= slotDeficit
                ? buildDecodeProposal(ep, DecodeEvictionProposal.CASE_KV,
                        kvOnly.victims(), kvOnly.weightedCost(), kvOnly.freedKvTokens())
                : null;
        if (slotSide != null || kvSide != null) {
            if (slotSide == null) {
                return kvSide;
            }
            if (kvSide == null) {
                return slotSide;
            }
            return DecodeEvictionProposal.ORDER.compare(slotSide, kvSide) <= 0 ? slotSide : kvSide;
        }

        // 13.3: scarcity-ordered combined plan — the scarcer dimension plans
        // first, the other covers only what remains after the first's side
        // effect, excluding already-selected victims (dedup by construction).
        double slotPressure = (double) slotDeficit / Math.max(1, ep.concurrencyLimit());
        double kvPressure = (double) kvDeficit / Math.max(1, ep.realKvTotal());
        DecodeVictimSet first;
        DecodeVictimSet second;
        if (kvPressure >= slotPressure) {
            first = selectKvVictims(envelope, ep, kvDeficit, Set.of(), maxVictims, ownership);
            if (!first.ok()) {
                failures.put(ep.endpointId(), first.failReason());
                return null;
            }
            long remainingSlots = slotDeficit - nonQueuedCount(first.victims());
            second = remainingSlots > 0
                    ? selectSlotVictims(envelope, ep, remainingSlots,
                            victimIds(first.victims()), remainingBudget(maxVictims, first),
                            ownership)
                    : DecodeVictimSet.EMPTY;
        } else {
            first = selectSlotVictims(envelope, ep, slotDeficit, Set.of(), maxVictims, ownership);
            if (!first.ok()) {
                failures.put(ep.endpointId(), first.failReason());
                return null;
            }
            long remainingKv = kvDeficit - first.freedKvTokens();
            second = remainingKv > 0
                    ? selectKvVictims(envelope, ep, remainingKv,
                            victimIds(first.victims()), remainingBudget(maxVictims, first),
                            ownership)
                    : DecodeVictimSet.EMPTY;
        }
        if (!second.ok()) {
            failures.put(ep.endpointId(), second.failReason());
            return null;
        }
        List<DecodeRequestSnapshot> victims = new ArrayList<>(first.victims());
        victims.addAll(second.victims());
        return buildDecodeProposal(ep, DecodeEvictionProposal.CASE_SLOT_AND_KV, victims,
                first.weightedCost() + second.weightedCost(),
                first.freedKvTokens() + second.freedKvTokens());
    }

    /**
     * 11.3: pick exactly {@code deficit} slot victims. Cost part:
     * {@code h(DECODE_SLOT_FULL) × Σ f(priority) × g(stage)}.
     */
    private static DecodeVictimSet selectSlotVictims(PriorityRequestEnvelope envelope,
                                                     DecodeEndpointSnapshot ep,
                                                     long deficit,
                                                     Set<Long> excludedVictimIds,
                                                     int maxVictims,
                                                     VictimOwnership ownership) {
        if (maxVictims > 0 && deficit > maxVictims) {
            return DecodeVictimSet.fail("deficit_exceeds_max_victims");
        }
        List<DecodeRequestSnapshot> candidates =
                lowerPriorityCandidates(envelope, ep, excludedVictimIds, false, ownership, true);
        if (candidates.size() < deficit) {
            return DecodeVictimSet.fail("insufficient_lower_priority_candidates");
        }
        candidates.sort(DECODE_SLOT_ORDER);
        List<DecodeRequestSnapshot> victims = List.copyOf(candidates.subList(0, (int) deficit));
        long cost = 0;
        long freedKv = 0;
        for (DecodeRequestSnapshot victim : victims) {
            cost += PriorityCostFunction.f(victim.priority()) * PriorityCostFunction.g(victim.phase());
            freedKv += victim.kvTokens();
        }
        return new DecodeVictimSet(victims,
                PriorityCostFunction.H_DECODE_SLOT_FULL * cost, freedKv, null);
    }

    /**
     * 12.4: greedily pick victims until the freed hard KV covers
     * {@code kvDeficit}. Cost part: {@code h(DECODE_KV_FULL) ×
     * Σ f(priority) × g(stage) × lengthWasteCost(kvTokens)}.
     */
    private static DecodeVictimSet selectKvVictims(PriorityRequestEnvelope envelope,
                                                   DecodeEndpointSnapshot ep,
                                                   long kvDeficit,
                                                   Set<Long> excludedVictimIds,
                                                   int maxVictims,
                                                   VictimOwnership ownership) {
        List<DecodeRequestSnapshot> candidates =
                lowerPriorityCandidates(envelope, ep, excludedVictimIds, true, ownership, false);
        candidates.sort(DECODE_KV_ORDER);
        List<DecodeRequestSnapshot> victims = new ArrayList<>();
        long cost = 0;
        long freedKv = 0;
        for (DecodeRequestSnapshot candidate : candidates) {
            if (freedKv >= kvDeficit) {
                break;
            }
            if (maxVictims > 0 && victims.size() >= maxVictims) {
                return DecodeVictimSet.fail("deficit_exceeds_max_victims");
            }
            victims.add(candidate);
            freedKv += candidate.kvTokens();
            cost += PriorityCostFunction.f(candidate.priority())
                    * PriorityCostFunction.g(candidate.phase())
                    * PriorityCostFunction.lengthWasteCost(candidate.kvTokens());
        }
        if (freedKv < kvDeficit) {
            return DecodeVictimSet.fail("insufficient_releasable_kv");
        }
        return new DecodeVictimSet(List.copyOf(victims),
                PriorityCostFunction.H_DECODE_KV_FULL * cost, freedKv, null);
    }

    /**
     * 3.3 + 10.1: only strictly lower-priority entries are candidates. The
     * base pool is the reserved (engine-unconfirmed) entries; both confirmed
     * layers join only behind the Phase 5 gate ({@code includeAccepted}), with
     * the same strict priority boundary. The stage comparator/cost makes
     * {@code ACCEPTED_NOT_RUNNING} cheaper than {@code RUNNING}.
     * Task40: no-priority entries (priority 0, legacy path) never qualify.
     * P1-3: slot selection additionally skips queued-phase reservations
     * ({@code excludeQueued}) — they hold no engine slot, so evicting them
     * cannot reduce a slot deficit; KV selection keeps them (their hard KV
     * is releasable either way).
     */
    private static List<DecodeRequestSnapshot> lowerPriorityCandidates(PriorityRequestEnvelope envelope,
                                                                       DecodeEndpointSnapshot ep,
                                                                       Set<Long> excludedVictimIds,
                                                                       boolean releasableKvOnly,
                                                                       VictimOwnership ownership,
                                                                       boolean excludeQueued) {
        List<DecodeRequestSnapshot> candidates = new ArrayList<>();
        for (DecodeRequestSnapshot entry : ep.reserved()) {
            boolean ownershipMatches = ownership == VictimOwnership.MASTER_LOCAL
                    ? entry.phase().isMasterQueued()
                    : entry.phase() == DecodeTaskPhase.ENGINE_MAY_HAVE_SEEN;
            if (ownershipMatches
                    && entry.priorityKnown()
                    && PriorityNormalizer.hasPriority(entry.priority())
                    && entry.priority() < envelope.priority()
                    && !excludedVictimIds.contains(entry.requestId())
                    && (!releasableKvOnly || entry.kvTokens() > 0)
                    && (!excludeQueued || !entry.queued())) {
                candidates.add(entry);
            }
        }
        if (ownership == VictimOwnership.ENGINE_CANCEL) {
            addConfirmedCandidates(candidates, ep.accepted(), envelope,
                    excludedVictimIds, releasableKvOnly);
            addConfirmedCandidates(candidates, ep.running(), envelope,
                    excludedVictimIds, releasableKvOnly);
        }
        return candidates;
    }

    private static void addConfirmedCandidates(List<DecodeRequestSnapshot> candidates,
                                               List<DecodeRequestSnapshot> entries,
                                               PriorityRequestEnvelope envelope,
                                               Set<Long> excludedVictimIds,
                                               boolean releasableKvOnly) {
        for (DecodeRequestSnapshot entry : entries) {
            if (entry.phase().isEngineConfirmed()
                    && entry.priorityKnown()
                    && PriorityNormalizer.hasPriority(entry.priority())
                    && entry.priority() < envelope.priority()
                    && !excludedVictimIds.contains(entry.requestId())
                    && (!releasableKvOnly || entry.kvTokens() > 0)) {
                candidates.add(entry);
            }
        }
    }

    private static Set<Long> victimIds(List<DecodeRequestSnapshot> victims) {
        Set<Long> ids = new java.util.HashSet<>(victims.size());
        for (DecodeRequestSnapshot victim : victims) {
            ids.add(victim.requestId());
        }
        return ids;
    }

    /** P1-3: victims that actually hold an engine slot (non-queued). */
    private static long nonQueuedCount(List<DecodeRequestSnapshot> victims) {
        long count = 0;
        for (DecodeRequestSnapshot victim : victims) {
            if (!victim.queued()) {
                count++;
            }
        }
        return count;
    }

    /** Remaining victim budget for the second part of a combined plan. */
    private static int remainingBudget(int maxVictims, DecodeVictimSet first) {
        return maxVictims > 0 ? Math.max(1, maxVictims - first.victims().size()) : 0;
    }

    private static DecodeEvictionProposal buildDecodeProposal(DecodeEndpointSnapshot ep,
                                                              String evictionCase,
                                                              List<DecodeRequestSnapshot> victims,
                                                              long totalCost,
                                                              long freedKvTokens) {
        int minVictimPriority = Integer.MAX_VALUE;
        long latestVictimDeadline = Long.MIN_VALUE;
        long tieBreak = Long.MAX_VALUE;
        for (DecodeRequestSnapshot victim : victims) {
            minVictimPriority = Math.min(minVictimPriority, victim.priority());
            latestVictimDeadline = Math.max(latestVictimDeadline, victim.deadlineMs());
            tieBreak = Math.min(tieBreak, victim.requestId());
        }
        PlanCost cost = new PlanCost(minVictimPriority, totalCost, victims.size(),
                0, latestVictimDeadline, tieBreak);
        return new DecodeEvictionProposal(ep.endpointId(), ep.admissionVersion(),
                List.copyOf(victims), evictionCase, totalCost, freedKvTokens, cost);
    }

    /**
     * Victim selection outcome for one dimension of a decode plan: either the
     * victims plus their h-weighted cost part, or a failure reason.
     */
    private record DecodeVictimSet(List<DecodeRequestSnapshot> victims,
                                   long weightedCost,
                                   long freedKvTokens,
                                   String failReason) {

        static final DecodeVictimSet EMPTY = new DecodeVictimSet(List.of(), 0, 0, null);

        static DecodeVictimSet fail(String reason) {
            return new DecodeVictimSet(null, 0, 0, reason);
        }

        boolean ok() {
            return failReason == null;
        }
    }
}
