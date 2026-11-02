package org.flexlb.balance.eviction;

import org.flexlb.balance.endpoint.DecodeEndpoint.CapacityDeficit;
import org.flexlb.balance.endpoint.DecodeEndpoint.CapacityRelease;
import org.flexlb.balance.endpoint.DecodeEndpoint.DecodeRequestView;
import org.flexlb.config.PreemptionConfig;
import org.flexlb.config.VictimStage;
import org.flexlb.enums.DecodeTaskPhase;
import org.flexlb.util.PriorityNormalizer;

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Set;

/** Plans resource reclamation for the exact selected Decode endpoint. */
public final class EvictionPlanner {

    private EvictionPlanner() {
    }

    /** A decode plan never mixes Master-local removal with Engine Cancel. */
    private enum VictimOwnership {
        MASTER_LOCAL,
        ENGINE_CANCEL
    }

    // ==================== Decode reserved-only eviction (design doc 11-13) ====================

    /**
     * Candidate preference for slot eviction (design doc 11.3, first = evicted
     * first): priority asc → stage asc (reserved before accepted) →
     * requestId asc.
     */
    static final Comparator<DecodeRequestView> DECODE_SLOT_ORDER = Comparator
            .comparingInt(DecodeRequestView::priority)
            .thenComparingInt(v -> v.phase().ordinal())
            .thenComparingLong(DecodeRequestView::requestId);

    /**
     * Candidate preference for KV eviction (design doc 12.4): priority asc →
     * stage asc (reserved before accepted) → kvBucket desc (bigger
     * releases first, fewer victims) → requestId asc.
     */
    static final Comparator<DecodeRequestView> DECODE_KV_ORDER = Comparator
            .comparingInt(DecodeRequestView::priority)
            .thenComparingInt(v -> v.phase().ordinal())
            .thenComparing(v -> PriorityCostFunction.kvBucket(v.kvTokens()), Comparator.reverseOrder())
            .thenComparingLong(DecodeRequestView::requestId);

    /**
     * Plan the cheapest decode eviction that clears the incoming request's
     * slot and/or KV deficit across the given endpoints.
     *
     * <p>Candidates are the strictly lower-priority reserved entries when
     * {@link VictimStage#DECODE_RESERVED} is allowed, plus — only when
     * {@link VictimStage#DECODE_ENGINE_OWNED} is allowed and the endpoint's
     * engine supports the Cancel RPC — the strictly lower-priority
     * engine-confirmed accepted/running entries. Running entries use a larger
     * stage cost, so an otherwise equivalent accepted-not-running victim is
     * preferred.
     *
     * @param incomingPriority incoming request priority
     * @param hardKvTokens prompt KV demand
     * @param expectedKvTokens complete prompt and output KV demand
     * @param decodes  candidate decode endpoint snapshots
     * @param preemption immutable policy for the current admission attempt
     * @param channel  engine cancel channel for the per-endpoint support gate;
     *                 {@code null} disables confirmed layers entirely
     * @param failures out-param: per-endpoint infeasibility reason
     * @return the best proposal by {@link DecodeEvictionProposal#ORDER}, or
     *         {@code null} when no endpoint has a feasible plan
     */
    public static DecodeEvictionProposal planDecode(int incomingPriority, long hardKvTokens, long expectedKvTokens,
                                                    List<DecodeEndpointSnapshot> decodes,
                                                    PreemptionConfig preemption,
                                                    EngineCancelChannel channel,
                                                    Map<String, String> failures) {
        DecodeEvictionProposal best = null;
        for (DecodeEndpointSnapshot ep : decodes) {
            DecodeEvictionProposal proposal =
                    planDecodeOne(incomingPriority, hardKvTokens, expectedKvTokens, ep, preemption, channel, failures);
            if (proposal != null
                    && (best == null || DecodeEvictionProposal.ORDER.compare(proposal, best) < 0)) {
                best = proposal;
            }
        }
        return best;
    }

    /** The same capacity decision labels both ordinary admission and eviction planning. */
    public static String decodeEvictionCase(long hardKvTokens, long expectedKvTokens, DecodeEndpointSnapshot ep) {
        CapacityDeficit deficit = ep.capacity().evaluate(ep.usage(), hardKvTokens, expectedKvTokens);
        if (deficit.requests() > 0L && deficit.needsKv()) {
            return DecodeEvictionProposal.CASE_SLOT_AND_KV;
        }
        if (deficit.requests() > 0L) { return DecodeEvictionProposal.CASE_SLOT; }
        return deficit.needsKv() ? DecodeEvictionProposal.CASE_KV : null;
    }

    private static DecodeEvictionProposal planDecodeOne(int incomingPriority, long hardKvTokens, long expectedKvTokens,
                                                        DecodeEndpointSnapshot ep,
                                                        PreemptionConfig preemption,
                                                        EngineCancelChannel channel,
                                                        Map<String, String> failures) {
        CapacityDeficit deficit = ep.capacity().evaluate(ep.usage(), hardKvTokens, expectedKvTokens);
        if (deficit.fits()) {
            failures.put(ep.endpointId(), "decode_capacity_sufficient");
            return null;
        }
        boolean localEvictionEnabled = preemption != null
                && preemption.allows(VictimStage.DECODE_RESERVED);
        boolean engineCancelEnabled = preemption != null
                && preemption.allows(VictimStage.DECODE_ENGINE_OWNED)
                && channel != null && channel.isSupported(ep.endpoint());
        DecodeEvictionProposal local = localEvictionEnabled
                ? planDecodeOneOwnership(incomingPriority, hardKvTokens, expectedKvTokens, ep, deficit,
                        VictimOwnership.MASTER_LOCAL, failures) : null;
        DecodeEvictionProposal engine = engineCancelEnabled
                ? planDecodeOneOwnership(incomingPriority, hardKvTokens, expectedKvTokens, ep, deficit,
                        VictimOwnership.ENGINE_CANCEL, failures) : null;
        if (local == null) { return engine; }
        if (engine == null) { return local; }
        return DecodeEvictionProposal.ORDER.compare(local, engine) <= 0 ? local : engine;
    }

    private static DecodeEvictionProposal planDecodeOneOwnership(
            int incomingPriority, long hardKvTokens, long expectedKvTokens, DecodeEndpointSnapshot ep, CapacityDeficit deficit,
            VictimOwnership ownership, Map<String, String> failures) {
        if (deficit.requests() > 0L && deficit.needsKv()) {
            return planDecodeCombined(incomingPriority, hardKvTokens, expectedKvTokens, ep, deficit, ownership, failures);
        }
        DecodeVictimSet set = deficit.requests() > 0L
                ? selectSlotVictims(incomingPriority, ep, deficit.requests(), Set.of(), ownership)
                : selectKvVictims(incomingPriority, hardKvTokens, expectedKvTokens, ep, CapacityRelease.NONE, Set.of(), ownership);
        if (!set.ok()) {
            failures.put(ep.endpointId(), set.failReason());
            return null;
        }
        if (!ep.capacity().evaluate(ep.usage(), hardKvTokens, expectedKvTokens, set.release()).fits()) {
            failures.put(ep.endpointId(), "insufficient_releasable_capacity");
            return null;
        }
        return buildDecodeProposal(ep, deficit.requests() > 0L
                        ? DecodeEvictionProposal.CASE_SLOT : DecodeEvictionProposal.CASE_KV,
                set.victims(), set.harmProfile(), set.weightedCost(), set.freedKvTokens());
    }

    /** Preserve the cost ordering while testing every proposal against all capacity dimensions. */
    private static DecodeEvictionProposal planDecodeCombined(
            int incomingPriority, long hardKvTokens, long expectedKvTokens, DecodeEndpointSnapshot ep, CapacityDeficit deficit,
            VictimOwnership ownership, Map<String, String> failures) {
        DecodeVictimSet slotOnly = selectSlotVictims(incomingPriority, ep, deficit.requests(), Set.of(), ownership);
        DecodeEvictionProposal slotSide = slotOnly.ok()
                && ep.capacity().evaluate(ep.usage(), hardKvTokens, expectedKvTokens, slotOnly.release()).fits()
                ? buildDecodeProposal(ep, DecodeEvictionProposal.CASE_SLOT, slotOnly.victims(),
                        slotOnly.harmProfile(), slotOnly.weightedCost(), slotOnly.freedKvTokens()) : null;
        DecodeVictimSet kvOnly = selectKvVictims(incomingPriority, hardKvTokens, expectedKvTokens, ep, CapacityRelease.NONE, Set.of(), ownership);
        DecodeEvictionProposal kvSide = kvOnly.ok()
                && ep.capacity().evaluate(ep.usage(), hardKvTokens, expectedKvTokens, kvOnly.release()).fits()
                ? buildDecodeProposal(ep, DecodeEvictionProposal.CASE_KV, kvOnly.victims(),
                        kvOnly.harmProfile(), kvOnly.weightedCost(), kvOnly.freedKvTokens()) : null;
        if (slotSide != null || kvSide != null) {
            if (slotSide == null) { return kvSide; }
            if (kvSide == null) { return slotSide; }
            return DecodeEvictionProposal.ORDER.compare(slotSide, kvSide) <= 0 ? slotSide : kvSide;
        }

        double slotPressure = (double) deficit.requests() / Math.max(1L, ep.capacity().maxEngineRequests());
        double kvPressure = (double) deficit.kvTokens() / Math.max(1L,
                ep.capacity().kvBudget(ep.usage().totalKvTokens()));
        DecodeVictimSet first = kvPressure >= slotPressure ? kvOnly : slotOnly;
        if (!first.ok()) {
            failures.put(ep.endpointId(), first.failReason());
            return null;
        }
        CapacityDeficit remaining = ep.capacity().evaluate(ep.usage(), hardKvTokens, expectedKvTokens, first.release());
        DecodeVictimSet second = kvPressure >= slotPressure
                ? selectSlotVictims(incomingPriority, ep, remaining.requests(), victimIds(first.victims()), ownership)
                : selectKvVictims(incomingPriority, hardKvTokens, expectedKvTokens, ep, first.release(), victimIds(first.victims()), ownership);
        if (!second.ok()) {
            failures.put(ep.endpointId(), second.failReason());
            return null;
        }
        CapacityRelease release = first.release().plus(second.release());
        if (!ep.capacity().evaluate(ep.usage(), hardKvTokens, expectedKvTokens, release).fits()) {
            failures.put(ep.endpointId(), "insufficient_releasable_capacity");
            return null;
        }
        List<DecodeRequestView> victims = new ArrayList<>(first.victims());
        victims.addAll(second.victims());
        return buildDecodeProposal(ep, DecodeEvictionProposal.CASE_SLOT_AND_KV, victims,
                first.harmProfile().plus(second.harmProfile()),
                PriorityCostFunction.saturatedAdd(first.weightedCost(), second.weightedCost()),
                release.hardKvTokens());
    }

    private static DecodeVictimSet selectSlotVictims(
            int incomingPriority, DecodeEndpointSnapshot ep, long deficit,
            Set<Long> excludedVictimIds, VictimOwnership ownership) {
        List<DecodeRequestView> candidates = lowerPriorityCandidates(
                incomingPriority, ep, excludedVictimIds, false, ownership);
        if (candidates.size() < deficit) {
            return DecodeVictimSet.fail("insufficient_lower_priority_candidates");
        }
        candidates.sort(DECODE_SLOT_ORDER);
        List<DecodeRequestView> victims = candidates.subList(0, (int) deficit);
        long cost = 0L;
        CapacityRelease release = CapacityRelease.NONE;
        PriorityHarmProfile.Builder harmProfile = PriorityHarmProfile.builder();
        for (DecodeRequestView victim : victims) {
            long stageCost = PriorityCostFunction.g(victim.phase());
            cost = PriorityCostFunction.saturatedAdd(cost, PriorityCostFunction.saturatedMultiply(
                    PriorityCostFunction.f(victim.priority()), stageCost));
            harmProfile.add(victim.priority(), BigInteger.valueOf(PriorityCostFunction.H_DECODE_SLOT_FULL)
                    .multiply(BigInteger.valueOf(stageCost)));
            release = release.plus(victim.placementRelease());
        }
        return new DecodeVictimSet(victims, harmProfile.build(),
                PriorityCostFunction.saturatedMultiply(PriorityCostFunction.H_DECODE_SLOT_FULL, cost),
                release, null);
    }

    /** Both physical prompt supply and the complete-output budget must be satisfied. */
    private static DecodeVictimSet selectKvVictims(
            int incomingPriority, long hardKvTokens, long expectedKvTokens, DecodeEndpointSnapshot ep, CapacityRelease priorRelease,
            Set<Long> excludedVictimIds, VictimOwnership ownership) {
        List<DecodeRequestView> candidates = lowerPriorityCandidates(
                incomingPriority, ep, excludedVictimIds, true, ownership);
        candidates.sort(DECODE_KV_ORDER);
        List<DecodeRequestView> victims = new ArrayList<>();
        long cost = 0L;
        CapacityRelease release = CapacityRelease.NONE;
        PriorityHarmProfile.Builder harmProfile = PriorityHarmProfile.builder();
        for (DecodeRequestView candidate : candidates) {
            if (!ep.capacity().evaluate(ep.usage(), hardKvTokens, expectedKvTokens, priorRelease.plus(release)).needsKv()) { break; }
            victims.add(candidate);
            release = release.plus(candidate.placementRelease());
            long stageCost = PriorityCostFunction.g(candidate.phase());
            long lengthCost = PriorityCostFunction.lengthWasteCost(candidate.kvTokens());
            long victimCost = PriorityCostFunction.saturatedMultiply(PriorityCostFunction.saturatedMultiply(
                    PriorityCostFunction.f(candidate.priority()), stageCost), lengthCost);
            cost = PriorityCostFunction.saturatedAdd(cost, victimCost);
            harmProfile.add(candidate.priority(), BigInteger.valueOf(PriorityCostFunction.H_DECODE_KV_FULL)
                    .multiply(BigInteger.valueOf(stageCost)).multiply(BigInteger.valueOf(lengthCost)));
        }
        if (ep.capacity().evaluate(ep.usage(), hardKvTokens, expectedKvTokens, priorRelease.plus(release)).needsKv()) {
            return DecodeVictimSet.fail("insufficient_releasable_kv");
        }
        return new DecodeVictimSet(victims, harmProfile.build(),
                PriorityCostFunction.saturatedMultiply(PriorityCostFunction.H_DECODE_KV_FULL, cost), release, null);
    }

    /**
     * Only strictly lower-priority entries are candidates. The
     * base pool is the reserved (engine-unconfirmed) entries; both confirmed
     * layers join only when engine-owned eviction is enabled, with
     * the same strict priority boundary. The stage comparator/cost makes
     * {@code ACCEPTED_NOT_RUNNING} cheaper than {@code RUNNING}.
     * Priority-neutral entries (priority 0) never qualify.
     * Placement counts queued reservations as well as Engine-facing owners;
     * either ownership class releases its placement request charge.
     */
    private static List<DecodeRequestView> lowerPriorityCandidates(int incomingPriority,
                                                                       DecodeEndpointSnapshot ep,
                                                                       Set<Long> excludedVictimIds,
                                                                       boolean releasableKvOnly,
                                                                       VictimOwnership ownership) {
        List<DecodeRequestView> candidates = new ArrayList<>();
        for (DecodeRequestView entry : ep.reserved()) {
            boolean ownershipMatches = ownership == VictimOwnership.MASTER_LOCAL
                    ? entry.phase().isMasterQueued()
                    : entry.phase() == DecodeTaskPhase.ENGINE_MAY_HAVE_SEEN;
            if (ownershipMatches
                    && entry.priorityKnown()
                    && PriorityNormalizer.hasPriority(entry.priority())
                    && entry.priority() < incomingPriority
                    && !excludedVictimIds.contains(entry.requestId())
                    && (!releasableKvOnly || entry.placementRelease().expectedKvTokens() > 0L)) {
                candidates.add(entry);
            }
        }
        if (ownership == VictimOwnership.ENGINE_CANCEL) {
            addConfirmedCandidates(candidates, ep.accepted(), incomingPriority,
                    excludedVictimIds, releasableKvOnly);
            addConfirmedCandidates(candidates, ep.running(), incomingPriority,
                    excludedVictimIds, releasableKvOnly);
        }
        return candidates;
    }

    private static void addConfirmedCandidates(List<DecodeRequestView> candidates,
                                               List<DecodeRequestView> entries,
                                               int incomingPriority,
                                               Set<Long> excludedVictimIds,
                                               boolean releasableKvOnly) {
        for (DecodeRequestView entry : entries) {
            if (entry.phase().isEngineConfirmed()
                    && entry.priorityKnown()
                    && PriorityNormalizer.hasPriority(entry.priority())
                    && entry.priority() < incomingPriority
                    && !excludedVictimIds.contains(entry.requestId())
                    && (!releasableKvOnly || entry.placementRelease().expectedKvTokens() > 0L)) {
                candidates.add(entry);
            }
        }
    }

    private static Set<Long> victimIds(List<DecodeRequestView> victims) {
        Set<Long> ids = new java.util.HashSet<>(victims.size());
        for (DecodeRequestView victim : victims) {
            ids.add(victim.requestId());
        }
        return ids;
    }

    private static DecodeEvictionProposal buildDecodeProposal(DecodeEndpointSnapshot ep,
                                                              String evictionCase,
                                                              List<DecodeRequestView> victims,
                                                              PriorityHarmProfile harmProfile,
                                                              long totalCost,
                                                              long freedKvTokens) {
        long tieBreak = Long.MAX_VALUE;
        for (DecodeRequestView victim : victims) {
            tieBreak = Math.min(tieBreak, victim.requestId());
        }
        PlanCost cost = new PlanCost(
                harmProfile, victims.size(), tieBreak);
        return new DecodeEvictionProposal(ep.endpointId(),
                victims, evictionCase, totalCost, freedKvTokens, cost);
    }

    /**
     * Victim selection outcome for one dimension of a decode plan: either the
     * victims plus their h-weighted cost part, or a failure reason.
     */
    private record DecodeVictimSet(List<DecodeRequestView> victims,
                                   PriorityHarmProfile harmProfile,
                                   long weightedCost,
                                   CapacityRelease release,
                                   String failReason) {

        static DecodeVictimSet fail(String reason) {
            return new DecodeVictimSet(null, PriorityHarmProfile.empty(), 0, CapacityRelease.NONE, reason);
        }

        long freedKvTokens() { return release.hardKvTokens(); }

        boolean ok() {
            return failReason == null;
        }
    }
}
