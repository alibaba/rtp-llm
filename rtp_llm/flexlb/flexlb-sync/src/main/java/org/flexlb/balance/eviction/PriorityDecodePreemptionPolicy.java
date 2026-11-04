package org.flexlb.balance.eviction;

import com.google.common.math.LongMath;
import org.flexlb.balance.endpoint.DecodeResources.AdmissionCapacity;
import org.flexlb.balance.endpoint.DecodeResources.CapacityDeficit;
import org.flexlb.balance.endpoint.DecodeResources.CapacityRelease;
import org.flexlb.balance.endpoint.DecodeResources.DecodeRequestView;
import org.flexlb.balance.endpoint.DecodeResources.ResourceSnapshot;
import org.flexlb.balance.eviction.DecodePreemptionPolicy.DecodeEvictionProposal;
import org.flexlb.balance.eviction.PriorityDecodePreemptionPolicy.PriorityHarmProfile;
import org.flexlb.config.PreemptionConfig;
import org.flexlb.config.VictimStage;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.enums.DecodeTaskPhase;
import org.flexlb.util.PriorityNormalizer;

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static com.google.common.base.Preconditions.checkArgument;

/** Pure victim selection for the selected Decode generation. */
public final class PriorityDecodePreemptionPolicy implements DecodePreemptionPolicy {

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
    static final Comparator<DecodeRequestView> DECODE_SLOT_ORDER =
            Comparator.comparingInt(DecodeRequestView::priority)
                    .thenComparingInt(v -> v.phase().ordinal())
                    .thenComparingLong(DecodeRequestView::requestId);

    /**
     * Candidate preference for KV eviction (design doc 12.4): priority asc →
     * stage asc (reserved before accepted) → kvBucket desc (bigger
     * releases first, fewer victims) → requestId asc.
     */
    static final Comparator<DecodeRequestView> DECODE_KV_ORDER =
            Comparator.comparingInt(DecodeRequestView::priority)
                    .thenComparingInt(v -> v.phase().ordinal())
                    .thenComparing(
                            v -> PriorityCostFunction.kvBucket(v.kvTokens()),
                            Comparator.reverseOrder())
                    .thenComparingLong(DecodeRequestView::requestId);

    /** Plan the cheapest single-ownership eviction for this snapshot's capacity deficit. */
    @Override
    public DecodeEvictionDecision planDecode(
            Request request,
            int incomingPriority,
            AdmissionCapacity capacity,
            ResourceSnapshot ep,
            PreemptionConfig preemption,
            Map<String, String> failures) {
        var deficit = deficit(request, capacity, ep, CapacityRelease.NONE);
        if (deficit.fits()) {
            failures.put(ep.routing().address(), "decode_capacity_sufficient");
            return new DecodeEvictionDecision(deficit, null);
        }
        boolean localEvictionEnabled =
                preemption != null && preemption.allows(VictimStage.DECODE_RESERVED);
        boolean engineCancelEnabled =
                preemption != null && preemption.allows(VictimStage.DECODE_ENGINE_OWNED);
        Selection local =
                localEvictionEnabled
                        ? planDecodeOneOwnership(
                                request,
                                capacity,
                                ep,
                                deficit,
                                candidates(ep, incomingPriority, VictimOwnership.MASTER_LOCAL),
                                failures)
                        : null;
        Selection engine =
                engineCancelEnabled
                        ? planDecodeOneOwnership(
                                request,
                                capacity,
                                ep,
                                deficit,
                                candidates(ep, incomingPriority, VictimOwnership.ENGINE_CANCEL),
                                failures)
                        : null;
        Selection proposal = cheaper(local, engine);
        return new DecodeEvictionDecision(deficit, proposal == null ? null : proposal.result());
    }

    /** Try slot and KV prefixes separately, then a disjoint combination if neither fits. */
    private static Selection planDecodeOneOwnership(
            Request request,
            AdmissionCapacity capacity,
            ResourceSnapshot ep,
            CapacityDeficit deficit,
            List<DecodeRequestView> candidates,
            Map<String, String> failures) {
        var kvCandidates =
                deficit.requests() > 0L && deficit.needsKv()
                        ? new ArrayList<>(candidates)
                        : candidates;
        var slots =
                deficit.requests() > 0L
                        ? selectVictims(
                                request, capacity, ep, CapacityRelease.NONE, candidates, false)
                        : null;
        var kv =
                deficit.needsKv()
                        ? selectVictims(
                                request, capacity, ep, CapacityRelease.NONE, kvCandidates, true)
                        : null;
        var single =
                cheaper(
                        proposalIfFits(
                                request, capacity, ep, slots, DecodeEvictionProposal.CASE_SLOT),
                        proposalIfFits(request, capacity, ep, kv, DecodeEvictionProposal.CASE_KV));
        if (single != null) return single;
        var combined =
                slots == null
                        ? kv
                        : kv == null
                                ? slots
                                : combineVictims(request, capacity, ep, slots, kv, candidates);
        var result =
                proposalIfFits(
                        request, capacity, ep, combined, DecodeEvictionProposal.CASE_SLOT_AND_KV);
        if (result == null)
            failures.put(
                    ep.routing().address(),
                    combined.ok() ? "insufficient_releasable_capacity" : combined.failReason());
        return result;
    }

    private static DecodeVictimSet combineVictims(
            Request request,
            AdmissionCapacity capacity,
            ResourceSnapshot ep,
            DecodeVictimSet slots,
            DecodeVictimSet kv,
            List<DecodeRequestView> candidates) {
        var required = deficit(request, capacity, ep, CapacityRelease.NONE);
        double slotPressure =
                (double) required.requests() / Math.max(1L, capacity.maxEngineRequests());
        double kvPressure =
                (double) required.kvTokens()
                        / Math.max(
                                1L,
                                capacity.kvBudget(ep.routing().placementUsage().totalKvTokens()));
        var first = kvPressure >= slotPressure ? kv : slots;
        if (!first.ok()) return first;
        Set<Long> excluded = new java.util.HashSet<>(first.victims().size());
        for (var victim : first.victims()) excluded.add(victim.requestId());
        var remaining = new ArrayList<>(candidates);
        remaining.removeIf(victim -> excluded.contains(victim.requestId()));
        var second =
                selectVictims(
                        request,
                        capacity,
                        ep,
                        first.release(),
                        remaining,
                        kvPressure < slotPressure);
        if (!second.ok()) return second;
        var victims = new ArrayList<>(first.victims());
        victims.addAll(second.victims());
        return new DecodeVictimSet(
                victims,
                first.harmProfile().plus(second.harmProfile()),
                first.release().plus(second.release()),
                null);
    }

    private static Selection proposalIfFits(
            Request request,
            AdmissionCapacity capacity,
            ResourceSnapshot ep,
            DecodeVictimSet victims,
            String evictionCase) {
        if (victims == null
                || !victims.ok()
                || !deficit(request, capacity, ep, victims.release()).fits()) return null;
        long tieBreak = Long.MAX_VALUE;
        for (var victim : victims.victims()) tieBreak = Math.min(tieBreak, victim.requestId());
        return new Selection(
                ep.routing().address(),
                victims.victims(),
                evictionCase,
                victims.freedKvTokens(),
                victims.harmProfile(),
                tieBreak);
    }

    private static Selection cheaper(Selection first, Selection second) {
        return first == null
                ? second
                : second == null
                        ? first
                        : Selection.ORDER.compare(first, second) <= 0 ? first : second;
    }

    private static CapacityDeficit deficit(
            Request request,
            AdmissionCapacity capacity,
            ResourceSnapshot ep,
            CapacityRelease release) {
        return capacity.evaluate(
                ep.routing().placementUsage(),
                request.getHardKvTokens(),
                request.getExpectedKvTokens(),
                release);
    }

    private static List<DecodeRequestView> candidates(
            ResourceSnapshot ep, int incomingPriority, VictimOwnership ownership) {
        List<DecodeRequestView> candidates = new ArrayList<>();
        for (var entry : ep.requests().values()) {
            boolean matches =
                    ownership == VictimOwnership.MASTER_LOCAL
                            ? entry.phase().isMasterQueued()
                            : entry.phase().requiresEngineCancel();
            if (matches
                    && !entry.claimedForPreemption()
                    && entry.priorityKnown()
                    && PriorityNormalizer.hasPriority(entry.priority())
                    && entry.priority() < incomingPriority) candidates.add(entry);
        }
        return candidates;
    }

    /** Mutate this private candidate list into the selected ordered prefix. */
    private static DecodeVictimSet selectVictims(
            Request request,
            AdmissionCapacity capacity,
            ResourceSnapshot ep,
            CapacityRelease priorRelease,
            List<DecodeRequestView> candidates,
            boolean reclaimKv) {
        if (reclaimKv) candidates.removeIf(victim -> victim.expectedKvTokens() <= 0L);
        long requiredSlots = deficit(request, capacity, ep, priorRelease).requests();
        if (!reclaimKv && candidates.size() < requiredSlots)
            return DecodeVictimSet.fail("insufficient_lower_priority_candidates");
        candidates.sort(reclaimKv ? DECODE_KV_ORDER : DECODE_SLOT_ORDER);
        int limit = reclaimKv ? candidates.size() : (int) requiredSlots;
        long caseWeight =
                reclaimKv
                        ? PriorityCostFunction.H_DECODE_KV_FULL
                        : PriorityCostFunction.H_DECODE_SLOT_FULL;
        CapacityRelease release = CapacityRelease.NONE;
        var harm = PriorityHarmProfile.builder();
        int selected = 0;
        while (selected < limit) {
            if (reclaimKv && !deficit(request, capacity, ep, priorRelease.plus(release)).needsKv())
                break;
            var victim = candidates.get(selected++);
            long length = reclaimKv ? PriorityCostFunction.lengthWasteCost(victim.kvTokens()) : 1L;
            harm.add(
                    victim.priority(),
                    BigInteger.valueOf(caseWeight)
                            .multiply(BigInteger.valueOf(PriorityCostFunction.g(victim.phase())))
                            .multiply(BigInteger.valueOf(length)));
            release = release.plus(victim.placementRelease());
        }
        if (reclaimKv && deficit(request, capacity, ep, priorRelease.plus(release)).needsKv())
            return DecodeVictimSet.fail("insufficient_releasable_kv");
        return new DecodeVictimSet(candidates.subList(0, selected), harm.build(), release, null);
    }

    /**
     * Victim selection outcome for one dimension of a decode plan: either the
     * victims plus their h-weighted cost part, or a failure reason.
     */
    private record DecodeVictimSet(
            List<DecodeRequestView> victims,
            PriorityHarmProfile harmProfile,
            CapacityRelease release,
            String failReason) {

        static DecodeVictimSet fail(String reason) {
            return new DecodeVictimSet(
                    null, PriorityHarmProfile.empty(), CapacityRelease.NONE, reason);
        }

        long freedKvTokens() {
            return release.hardKvTokens();
        }

        boolean ok() {
            return failReason == null;
        }
    }

    /** Default policy ranking stays private; custom policies return only their chosen plan. */
    private record Selection(
            String endpointId,
            List<DecodeRequestView> victims,
            String evictionCase,
            long freedKvTokens,
            PriorityHarmProfile harm,
            long tieBreak) {
        private static final Comparator<Selection> ORDER =
                Comparator.comparing(Selection::harm)
                        .thenComparingInt(value -> value.victims().size())
                        .thenComparingLong(Selection::tieBreak)
                        .thenComparing(Selection::endpointId);

        DecodeEvictionProposal result() {
            return new DecodeEvictionProposal(
                    endpointId, victims, evictionCase, freedKvTokens, harm.totalCost());
        }
    }

    /**
     * Exact, overflow-free priority harm of an eviction plan.
     *
     * <p>Each valid priority (1-100) owns an independent bucket. Plans compare
     * buckets from priority 100 down to 1, so any positive harm at a higher
     * priority is worse than an arbitrary amount of harm at every lower
     * priority. The value in a bucket retains the existing case/stage/length
     * weighting; {@link BigInteger} prevents a large victim set from wrapping.
     */
    public static final class PriorityHarmProfile implements Comparable<PriorityHarmProfile> {

        private static final PriorityHarmProfile EMPTY =
                new PriorityHarmProfile(new BigInteger[PriorityNormalizer.MAX_PRIORITY + 1]);

        private final BigInteger[] harmByPriority;

        private PriorityHarmProfile(BigInteger[] harmByPriority) {
            this.harmByPriority = harmByPriority;
        }

        public static PriorityHarmProfile empty() {
            return EMPTY;
        }

        public static Builder builder() {
            return new Builder();
        }

        /** Add two profiles without losing precision. */
        public PriorityHarmProfile plus(PriorityHarmProfile other) {
            BigInteger[] sum = new BigInteger[PriorityNormalizer.MAX_PRIORITY + 1];
            for (int priority = PriorityNormalizer.MIN_PRIORITY;
                    priority <= PriorityNormalizer.MAX_PRIORITY;
                    priority++) {
                BigInteger left = harmByPriority[priority];
                BigInteger right = other.harmByPriority[priority];
                if (left != null || right != null) {
                    sum[priority] = valueOrZero(left).add(valueOrZero(right));
                }
            }
            return new PriorityHarmProfile(sum);
        }

        /** Diagnostic scalar derived from exact weighted harm; plan ordering stays lexicographic. */
        public long totalCost() {
            BigInteger cost = BigInteger.ZERO;
            BigInteger maximum = BigInteger.valueOf(Long.MAX_VALUE);
            for (int priority = PriorityNormalizer.MIN_PRIORITY;
                    priority <= PriorityNormalizer.MAX_PRIORITY;
                    priority++) {
                if (harmByPriority[priority] != null) {
                    cost =
                            cost.add(
                                    harmByPriority[priority].multiply(
                                            BigInteger.valueOf(PriorityCostFunction.f(priority))));
                    if (cost.compareTo(maximum) >= 0) {
                        return Long.MAX_VALUE;
                    }
                }
            }
            return cost.longValueExact();
        }

        /**
         * Smaller harm is preferable. The highest exact priority at which two
         * plans differ decides the result; lower-priority buckets are considered
         * only when every higher-priority bucket is equal.
         */
        @Override
        public int compareTo(PriorityHarmProfile other) {
            for (int priority = PriorityNormalizer.MAX_PRIORITY;
                    priority >= PriorityNormalizer.MIN_PRIORITY;
                    priority--) {
                int comparison =
                        valueOrZero(harmByPriority[priority])
                                .compareTo(valueOrZero(other.harmByPriority[priority]));
                if (comparison != 0) {
                    return comparison;
                }
            }
            return 0;
        }

        @Override
        public boolean equals(Object object) {
            return object instanceof PriorityHarmProfile other
                    && Arrays.equals(harmByPriority, other.harmByPriority);
        }

        @Override
        public int hashCode() {
            return Arrays.hashCode(harmByPriority);
        }

        private static BigInteger valueOrZero(BigInteger value) {
            return value == null ? BigInteger.ZERO : value;
        }

        /** Mutable accumulator used only while one immutable plan is built. */
        public static final class Builder {

            private final BigInteger[] harmByPriority =
                    new BigInteger[PriorityNormalizer.MAX_PRIORITY + 1];

            private Builder() {}

            public Builder add(int priority, BigInteger harm) {
                checkArgument(
                        PriorityNormalizer.isValid(priority),
                        "priority must be in ["
                                + PriorityNormalizer.MIN_PRIORITY
                                + ", "
                                + PriorityNormalizer.MAX_PRIORITY
                                + "]: %s",
                        priority);
                if (harm.signum() < 0) {
                    throw new IllegalArgumentException("harm must be non-negative: " + harm);
                }
                if (harm.signum() != 0) {
                    harmByPriority[priority] = valueOrZero(harmByPriority[priority]).add(harm);
                }
                return this;
            }

            public PriorityHarmProfile build() {
                return new PriorityHarmProfile(
                        Arrays.copyOf(harmByPriority, harmByPriority.length));
            }
        }
    }

    /** Case, stage and length costs. Scalar priority cost is diagnostic;
     * {@link PriorityHarmProfile} enforces exact priority ordering. */
    public static final class PriorityCostFunction {

        private static final int MIN_PRIORITY_RANK = 0;
        private static final int MAX_PRIORITY_RANK = 4;
        private static final int PRIORITY_RANK_BASE = 30;
        private static final int PRIORITY_POINTS_PER_RANK = 10;
        private static final long COST_RADIX = 1_024L;
        private static final long KV_TOKENS_PER_COST_BUCKET = 1_024L;
        private static final long MASTER_QUEUED_STAGE_WEIGHT = 1L;
        private static final long ENGINE_MAY_HAVE_SEEN_STAGE_WEIGHT = 4L;
        private static final long ACCEPTED_STAGE_WEIGHT = 16L;
        private static final long RUNNING_STAGE_WEIGHT = 64L;

        // h(case) cross-type weights (design doc 7.6). A combined slot+KV plan
        // sums its already-weighted parts and is never multiplied again.
        /** h(DECODE_SLOT_FULL): frees concurrency, may affect D admission. */
        public static final long H_DECODE_SLOT_FULL = 4L;

        /** h(DECODE_KV_FULL): KV-sensitive, released KV needs confirmation. */
        public static final long H_DECODE_KV_FULL = 8L;

        private PriorityCostFunction() {}

        /** Rank 0..4 of a normalized priority (30..70), clamped for safety. */
        public static int rank(int priority) {
            return Math.clamp(
                    (priority - PRIORITY_RANK_BASE) / PRIORITY_POINTS_PER_RANK,
                    MIN_PRIORITY_RANK,
                    MAX_PRIORITY_RANK);
        }

        /** Single-value victim cost: 1024^rank. */
        public static long f(int priority) {
            return LongMath.pow(COST_RADIX, rank(priority));
        }

        /**
         * Stage multiplier g(stage) (design doc 11.4/12.5): deeper stages are
         * exponentially more expensive to evict.
         */
        public static long g(DecodeTaskPhase stage) {
            return switch (stage) {
                case LOCAL_RESERVED, MASTER_QUEUED_NOT_DISPATCHED -> MASTER_QUEUED_STAGE_WEIGHT;
                case ENGINE_MAY_HAVE_SEEN -> ENGINE_MAY_HAVE_SEEN_STAGE_WEIGHT;
                case ACCEPTED_NOT_RUNNING -> ACCEPTED_STAGE_WEIGHT;
                case RUNNING -> RUNNING_STAGE_WEIGHT;
            };
        }

        /** KV bucket of a reservation: ceil(kvTokens / 1024) (design doc 12.3). */
        public static long kvBucket(long kvTokens) {
            return Math.ceilDiv(Math.max(0L, kvTokens), KV_TOKENS_PER_COST_BUCKET);
        }

        /**
         * Length waste factor for KV eviction (design doc 12.5):
         * {@code max(1, sqrt(kvBucket))}, rounded to long — large releases sort
         * first but must not dramatically shrink cross-priority costs.
         */
        public static long lengthWasteCost(long kvTokens) {
            return Math.max(1L, Math.round(Math.sqrt((double) kvBucket(kvTokens))));
        }
    }
}
