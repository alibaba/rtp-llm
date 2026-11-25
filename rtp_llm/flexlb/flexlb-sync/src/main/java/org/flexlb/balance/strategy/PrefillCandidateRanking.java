package org.flexlb.balance.strategy;

import org.flexlb.balance.AdmissionVersion;
import org.flexlb.config.RoutingConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.util.Logger;

import java.util.BitSet;
import java.util.function.IntPredicate;

import static com.google.common.math.LongMath.saturatedAdd;

/** Ranks Prefill workers and proposes a redirect for validation when the target is reserved. */
final class PrefillCandidateRanking {
    private final EngineHealthReporter reporter;
    private final EndpointRoundRobin tieRotation = new EndpointRoundRobin();
    private final EndpointRoundRobin redirectRotation = new EndpointRoundRobin();

    PrefillCandidateRanking(EngineHealthReporter reporter) {
        this.reporter = reporter;
    }

    record Choice(int index, int fallback, boolean cacheFirst, String reason) {
        static Choice ordinary(int index, boolean cacheFirst, String reason) {
            return new Choice(index, index, cacheFirst, reason);
        }

        Choice redirectTo(int target) {
            return new Choice(target, index, cacheFirst, "HOTSPOT_RELEASE");
        }

        Choice rateLimited() {
            return ordinary(fallback, cacheFirst, "HOTSPOT_RATE_LIMIT");
        }

        boolean redirect() { return index != fallback; }

        long expectedVersion(PrefillCandidateSet candidates) {
            return cacheFirst && !redirect() ? AdmissionVersion.ANY : candidates.workload(index).admissionVersion();
        }
    }

    Choice select(PrefillCandidateSet candidates, RoleType role, String group, BalanceContext request) {
        if (candidates.size() == 0) {
            return Choice.ordinary(-1, false, null);
        }
        var prefill = request.getConfig().getRouter().getRoles().getPrefill();
        long inputTokens = request.getRequest().getSeqLen();
        var release = prefill.getHotspotRelease();
        boolean direct = request.getConfig().isDirect();
        return prefill.getTokenBalance() == null
                ? selectByTtft(candidates, role, group, inputTokens, prefill.getCacheAffinity(), release, direct)
                : selectTokenBalancedCandidate(candidates, role, group, inputTokens, prefill.getTokenBalance(), release, direct);
    }

    private static BitSet redirectTargets(PrefillCandidateSet candidates, int source, boolean direct) {
        BitSet targets = new BitSet(candidates.size());
        for (int i = 0; i < candidates.size(); i++) {
            if (candidates.mayRedirect(source, i) && candidates.admissionBlock(i) == null
                    && !candidates.endpoint(i).engineQueueAdmissionBlocked()
                    && (!direct || candidates.endpoint(i).canAcceptRequest())) {
                targets.set(i);
            }
        }
        return targets;
    }

    /** Cache leaders pin hot requests unless a detected hotspot can redirect under the shared limit. */
    private Choice selectTokenBalancedCandidate(PrefillCandidateSet survivors,
            RoleType roleType, String group, long seqLen,
            RoutingConfig.TokenBalanceConfig tokenBalance, RoutingConfig.HotspotReleaseConfig release, boolean direct) {
        boolean hotspotRelease = release.isEnabled();
        boolean cacheFirst = PrefillTokenBalance.cacheFirst(
                survivors.maximumCacheHit, seqLen, tokenBalance.getMinPrefixHitPercent());
        if (!cacheFirst) {
            int target = selectByTokenWork(survivors, roleType, group, seqLen, hotspotRelease,
                    i -> !survivors.endpoint(i).engineQueueAdmissionBlocked(), tieRotation);
            return Choice.ordinary(target, false, null);
        }

        BitSet leaders = new BitSet();
        BitSet openLeaders = new BitSet();
        for (int i = 0; i < survivors.size(); i++) {
            if (survivors.cacheHit(i) != survivors.maximumCacheHit) { continue; }
            leaders.set(i);
            if (!survivors.endpoint(i).engineQueueAdmissionBlocked()) { openLeaders.set(i); }
        }
        // Keep the original affinity choice as the fallback when the shared quota is exhausted.
        int source = selectTokenLeader(survivors, openLeaders.isEmpty() ? leaders : openLeaders,
                roleType, group, tieRotation);
        Choice fallback = Choice.ordinary(openLeaders.get(source) ? source : -1, true, null);
        if (!hotspotRelease || !survivors.isHotspot(source)) { return fallback; }

        BitSet targets = redirectTargets(survivors, source, direct);
        if (targets.isEmpty()) { return fallback; }

        leaders.and(targets);
        int selected = leaders.isEmpty()
                ? selectByTokenWork(survivors, roleType, group, seqLen, true, targets::get, redirectRotation)
                : selectTokenLeader(survivors, leaders, roleType, group, redirectRotation);
        return selected >= 0 ? fallback.redirectTo(selected) : fallback;
    }

    private static int selectTokenLeader(PrefillCandidateSet survivors, BitSet leaders, RoleType role,
                                         String group, EndpointRoundRobin rotation) {
        if (leaders.isEmpty()) { return -1; }
        long longest = 0L;
        for (int i = leaders.nextSetBit(0); i >= 0; i = leaders.nextSetBit(i + 1)) {
            longest = Math.max(longest, survivors.routingCacheMatchTokens(i));
        }
        long prefix = longest;
        return rotation.next(role, group, survivors.size(),
                i -> leaders.get(i) && survivors.routingCacheMatchTokens(i) == prefix, survivors::endpointAddress);
    }

    private static int selectByTokenWork(PrefillCandidateSet survivors, RoleType role, String group,
                                         long seqLen, boolean accountForHits, IntPredicate eligible, EndpointRoundRobin rotation) {
        double[] scores = new double[survivors.size()];
        double minimum = Double.POSITIVE_INFINITY;
        for (int i = 0; i < survivors.size(); i++) {
            if (!eligible.test(i)) {
                scores[i] = Double.POSITIVE_INFINITY;
                continue;
            }
            long tokens = accountForHits ? Math.max(0L, seqLen - survivors.cacheHit(i)) : seqLen;
            scores[i] = survivors.workload(i).score() + tokens;
            minimum = Math.min(minimum, scores[i]);
        }
        if (!Double.isFinite(minimum)) { return -1; }
        double score = minimum;
        return rotation.next(role, group, survivors.size(), i -> scores[i] == score, survivors::endpointAddress);
    }

    private record TtftChoice(int index, String reason) { }

    /** Apply the ordinary ranking first, then release its winner if that endpoint is hot. */
    private Choice selectByTtft(PrefillCandidateSet survivors, RoleType role, String group, long seqLen,
                             RoutingConfig.CacheAffinityConfig affinity, RoutingConfig.HotspotReleaseConfig release,
                             boolean direct) {
        BitSet eligible = new BitSet(survivors.size());
        eligible.set(0, survivors.size());
        TtftChoice choice = rankByTtft(survivors, eligible, role, group, seqLen, affinity,
                release.isEnabled(), tieRotation);
        int selected = choice.index();
        Choice ordinary = Choice.ordinary(selected, "CACHE_LEADER".equals(choice.reason()), choice.reason());
        if (release.isEnabled() && survivors.isHotspot(selected)) {
            BitSet targets = redirectTargets(survivors, selected, direct);
            if (!targets.isEmpty()) {
                int alternate = rankByTtft(survivors, targets, role, group, seqLen, affinity,
                        true, redirectRotation).index();
                return ordinary.redirectTo(alternate);
            }
        }
        return ordinary;
    }

    void report(PrefillCandidateSet survivors, RoleType role, String group, int selected, String reason) {
        if (reason != null) {
            reporter.reportCacheAffinityDecision(role, survivors.endpoint(selected).getIp(), reason);
            if (Logger.isDebugEnabled()) {
                Logger.debug("CostBasedPrefill cache-affinity decision - role: {}, group: {}, selected: {}, "
                                + "minProjectedTtftMs: {}, selectedProjectedTtftMs: {}, hitTokens: {}, reason: {}",
                        role, group, survivors.endpointAddress(selected), survivors.minimumProjectedTtftMs,
                        survivors.projectedTtftMs(selected), survivors.cacheHit(selected), reason);
            }
        }
    }

    private static TtftChoice rankByTtft(PrefillCandidateSet survivors, BitSet eligible, RoleType role, String group,
                                         long seqLen, RoutingConfig.CacheAffinityConfig affinity,
                                         boolean hotspotRelease, EndpointRoundRobin rotation) {
        int baseline = eligible.nextSetBit(0);
        for (int i = eligible.nextSetBit(baseline + 1); i >= 0; i = eligible.nextSetBit(i + 1)) {
            long time = survivors.projectedTtftMs(i);
            long bestTime = survivors.projectedTtftMs(baseline);
            if (time < bestTime || time == bestTime && affinity != null && survivors.compareCacheMatch(i, baseline) > 0) {
                baseline = i;
            }
        }
        String reason = null;
        if (affinity != null) {
            BitSet preferred = new BitSet(survivors.size());
            long cutoff = saturatedAdd(survivors.projectedTtftMs(baseline), Math.max(0L, affinity.getMaxExtraTtftMs()));
            double minimumHitRate = normalizedHitRate(affinity.getMinPrefixHitPercent());
            boolean strongerCache = false;
            boolean minimumHitRateMet = false;
            for (int i = eligible.nextSetBit(0); i >= 0; i = eligible.nextSetBit(i + 1)) {
                if (survivors.compareCacheMatch(i, baseline) <= 0) {
                    continue;
                }
                strongerCache = true;
                if (minimumHitRate > 0.0 && (seqLen <= 0L
                        || survivors.cacheHit(i) * RoutingConfig.PERCENTAGE_SCALE / seqLen < minimumHitRate)) {
                    continue;
                }
                minimumHitRateMet = true;
                if (hotspotRelease || survivors.projectedTtftMs(i) <= cutoff) {
                    preferred.set(i);
                }
            }
            if (!preferred.isEmpty()) {
                return new TtftChoice(selectCacheLeader(survivors, preferred, role, group, rotation), "CACHE_LEADER");
            }
            reason = "NO_CACHE_LEAD";
            if (minimumHitRateMet) {
                reason = "OVER_CAP";
            } else if (strongerCache) {
                reason = "LOW_CACHE_HIT";
            }
        }
        int reference = baseline;
        IntPredicate tied = i -> eligible.get(i) && survivors.projectedTtftMs(i) == survivors.projectedTtftMs(reference)
                && (affinity == null || survivors.compareCacheMatch(i, reference) == 0);
        int lightest = reference;
        boolean zeroDuration = survivors.prefillMs(reference) == 0;
        for (int i = eligible.nextSetBit(0); zeroDuration && i >= 0; i = eligible.nextSetBit(i + 1)) {
            if (!tied.test(i)) { continue; }
            if (survivors.prefillMs(i) != 0) {
                zeroDuration = false;
            } else if (compareOutstandingWork(survivors.workload(i), survivors.workload(lightest)) < 0) {
                lightest = i;
            }
        }
        // Sub-millisecond formulas can yield the same zero service estimate for a whole fleet.
        // Resolve these cost ties with already placed work, then rotate among exact equals.
        var preferredWork = zeroDuration ? survivors.workload(lightest) : null;
        int selected = rotation.next(role, group, survivors.size(),
                i -> tied.test(i) && (preferredWork == null || compareOutstandingWork(survivors.workload(i), preferredWork) == 0),
                survivors::endpointAddress);
        return new TtftChoice(selected, reason);
    }

    private static int compareOutstandingWork(PrefillTokenWorkload.Snapshot left, PrefillTokenWorkload.Snapshot right) {
        int tokens = Long.compare(saturatedAdd(left.pendingTokens(), left.plannedTokens()),
                saturatedAdd(right.pendingTokens(), right.plannedTokens()));
        return tokens != 0 ? tokens : Long.compare(saturatedAdd(left.observedRequests(), left.plannedRequests()),
                saturatedAdd(right.observedRequests(), right.plannedRequests()));
    }

    private static double normalizedHitRate(double configuredRate) {
        if (Double.isNaN(configuredRate)
                || configuredRate == Double.POSITIVE_INFINITY) {
            return RoutingConfig.PERCENTAGE_SCALE;
        }
        if (configuredRate == Double.NEGATIVE_INFINITY) {
            return 0.0;
        }
        return Math.min(
                RoutingConfig.PERCENTAGE_SCALE,
                Math.max(0.0, configuredRate));
    }

    /** Fairly rotate only endpoints with the same best cache hit and projected TTFT. */
    private static int selectCacheLeader(
            PrefillCandidateSet survivors, BitSet preferredCandidates, RoleType role, String group, EndpointRoundRobin rotation) {
        int best = -1;
        for (int candidateIndex = preferredCandidates.nextSetBit(0); candidateIndex >= 0;
             candidateIndex = preferredCandidates.nextSetBit(candidateIndex + 1)) {
            int cacheOrder = best < 0 ? 1 : survivors.compareCacheMatch(candidateIndex, best);
            if (cacheOrder > 0 || cacheOrder == 0
                    && survivors.projectedTtftMs(candidateIndex) < survivors.projectedTtftMs(best)) {
                best = candidateIndex;
            }
        }
        if (best < 0) {
            return -1;
        }
        int leader = best;
        return rotation.next(role, group, survivors.size(),
                i -> preferredCandidates.get(i) && survivors.compareCacheMatch(i, leader) == 0
                        && survivors.projectedTtftMs(i) == survivors.projectedTtftMs(leader),
                survivors::endpointAddress);
    }
}
