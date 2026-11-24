package org.flexlb.balance.strategy;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.config.RoutingConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.util.Logger;

import java.util.BitSet;

/** Rank one projection snapshot. Both policies spend from the same Master-wide redirect budget. */
final class PrefillCandidateRanking {
    private final EngineHealthReporter reporter;
    private final HotspotRedirectLimiter redirectLimiter;
    private final EndpointRoundRobin tieRotation = new EndpointRoundRobin();
    private final EndpointRoundRobin redirectRotation = new EndpointRoundRobin();

    PrefillCandidateRanking(EngineHealthReporter reporter, HotspotRedirectLimiter redirectLimiter) {
        this.reporter = reporter;
        this.redirectLimiter = redirectLimiter;
    }

    int select(PrefillCandidateSet candidates, RoleType role, String group, BalanceContext request) {
        if (candidates.size() == 0) {
            return -1;
        }
        var prefill = request.getConfig().getRouter().getRoles().getPrefill();
        long inputTokens = request.getRequest().getSeqLen();
        var release = prefill.getHotspotRelease();
        if (release.isEnabled()) {
            redirectLimiter.recordPlacement(request, candidates.maximumCacheHit);
        }
        return prefill.getTokenBalance() == null
                ? selectByTtft(candidates, role, group, inputTokens, prefill.getCacheAffinity(), release, request.getConfig().isDirect())
                : selectTokenBalancedCandidate(candidates, role, group, inputTokens, prefill.getTokenBalance(), release);
    }

    private boolean permitsRedirect(PrefillCandidateSet candidates, int pinned, int alternate,
                                    RoutingConfig.HotspotReleaseConfig release) {
        return alternate < 0 || alternate == pinned || redirectLimiter.tryAcquire(
                candidates.maximumCacheHit - candidates.cacheHit(alternate), release.getMaxExtraComputePermille());
    }

    /** Cache leaders pin hot requests unless a detected hotspot can redirect under the shared limit. */
    private int selectTokenBalancedCandidate(PrefillCandidateSet survivors,
            RoleType roleType, String group, long seqLen,
            RoutingConfig.TokenBalanceConfig tokenBalance, RoutingConfig.HotspotReleaseConfig release) {
        boolean hotspotRelease = release.isEnabled();
        boolean cacheFirst = PrefillTokenBalance.cacheFirst(
                survivors.maximumCacheHit, seqLen, tokenBalance.getMinPrefixHitPercent());
        if (!cacheFirst) {
            return selectByTokenWork(survivors, roleType, group, seqLen, hotspotRelease, tieRotation);
        }

        BitSet leaders = new BitSet();
        BitSet openLeaders = new BitSet();
        BitSet pinnedLeaders = new BitSet();
        for (int i = 0; i < survivors.size(); i++) {
            if (survivors.cacheHit(i) != survivors.maximumCacheHit) { continue; }
            leaders.set(i);
            PrefillEndpoint endpoint = survivors.endpoint(i);
            if (!endpoint.engineQueueAdmissionBlocked()) { openLeaders.set(i); }
            if (!hotspotRelease || !survivors.isHotspot(i)) {
                pinnedLeaders.set(i);
            }
        }
        // Keep the original affinity choice as the fallback when the shared quota is exhausted.
        int source = selectTokenLeader(survivors, openLeaders.isEmpty() ? leaders : openLeaders,
                roleType, group, tieRotation);
        int fallback = openLeaders.get(source) ? source : -1;
        if (pinnedLeaders.get(source)) { return fallback; }

        int selected;
        if (pinnedLeaders.isEmpty()) {
            selected = selectByTokenWork(survivors, roleType, group, seqLen, true, redirectRotation);
        } else {
            pinnedLeaders.and(openLeaders);
            selected = selectTokenLeader(survivors, pinnedLeaders, roleType, group, redirectRotation);
        }
        return permitsRedirect(survivors, source, selected, release) ? selected : fallback;
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
                                         long seqLen, boolean accountForHits, EndpointRoundRobin rotation) {
        double[] scores = new double[survivors.size()];
        double minimum = Double.POSITIVE_INFINITY;
        for (int i = 0; i < survivors.size(); i++) {
            PrefillEndpoint endpoint = survivors.endpoint(i);
            if (endpoint.engineQueueAdmissionBlocked()) {
                scores[i] = Double.POSITIVE_INFINITY;
                continue;
            }
            long tokens = accountForHits ? Math.max(0L, seqLen - survivors.cacheHit(i)) : seqLen;
            scores[i] = endpoint.tokenWorkloadScore() + tokens;
            minimum = Math.min(minimum, scores[i]);
        }
        if (!Double.isFinite(minimum)) { return -1; }
        double score = minimum;
        return rotation.next(role, group, survivors.size(), i -> scores[i] == score, survivors::endpointAddress);
    }

    private record TtftChoice(int index, String reason) { }

    /** Apply the ordinary ranking first, then release its winner if that endpoint is hot. */
    private int selectByTtft(PrefillCandidateSet survivors, RoleType role, String group, long seqLen,
                             RoutingConfig.CacheAffinityConfig affinity, RoutingConfig.HotspotReleaseConfig release,
                             boolean direct) {
        BitSet eligible = new BitSet(survivors.size());
        eligible.set(0, survivors.size());
        TtftChoice choice = rankByTtft(survivors, eligible, role, group, seqLen, affinity,
                release.isEnabled(), tieRotation);
        int selected = choice.index();
        String reason = choice.reason();
        if (release.isEnabled() && survivors.isHotspot(selected)) {
            for (int i = 0; i < survivors.size(); i++) {
                if (survivors.isHotspot(i) || survivors.admissionBlock(i) != null
                        || survivors.endpoint(i).engineQueueAdmissionBlocked()
                        || direct && !survivors.endpoint(i).canAcceptRequest()) {
                    eligible.clear(i);
                }
            }
            if (!eligible.isEmpty()) {
                int alternate = rankByTtft(survivors, eligible, role, group, seqLen, affinity,
                        true, redirectRotation).index();
                if (permitsRedirect(survivors, selected, alternate, release)) {
                    selected = alternate;
                    reason = "HOTSPOT_RELEASE";
                } else {
                    reason = "HOTSPOT_RATE_LIMIT";
                }
            }
        }
        if (reason != null) {
            reporter.reportCacheAffinityDecision(role, survivors.endpoint(selected).getIp(), reason);
            if (Logger.isDebugEnabled()) {
                Logger.debug("CostBasedPrefill cache-affinity decision - role: {}, group: {}, selected: {}, "
                                + "minProjectedTtftMs: {}, selectedProjectedTtftMs: {}, hitTokens: {}, reason: {}",
                        role, group, survivors.endpointAddress(selected), survivors.minimumProjectedTtftMs,
                        survivors.projectedTtftMs(selected), survivors.cacheHit(selected), reason);
            }
        }
        return selected;
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
            long cutoff = saturatingAdd(survivors.projectedTtftMs(baseline), Math.max(0L, affinity.getMaxExtraTtftMs()));
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
        int selected = rotation.next(role, group, survivors.size(),
                i -> eligible.get(i) && survivors.projectedTtftMs(i) == survivors.projectedTtftMs(reference)
                        && (affinity == null || survivors.compareCacheMatch(i, reference) == 0),
                survivors::endpointAddress);
        return new TtftChoice(selected, reason);
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

    private static long saturatingAdd(long left, long right) {
        if (right > 0L && left > Long.MAX_VALUE - right) {
            return Long.MAX_VALUE;
        }
        if (right < 0L && left < Long.MIN_VALUE - right) {
            return Long.MIN_VALUE;
        }
        return left + right;
    }

}
