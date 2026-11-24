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
                ? selectByTtft(candidates, role, group, inputTokens, prefill.getCacheAffinity(), release)
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

    /** Rank by cache affinity, then consider a paced redirect from a congested cache leader. */
    private int selectByTtft(PrefillCandidateSet survivors, RoleType roleType, String group, long seqLen,
                             RoutingConfig.CacheAffinityConfig cacheAffinity, RoutingConfig.HotspotReleaseConfig release) {
        boolean hotspotRelease = release.isEnabled();
        long minProjectedTtftMs = survivors.minimumProjectedTtftMs;
        BitSet preferredCandidates = new BitSet(survivors.size());
        BitSet releasedCandidates = new BitSet();
        long affinityCutoffMs = 0L;
        String affinityReason = null;
        int baselineCacheLeader = -1;
        if (cacheAffinity != null) {
            affinityCutoffMs = saturatingAdd(
                    minProjectedTtftMs,
                    Math.max(0L, cacheAffinity.getMaxExtraTtftMs()));
            affinityReason = "NO_CACHE_LEAD";
            for (int i = 0; i < survivors.size(); i++) {
                if (survivors.projectedTtftMs(i) == minProjectedTtftMs
                        && (baselineCacheLeader < 0 || survivors.compareCacheMatch(i, baselineCacheLeader) > 0)) {
                    baselineCacheLeader = i;
                }
            }
            boolean strongerCache = false;
            boolean minimumHitRateMet = false;
            double minimumHitRate = normalizedHitRate(cacheAffinity.getMinPrefixHitPercent());
            for (int i = 0; i < survivors.size(); i++) {
                if (survivors.compareCacheMatch(i, baselineCacheLeader) <= 0) {
                    continue;
                }
                strongerCache = true;
                long hitTokens = survivors.cacheHit(i);
                if (minimumHitRate > 0.0 && (seqLen <= 0L
                        || hitTokens * RoutingConfig.PERCENTAGE_SCALE / seqLen < minimumHitRate)) {
                    continue;
                }
                minimumHitRateMet = true;
                if (hotspotRelease || survivors.projectedTtftMs(i) <= affinityCutoffMs) {
                    preferredCandidates.set(i);
                    if (hotspotRelease && survivors.isHotspot(i)) {
                        releasedCandidates.set(i);
                    }
                }
            }
            if (!preferredCandidates.isEmpty()) {
                affinityReason = "CACHE_LEADER";
            } else if (minimumHitRateMet) {
                affinityReason = "OVER_CAP";
            } else if (strongerCache) {
                affinityReason = "LOW_CACHE_HIT";
            }
        }

        int selectedIndex;
        if (preferredCandidates.isEmpty()) {
            selectedIndex = selectBaseline(survivors, roleType, group, baselineCacheLeader, tieRotation);
        } else {
            int source = selectCacheLeader(survivors, preferredCandidates, roleType, group, tieRotation);
            selectedIndex = source;
            if (releasedCandidates.get(source)) {
                preferredCandidates.andNot(releasedCandidates);
                selectedIndex = preferredCandidates.isEmpty()
                        ? selectBaseline(survivors, roleType, group, baselineCacheLeader, redirectRotation)
                        : selectCacheLeader(survivors, preferredCandidates, roleType, group, redirectRotation);
                affinityReason = "HOTSPOT_RELEASE";
            }
            if (!permitsRedirect(survivors, source, selectedIndex, release)) {
                selectedIndex = source;
                affinityReason = "HOTSPOT_RATE_LIMIT";
            }
        }

        if (selectedIndex >= 0 && affinityReason != null) {
            reporter.reportCacheAffinityDecision(
                    roleType, survivors.endpoint(selectedIndex).getIp(),
                    affinityReason);
            if (Logger.isDebugEnabled()) {
                Logger.debug(
                        "CostBasedPrefill cache-affinity decision - role: {}, group: {}, "
                                + "selected: {}, minProjectedTtftMs: {}, "
                                + "selectedProjectedTtftMs: {}, ttftCutoffMs: {}, "
                                + "hitTokens: {}, reason: {}",
                        roleType,
                        group,
                        survivors.endpointAddress(selectedIndex),
                        minProjectedTtftMs,
                        survivors.projectedTtftMs(selectedIndex),
                        affinityCutoffMs,
                        survivors.cacheHit(selectedIndex),
                        affinityReason);
            }
        }
        return selectedIndex;
    }

    private static int selectBaseline(PrefillCandidateSet survivors, RoleType role, String group,
                                      int reference, EndpointRoundRobin rotation) {
        return rotation.next(role, group, survivors.size(),
                i -> survivors.projectedTtftMs(i) == survivors.minimumProjectedTtftMs
                        && (reference < 0 || survivors.compareCacheMatch(i, reference) == 0),
                survivors::endpointAddress);
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
