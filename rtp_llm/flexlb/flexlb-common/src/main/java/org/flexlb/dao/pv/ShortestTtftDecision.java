package org.flexlb.dao.pv;

import com.fasterxml.jackson.annotation.JsonInclude;
import org.flexlb.dao.route.RoleType;

import java.util.List;

/**
 * Compact PV snapshot of the inputs used by one TTFT-based routing decision.
 */
public record ShortestTtftDecision(
        RoleType role,
        String group,
        String strategy,
        String selectionReason,
        long decisionTimeMs,
        int routingAttempt,
        double p2pHitDiscount,
        long requestInputTokens,
        long minimumTtft,
        double similarTtftThreshold,
        int totalWorkerCount,
        int candidateWorkerCount,
        int similarWorkerCount,
        int snapshotWorkerLimit,
        boolean snapshotTruncated,
        long snapshotOutstandingUncachedTokens,
        List<WorkerDecision> workers,
        @JsonInclude(JsonInclude.Include.NON_NULL)
        CacheAffinityDecision cacheAffinityDecision) {

    /**
     * Cache-affinity-specific inputs for one routing decision. Null for other TTFT strategies.
     */
    public record CacheAffinityDecision(
            String cacheLeaderIpPort,
            String shortestTtftWorkerIpPort,
            long cacheLeadTokens,
            long extraTtft,
            double toleratedExtraTtft,
            long outstandingUncachedTokensThreshold,
            boolean cacheLeaderOutstandingEligible) {
    }

    public record WorkerDecision(
            int estimatedTtftRank,
            String ip,
            int port,
            boolean topCandidate,
            boolean similarTtftCandidate,
            boolean selected,
            boolean cacheLeader,
            boolean shortestTtftWorker,
            boolean outstandingGuardEligible,
            long cacheBlockSize,
            long requestHitCacheTokens,
            double requestHitRatePct,
            long requestUncachedTokens,
            long requestLocalMatchTokens,
            long requestP2pFetchTokens,
            long requestP2pTotalMatchTokens,
            long requestP2pAddedMatchTokens,
            long requestPrefillTime,
            long queueTime,
            long estimatedTtft,
            long outstandingUncachedTokens,
            long outstandingAfterRequestUncachedTokens,
            long lastSelectedTimeUs,
            int trackedTaskCount,
            long inTransitAndWaitingTaskCount,
            long inTransitAndWaitingUncachedTokens,
            int trackedRunningTaskCount,
            long trackedRunningRemainingPrefillTokens,
            int engineWaitingTaskCount,
            long engineWaitingUncachedTokens,
            int engineRunningTaskCount,
            long engineRunningRemainingPrefillTokens,
            boolean alive,
            boolean resourceAvailable,
            Long availableConcurrency,
            long availableKvCacheTokens,
            long usedKvCacheTokens,
            long statusVersion,
            long statusAgeUs,
            long statusUpdateIntervalUs,
            long cacheAgeUs) {
    }
}
