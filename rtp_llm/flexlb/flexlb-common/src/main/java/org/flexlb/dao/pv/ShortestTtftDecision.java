package org.flexlb.dao.pv;

import com.fasterxml.jackson.annotation.JsonInclude;
import org.flexlb.dao.route.RoleType;

import java.util.List;

/**
 * Debug-only snapshot of the inputs used by TTFT-based strategies.
 */
public record ShortestTtftDecision(
        RoleType role,
        String group,
        String strategy,
        String selectionReason,
        long requestInputTokens,
        long minimumTtft,
        double similarTtftThreshold,
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
            double toleratedExtraTtft) {
    }

    public record WorkerDecision(
            String ip,
            int port,
            boolean topCandidate,
            boolean similarTtftCandidate,
            boolean selected,
            long cacheBlockSize,
            long requestHitCacheTokens,
            long requestPrefillTime,
            long queueTime,
            long estimatedTtft,
            long lastSelectedTimeUs,
            int trackedTaskCount,
            int waitingTaskCount,
            int runningTaskCount,
            List<QueueTask> trackedTasks,
            List<QueueTask> waitingTasks,
            List<QueueTask> runningTasks) {
    }

    public record QueueTask(
            String requestId,
            String state,
            long inputTokens,
            long hitCacheTokens,
            long estimatedPrefillTime,
            long waitingTime) {
    }
}
