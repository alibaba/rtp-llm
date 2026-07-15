package org.flexlb.dao.pv;

import org.flexlb.dao.route.RoleType;

import java.util.List;

/**
 * Debug-only snapshot of the inputs used by the shortest-TTFT strategy.
 */
public record ShortestTtftDecision(
        RoleType role,
        String group,
        long requestInputTokens,
        long minimumTtft,
        double similarTtftThreshold,
        List<WorkerDecision> workers) {

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
