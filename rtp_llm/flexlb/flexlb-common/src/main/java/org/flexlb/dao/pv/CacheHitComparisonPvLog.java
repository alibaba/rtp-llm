package org.flexlb.dao.pv;

/**
 * Actual engine cache hit compared with the prediction used for scheduling.
 */
public record CacheHitComparisonPvLog(
        String eventType,
        String requestId,
        String cacheMatchSource,
        String role,
        String group,
        String workerIp,
        int workerPort,
        String taskState,
        long inputTokens,
        long blockSize,
        long predictedHitTokens,
        long actualHitTokens,
        long deltaHitTokens) {
}
