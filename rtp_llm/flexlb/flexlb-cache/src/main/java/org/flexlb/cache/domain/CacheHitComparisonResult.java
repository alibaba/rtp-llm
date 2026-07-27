package org.flexlb.cache.domain;

/**
 * Cache-hit predictions compared with the actual engine cache hit.
 */
public record CacheHitComparisonResult(
        String eventType,
        String requestId,
        String cacheMatchSource,
        String role,
        String group,
        String workerIp,
        int workerPort,
        String taskState,
        long inputTokens,
        long routingBlockSize,
        long localStandbyBlockSize,
        long routingPredictedHitTokens,
        long localStandbyPredictedHitTokens,
        boolean localStandbyPredictionAvailable,
        long actualHitTokens,
        long routingDeltaHitTokens,
        long localStandbyDeltaHitTokens) {
}
