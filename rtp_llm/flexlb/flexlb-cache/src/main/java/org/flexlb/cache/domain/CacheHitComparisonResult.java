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
        boolean kvcmPredictionAvailable,
        long kvcmLocalPredictedHitTokens,
        long kvcmP2pFetchTokens,
        long kvcmP2pTotalMatchTokens,
        long localStandbyPredictedHitTokens,
        boolean localStandbyPredictionAvailable,
        long actualHitTokens,
        long routingDeltaHitTokens,
        long kvcmLocalDeltaHitTokens,
        long kvcmP2pTotalMatchDeltaHitTokens,
        long localStandbyDeltaHitTokens) {

    public CacheHitComparisonResult(
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
        this(
                eventType,
                requestId,
                cacheMatchSource,
                role,
                group,
                workerIp,
                workerPort,
                taskState,
                inputTokens,
                routingBlockSize,
                localStandbyBlockSize,
                routingPredictedHitTokens,
                false,
                0,
                0,
                0,
                localStandbyPredictedHitTokens,
                localStandbyPredictionAvailable,
                actualHitTokens,
                routingDeltaHitTokens,
                0,
                0,
                localStandbyDeltaHitTokens);
    }
}
