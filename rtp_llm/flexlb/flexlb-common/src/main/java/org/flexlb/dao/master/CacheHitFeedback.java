package org.flexlb.dao.master;

/**
 * Engine feedback comparing the routing cache-hit prediction with the actual cache hit.
 */
public record CacheHitFeedback(
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
        boolean kvcmMatchAvailable,
        long kvcmLocalMatchTokens,
        long kvcmP2pFetchTokens,
        long kvcmP2pTotalMatchTokens,
        long actualHitTokens,
        long deltaHitTokens) {

    public CacheHitFeedback(
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
                blockSize,
                predictedHitTokens,
                false,
                0,
                0,
                0,
                actualHitTokens,
                deltaHitTokens);
    }
}
