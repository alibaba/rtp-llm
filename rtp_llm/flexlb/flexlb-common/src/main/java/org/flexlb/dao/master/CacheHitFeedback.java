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
        long actualHitTokens,
        long deltaHitTokens) {
}
