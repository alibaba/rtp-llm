package org.flexlb.dao.master;

/**
 * Engine feedback comparing the routing cache-hit prediction with the actual cache hit.
 * The embedded {@link WorkerIdentity} preserves both the routing identity
 * ({@code ip:port@engineIndex}) and metrics identity ({@code ip@engineIndex}).
 */
public record CacheHitFeedback(
        String eventType,
        String requestId,
        String cacheMatchSource,
        String role,
        String group,
        WorkerIdentity workerIdentity,
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
            int engineIndex,
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
        this(
                eventType,
                requestId,
                cacheMatchSource,
                role,
                group,
                new WorkerIdentity(workerIp, workerPort, engineIndex),
                taskState,
                inputTokens,
                blockSize,
                predictedHitTokens,
                kvcmMatchAvailable,
                kvcmLocalMatchTokens,
                kvcmP2pFetchTokens,
                kvcmP2pTotalMatchTokens,
                actualHitTokens,
                deltaHitTokens);
    }

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
            boolean kvcmMatchAvailable,
            long kvcmLocalMatchTokens,
            long kvcmP2pFetchTokens,
            long kvcmP2pTotalMatchTokens,
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
                0,
                taskState,
                inputTokens,
                blockSize,
                predictedHitTokens,
                kvcmMatchAvailable,
                kvcmLocalMatchTokens,
                kvcmP2pFetchTokens,
                kvcmP2pTotalMatchTokens,
                actualHitTokens,
                deltaHitTokens);
    }

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
                0,
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

    /**
     * Returns the feedback target in {@code ip:port@engineIndex} format. The index identifies
     * one independently routable engine behind the physical frontend.
     */
    public String logicalWorkerId() {
        return workerIdentity.getLogicalIpPort();
    }

    /** Returns the metrics identity in {@code ip@engineIndex} format. */
    public String ipIndex() {
        return workerIdentity.getIpIndex();
    }

    public int workerPort() {
        return workerIdentity.getPort();
    }

    public int engineIndex() {
        return workerIdentity.getEngineIndex();
    }
}
