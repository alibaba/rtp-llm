package org.flexlb.worker.cache;

import org.flexlb.worker.status.WorkerStatus;

/**
 * Applies Worker cache observations and removes cache state when an endpoint is withdrawn.
 */
public interface WorkerCaches {
    org.flexlb.worker.cache.WorkerCacheUpdateResult updateFromWorkerStatus(WorkerStatus status);

    void removeEngineBlockCache(String address);

    java.util.List<
                    java.util.concurrent.CompletableFuture<
                            org.flexlb.worker.cache.CacheHitComparisonResult>>
            observeCacheHitFeedback(WorkerStatus worker, WorkerStatus.StatusObservation status);
}
