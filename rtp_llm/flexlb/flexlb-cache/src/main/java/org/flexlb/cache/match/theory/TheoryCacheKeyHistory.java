package org.flexlb.cache.match.theory;

import org.flexlb.config.ConfigService;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusProvider;
import org.flexlb.dao.route.RoleType;
import org.springframework.stereotype.Component;

import java.util.Collection;
import java.util.List;
import java.util.OptionalLong;

/**
 * All requests in one Master share a history pool sized from Prefill KV capacity.
 * Prefill workers share the same KV capacity and block size; one worker supplies the capacity estimate.
 * The pool is allocated when that worker has reported its capacity.
 * Calls are serialized by the Master theory-hit reporting thread.
 */
@Component
public class TheoryCacheKeyHistory {
    private final WorkerStatusProvider workerStatusProvider;
    private final long historyDurationMs;
    private final long configuredMaxKeyOccurrences;
    private RecentCacheKeyWindow historyWindow;

    public TheoryCacheKeyHistory(ConfigService configService, WorkerStatusProvider workerStatusProvider) {
        this.workerStatusProvider = workerStatusProvider;
        var historyConfig = configService.loadBalanceConfig().getObservability().getCacheHit().getRecentKeyWindow();
        this.historyDurationMs = historyConfig.getDurationMs();
        this.configuredMaxKeyOccurrences = Math.min(Integer.MAX_VALUE - 8L, historyConfig.getMaxKeyOccurrences());
    }

    /**
     * Returns null while Prefill capacity has not been reported; no sample is recorded in that state.
     * History capacity and duration stay fixed until the Master restarts.
     */
    public RecentCacheKeyWindow.Snapshot record(List<Long> cacheKeys) {
        if (historyWindow == null) {
            OptionalLong estimatedHistoryCapacity = estimateHistoryCapacity();
            if (estimatedHistoryCapacity.isEmpty()) {
                return null;
            }
            historyWindow = new RecentCacheKeyWindow(
                    historyDurationMs, estimatedHistoryCapacity.getAsLong(), System::currentTimeMillis);
        }
        return historyWindow.record(cacheKeys);
    }

    private OptionalLong estimateHistoryCapacity() {
        Collection<WorkerStatus> prefillWorkers = workerStatusProvider.getWorkerStatuses(RoleType.PREFILL, null);
        if (prefillWorkers.isEmpty()) {
            return OptionalLong.empty();
        }
        WorkerStatus representativePrefillWorker = prefillWorkers.iterator().next();
        WorkerStatus.EngineObservation prefillCapacitySnapshot = representativePrefillWorker.committedEngineObservation();
        if (prefillCapacitySnapshot.totalKvCacheTokens() <= 0L || prefillCapacitySnapshot.blockSize() <= 0L) {
            return OptionalLong.empty();
        }
        long workerKvBlockCapacity = prefillCapacitySnapshot.totalKvCacheTokens() / prefillCapacitySnapshot.blockSize();
        // Cap each multiplication before scaling to keep the estimate within long range.
        long cappedWorkerBlockCapacity = Math.min(configuredMaxKeyOccurrences, workerKvBlockCapacity);
        long cappedFleetBlockCapacity = Math.min(configuredMaxKeyOccurrences, cappedWorkerBlockCapacity * prefillWorkers.size());
        long estimatedHistoryKeyOccurrences = Math.min(configuredMaxKeyOccurrences, cappedFleetBlockCapacity * 10L);
        return estimatedHistoryKeyOccurrences > 0L
                ? OptionalLong.of(estimatedHistoryKeyOccurrences)
                : OptionalLong.empty();
    }
}
