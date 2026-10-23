package org.flexlb.cache.match.theory;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
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
 * When the window settings change, a new pool is allocated on the next record and the old one is released.
 * Calls are serialized by the Master theory-hit reporting thread.
 */
@Slf4j
@Component
public class TheoryCacheKeyHistory {
    private final WorkerStatusProvider workerStatusProvider;
    private volatile WindowSettings configuredSettings;
    private WindowSettings activeSettings;
    private RecentCacheKeyWindow historyWindow;

    public TheoryCacheKeyHistory(ConfigService configService, WorkerStatusProvider workerStatusProvider) {
        this.workerStatusProvider = workerStatusProvider;
        this.configuredSettings = WindowSettings.from(configService.loadBalanceConfig());
        configService.addUpdateListener(config -> configuredSettings = WindowSettings.from(config));
    }

    /**
     * Returns null while Prefill capacity has not been reported; no sample is recorded in that state.
     * Changed window settings drop the accumulated history and start a new pool.
     */
    public RecentCacheKeyWindow.Snapshot record(List<Long> cacheKeys) {
        WindowSettings settings = configuredSettings;
        if (historyWindow == null || !settings.equals(activeSettings)) {
            OptionalLong estimatedHistoryCapacity = estimateHistoryCapacity(settings);
            if (estimatedHistoryCapacity.isPresent()) {
                historyWindow = new RecentCacheKeyWindow(
                        settings.durationMs(), estimatedHistoryCapacity.getAsLong(), System::currentTimeMillis);
                activeSettings = settings;
                log.info("Theory cache-key history rebuilt: {}, capacityKeys={}", settings, estimatedHistoryCapacity.getAsLong());
            } else if (historyWindow == null) {
                return null;
            }
        }
        return historyWindow.record(cacheKeys);
    }

    private OptionalLong estimateHistoryCapacity(WindowSettings settings) {
        Collection<WorkerStatus> prefillWorkers = workerStatusProvider.getWorkerStatuses(RoleType.PREFILL, null);
        if (prefillWorkers.isEmpty()) {
            return OptionalLong.empty();
        }
        WorkerStatus representativePrefillWorker = prefillWorkers.iterator().next();
        WorkerStatus.EngineObservation prefillCapacitySnapshot = representativePrefillWorker.committedEngineObservation();
        if (prefillCapacitySnapshot.totalKvCacheTokens() <= 0L || prefillCapacitySnapshot.blockSize() <= 0L) {
            return OptionalLong.empty();
        }
        long maxKeys = settings.maxKeyOccurrences();
        long workerKvBlockCapacity = prefillCapacitySnapshot.totalKvCacheTokens() / prefillCapacitySnapshot.blockSize();
        // Cap each multiplication before scaling to keep the estimate within long range.
        long cappedWorkerBlockCapacity = Math.min(maxKeys, workerKvBlockCapacity);
        long cappedFleetBlockCapacity = Math.min(maxKeys, cappedWorkerBlockCapacity * prefillWorkers.size());
        long multiplier = settings.capacityMultiplier();
        long estimatedHistoryKeyOccurrences = cappedFleetBlockCapacity > maxKeys / multiplier
                ? maxKeys
                : cappedFleetBlockCapacity * multiplier;
        return estimatedHistoryKeyOccurrences > 0L
                ? OptionalLong.of(estimatedHistoryKeyOccurrences)
                : OptionalLong.empty();
    }

    private record WindowSettings(long durationMs, long maxKeyOccurrences, long capacityMultiplier) {
        static WindowSettings from(FlexlbConfig config) {
            var window = config.getObservability().getCacheHit().getRecentKeyWindow();
            return new WindowSettings(window.getDurationMs(),
                    Math.min(Integer.MAX_VALUE - 8L, window.getMaxKeyOccurrences()),
                    window.getCapacityMultiplier());
        }
    }
}
