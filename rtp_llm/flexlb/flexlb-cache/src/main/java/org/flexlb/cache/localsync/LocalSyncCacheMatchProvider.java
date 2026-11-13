package org.flexlb.cache.localsync;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.cache.CacheMatchSource;
import org.flexlb.cache.HostCacheMatch;
import org.flexlb.cache.match.CacheMatchProvider;
import org.flexlb.cache.metrics.CacheMetricsReporter;
import org.flexlb.config.cache.CacheMatchConfiguration;
import org.flexlb.config.service.RoleType;
import org.flexlb.util.discovery.DiscoveredHost;
import org.flexlb.worker.cache.CacheStatus;
import org.flexlb.worker.cache.WorkerCacheUpdateResult;
import org.flexlb.worker.rpc.EngineAddressResolver;
import org.flexlb.worker.status.WorkerStatus;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Local cache matching backed by complete cache snapshots synchronized through GetCacheStatus.
 */
@Slf4j
@Component
public class LocalSyncCacheMatchProvider
        implements CacheMatchProvider, EngineAddressResolver.Listener {
    private final KvCacheManager kvCacheManager;
    private final CacheMetricsReporter cacheMetricsReporter;

    @Autowired
    public LocalSyncCacheMatchProvider(
            KvCacheManager kvCacheManager,
            CacheMetricsReporter cacheMetricsReporter,
            EngineAddressResolver addressResolver,
            CacheMatchConfiguration configuration) {
        this(kvCacheManager, cacheMetricsReporter);
        if (configuration.isLocalSyncEnabled()) {
            addressResolver.subscribe(this);
        }
    }

    public LocalSyncCacheMatchProvider(
            KvCacheManager kvCacheManager, CacheMetricsReporter cacheMetricsReporter) {
        this.kvCacheManager = kvCacheManager;
        this.cacheMetricsReporter = cacheMetricsReporter;
    }

    @Override
    public void onAddressUpdate(List<DiscoveredHost> hosts) {
        if (hosts == null) {
            log.warn("Ignoring null engine address update for local cache cleanup");
            return;
        }
        kvCacheManager.removeStaleEngineCaches(
                hosts.stream().map(DiscoveredHost::getIpPort).toList());
    }

    @Override
    public CacheMatchSource source() {
        return CacheMatchSource.LOCAL_SYNC;
    }

    @Override
    public Map<String, HostCacheMatch> findMatchingEngines(
            String requestId,
            List<Long> blockCacheKeys,
            long blockSize,
            RoleType roleType,
            String group) {
        return HostCacheMatch.fromLocalMatches(
                kvCacheManager.findMatchingEngines(blockCacheKeys, roleType, group));
    }

    public WorkerCacheUpdateResult updateFromWorkerStatus(WorkerStatus workerStatus) {
        long startTime = System.nanoTime() / 1000;
        String logicalIpPort = workerStatus.getLogicalIpPort();
        String metricIpPort = workerStatus.getMetricIpPort();
        String role = workerStatus.getRole() == null ? null : workerStatus.getRole().name();

        try {
            CacheStatus cacheStatus = workerStatus.getCacheStatus();
            if (cacheStatus == null) {
                WorkerCacheUpdateResult result =
                        buildFailureResult(logicalIpPort, "Worker Cache Status is null");
                reportUpdateTime(metricIpPort, role, startTime, "0");
                return result;
            }

            Set<Long> cachedKeys = cacheStatus.getCachedKeys();
            if (cachedKeys == null) {
                WorkerCacheUpdateResult result =
                        buildFailureResult(logicalIpPort, "Worker Cached Keys is null");
                reportUpdateTime(metricIpPort, role, startTime, "0");
                return result;
            }

            kvCacheManager.updateEngineCache(workerStatus, role, cachedKeys);
            WorkerCacheUpdateResult result = buildSuccessResult(workerStatus, cacheStatus);
            reportUpdateTime(metricIpPort, role, startTime, "1");
            return result;
        } catch (Throwable e) {
            log.error("Error updating worker cache for: {}", logicalIpPort, e);
            WorkerCacheUpdateResult result = buildFailureResult(logicalIpPort, e.getMessage());
            reportUpdateTime(metricIpPort, role, startTime, "0");
            return result;
        }
    }

    public void removeEngineBlockCache(String engineIpPort) {
        kvCacheManager.removeEngineCache(engineIpPort);
    }

    private void reportUpdateTime(String address, String role, long startTime, String success) {
        org.flexlb.util.metrics.SafeFlexMonitor.run(
                "cache.update",
                () ->
                        cacheMetricsReporter.reportUpdateEngineBlockCacheRT(
                                address, role, startTime, success));
    }

    private WorkerCacheUpdateResult buildSuccessResult(
            WorkerStatus workerStatus, CacheStatus cacheStatus) {
        return WorkerCacheUpdateResult.builder()
                .success(true)
                .logicalIpPort(workerStatus.getLogicalIpPort())
                .cacheBlockCount(cacheStatus.getCachedKeys().size())
                .availableKvCache(cacheStatus.getAvailableKvCache())
                .totalKvCache(cacheStatus.getTotalKvCache())
                .cacheVersion(cacheStatus.getVersion())
                .build();
    }

    private WorkerCacheUpdateResult buildFailureResult(String logicalIpPort, String errorMessage) {
        return WorkerCacheUpdateResult.builder()
                .success(false)
                .logicalIpPort(logicalIpPort)
                .errorMessage(errorMessage)
                .build();
    }
}
