package org.flexlb.cache.match.localsync;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.cache.domain.CacheMatchSource;
import org.flexlb.cache.domain.WorkerCacheUpdateResult;
import org.flexlb.cache.match.CacheMatchProvider;
import org.flexlb.cache.telemetry.CacheMetricsReporter;
import org.flexlb.config.CacheMatchConfiguration;
import org.flexlb.dao.cache.HostCacheMatch;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.EndpointTrace;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.nameresolver.EngineAddressResolver;
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
public class LocalSyncCacheMatchProvider implements CacheMatchProvider {

    private final KvCacheManager kvCacheManager;
    private final CacheMetricsReporter cacheMetricsReporter;

    @Autowired
    public LocalSyncCacheMatchProvider(
            KvCacheManager kvCacheManager,
            CacheMetricsReporter cacheMetricsReporter,
            EngineAddressResolver addressResolver,
            CacheMatchConfiguration configuration) {
        this(kvCacheManager, cacheMetricsReporter);
    }

    public LocalSyncCacheMatchProvider(KvCacheManager kvCacheManager, CacheMetricsReporter cacheMetricsReporter) {
        this.kvCacheManager = kvCacheManager;
        this.cacheMetricsReporter = cacheMetricsReporter;
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
        return HostCacheMatch.fromLocalMatches(kvCacheManager.findMatchingEngines(blockCacheKeys, roleType, group));
    }

    public WorkerCacheUpdateResult updateFromWorkerStatus(WorkerStatus workerStatus, EndpointTrace.Poll pollTrace) {
        requireCacheMutationLock(workerStatus);
        long startTime = System.nanoTime() / 1000;
        String logicalIpPort = workerStatus.getLogicalIpPort();
        String metricIpPort = workerStatus.getMetricIpPort();
        String role = workerStatus.getRole() == null ? null : workerStatus.getRole().name();

        try {
            CacheStatus cacheStatus = workerStatus.getCacheStatus();
            if (cacheStatus == null) {
                WorkerCacheUpdateResult result = buildFailureResult(logicalIpPort, "Worker Cache Status is null");
                cacheMetricsReporter.reportUpdateEngineBlockCacheRT(metricIpPort, role, startTime, "0");
                return result;
            }

            Set<Long> cachedKeys = cacheStatus.getCachedKeys();
            if (cachedKeys == null) {
                WorkerCacheUpdateResult result = buildFailureResult(logicalIpPort, "Worker Cached Keys is null");
                cacheMetricsReporter.reportUpdateEngineBlockCacheRT(metricIpPort, role, startTime, "0");
                return result;
            }

            kvCacheManager.updateEngineCache(workerStatus, role, cachedKeys, pollTrace);
            WorkerCacheUpdateResult result = buildSuccessResult(workerStatus, cacheStatus);
            cacheMetricsReporter.reportUpdateEngineBlockCacheRT(metricIpPort, role, startTime, "1");
            return result;
        } catch (Throwable e) {
            log.error("Error updating worker cache for: {}", logicalIpPort, e);
            WorkerCacheUpdateResult result = buildFailureResult(logicalIpPort, e.getMessage());
            cacheMetricsReporter.reportUpdateEngineBlockCacheRT(metricIpPort, role, startTime, "0");
            return result;
        }
    }

    public void removeEngineBlockCache(WorkerStatus workerStatus, EndpointTrace trace) {
        requireCacheMutationLock(workerStatus);
        kvCacheManager.removeEngineCache(workerStatus.getLogicalIpPort(), trace);
    }

    private static void requireCacheMutationLock(WorkerStatus workerStatus) {
        java.util.Objects.requireNonNull(workerStatus, "workerStatus");
        if (!workerStatus.cacheMutationLock.isHeldByCurrentThread()) {
            throw new IllegalStateException("Cache mutation requires the owning generation's cacheMutationLock");
        }
    }

    private WorkerCacheUpdateResult buildSuccessResult(WorkerStatus workerStatus, CacheStatus cacheStatus) {
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
