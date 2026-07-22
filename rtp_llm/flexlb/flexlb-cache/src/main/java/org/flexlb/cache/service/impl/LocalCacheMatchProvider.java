package org.flexlb.cache.service.impl;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.cache.core.KvCacheManager;
import org.flexlb.cache.domain.WorkerCacheUpdateResult;
import org.flexlb.cache.monitor.CacheMetricsReporter;
import org.flexlb.cache.service.CacheMatchProvider;
import org.flexlb.cache.service.CacheMatchSource;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Map;
import java.util.Set;

@Slf4j
@Component
public class LocalCacheMatchProvider implements CacheMatchProvider {

    private final KvCacheManager kvCacheManager;
    private final CacheMetricsReporter cacheMetricsReporter;

    public LocalCacheMatchProvider(
            KvCacheManager kvCacheManager,
            CacheMetricsReporter cacheMetricsReporter) {
        this.kvCacheManager = kvCacheManager;
        this.cacheMetricsReporter = cacheMetricsReporter;
    }

    @Override
    public CacheMatchSource source() {
        return CacheMatchSource.LOCAL;
    }

    @Override
    public Map<String, Integer> findMatchingEngines(
            String requestId,
            List<Long> blockCacheKeys,
            long blockSize,
            RoleType roleType,
            String group) {
        return kvCacheManager.findMatchingEngines(blockCacheKeys, roleType, group);
    }

    @Override
    public WorkerCacheUpdateResult updateEngineBlockCache(WorkerStatus workerStatus) {
        long startTime = System.nanoTime() / 1000;
        String engineIpPort = workerStatus.getIpPort();
        String role = workerStatus.getRole();

        try {
            CacheStatus cacheStatus = workerStatus.getCacheStatus();
            if (cacheStatus == null) {
                WorkerCacheUpdateResult result = buildFailureResult(
                        engineIpPort, "Worker Cache Status is null");
                cacheMetricsReporter.reportUpdateEngineBlockCacheRT(
                        engineIpPort, role, startTime, "0");
                return result;
            }

            Set<Long> cachedKeys = cacheStatus.getCachedKeys();
            if (cachedKeys == null) {
                WorkerCacheUpdateResult result = buildFailureResult(
                        engineIpPort, "Worker Cached Keys is null");
                cacheMetricsReporter.reportUpdateEngineBlockCacheRT(
                        engineIpPort, role, startTime, "0");
                return result;
            }

            kvCacheManager.updateEngineCache(engineIpPort, role, cachedKeys);
            WorkerCacheUpdateResult result = buildSuccessResult(workerStatus, cacheStatus);
            cacheMetricsReporter.reportUpdateEngineBlockCacheRT(
                    engineIpPort, role, startTime, "1");
            return result;
        } catch (Throwable e) {
            log.error("Error updating worker cache for: {}", engineIpPort, e);
            WorkerCacheUpdateResult result = buildFailureResult(engineIpPort, e.getMessage());
            cacheMetricsReporter.reportUpdateEngineBlockCacheRT(
                    engineIpPort, role, startTime, "0");
            return result;
        }
    }

    private WorkerCacheUpdateResult buildSuccessResult(
            WorkerStatus workerStatus,
            CacheStatus cacheStatus) {
        return WorkerCacheUpdateResult.builder()
                .success(true)
                .engineIpPort(workerStatus.getIpPort())
                .cacheBlockCount(cacheStatus.getCachedKeys().size())
                .availableKvCache(cacheStatus.getAvailableKvCache())
                .totalKvCache(cacheStatus.getTotalKvCache())
                .cacheVersion(cacheStatus.getVersion())
                .build();
    }

    private WorkerCacheUpdateResult buildFailureResult(
            String engineIpPort,
            String errorMessage) {
        return WorkerCacheUpdateResult.builder()
                .success(false)
                .engineIpPort(engineIpPort)
                .errorMessage(errorMessage)
                .build();
    }
}
