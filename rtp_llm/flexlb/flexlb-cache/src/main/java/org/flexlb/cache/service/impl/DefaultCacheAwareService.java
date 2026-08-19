package org.flexlb.cache.service.impl;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.cache.core.KvCacheManager;
import org.flexlb.cache.domain.WorkerCacheUpdateResult;
import org.flexlb.cache.monitor.CacheMetricsReporter;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Default implementation of cache-aware service
 * Provides unified cache management service, encapsulating underlying KvCacheManager
 *
 * @author FlexLB
 */
@Slf4j
@Service
public class DefaultCacheAwareService implements CacheAwareService {

    @Autowired
    private KvCacheManager kvCacheManager;

    @Autowired
    private CacheMetricsReporter cacheMetricsReporter;

    @Override
    public Map<String, Integer> findMatchingEngines(List<Long> blockCacheKeys,
        RoleType roleType, String group) {

        long startTime = System.nanoTime() / 1000;

        try {
            if (blockCacheKeys == null || blockCacheKeys.isEmpty()) {
                return Collections.emptyMap();
            }

            Map<String/*engineIpPort*/, Integer/*prefixMatchLength*/> resultMap
                = kvCacheManager.findMatchingEngines(blockCacheKeys, roleType, group);

            cacheMetricsReporter.reportFindMatchingEnginesRT(roleType, startTime, "0");

            return resultMap;
        } catch (Exception e) {
            cacheMetricsReporter.reportFindMatchingEnginesRT(roleType, startTime, "1");
            log.error("Error finding matching engines for role: {}", roleType, e);
            return Collections.emptyMap();
        }
    }

    @Override
    public WorkerCacheUpdateResult publishEngineCacheSnapshot(
            String engineIpPort, RoleType roleType, Set<Long> cachedKeys) {
        long startTime = System.nanoTime() / 1000;
        String role = roleType.getCode();

        try {
            if (engineIpPort == null || cachedKeys == null) {
                WorkerCacheUpdateResult result = buildFailureResult(engineIpPort, "Worker Cached Keys is null");
                reportUpdateLatency(role, startTime, "0");
                return result;
            }

            kvCacheManager.updateEngineCache(engineIpPort, role, cachedKeys);

            WorkerCacheUpdateResult result = WorkerCacheUpdateResult.builder()
                    .success(true)
                    .engineIpPort(engineIpPort)
                    .cacheBlockCount(cachedKeys.size())
                    .build();

            reportUpdateLatency(role, startTime, "1");

            return result;

        } catch (Throwable e) {
            log.error("Error updating worker cache for: {}", engineIpPort, e);

            String message = e.getMessage() == null
                    ? e.getClass().getSimpleName() : e.getMessage();
            WorkerCacheUpdateResult result = buildFailureResult(engineIpPort, message);

            reportUpdateLatency(role, startTime, "0");

            return result;
        }
    }

    @Override
    @Deprecated
    public WorkerCacheUpdateResult updateEngineBlockCache(WorkerStatus workerStatus) {
        if (workerStatus == null || workerStatus.getCacheStatus() == null) {
            return buildFailureResult(
                    workerStatus == null ? null : workerStatus.getIpPort(),
                    "Worker Cache Status is null");
        }
        return publishEngineCacheSnapshot(
                workerStatus.getIpPort(),
                workerStatus.getRole(),
                workerStatus.getCacheStatus().getCachedKeys());
    }

    @Override
    public void clearEngineCache(String engineIpPort) {
        kvCacheManager.clearEngineCache(engineIpPort);
    }

    private void reportUpdateLatency(String role, long startTime, String success) {
        try {
            cacheMetricsReporter.reportUpdateEngineBlockCacheRT(role, startTime, success);
        } catch (RuntimeException metricFailure) {
            log.warn("Failed to report cache publication latency for role={}",
                    role, metricFailure);
        }
    }

    /**
     * Build failure result
     */
    private WorkerCacheUpdateResult buildFailureResult(String engineIpPort, String errorMessage) {
        return WorkerCacheUpdateResult.builder()
            .success(false)
            .engineIpPort(engineIpPort)
            .errorMessage(errorMessage)
            .build();
    }
}
