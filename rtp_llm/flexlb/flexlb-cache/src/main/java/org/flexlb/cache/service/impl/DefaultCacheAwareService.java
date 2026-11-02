package org.flexlb.cache.service.impl;

import org.flexlb.engine.grpc.client.KvcmGrpcClient;
import org.flexlb.config.CacheMatchConfiguration;
import org.flexlb.cache.match.CacheMatchFailoverManager;
import org.flexlb.cache.match.localstandby.LocalStandbyCacheManager;
import org.flexlb.cache.domain.CacheMatchSource;
import org.flexlb.exception.KvcmQueryException;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import lombok.extern.slf4j.Slf4j;
import org.flexlb.cache.core.KvCacheManager;
import org.flexlb.cache.domain.WorkerCacheUpdateResult;
import org.flexlb.cache.monitor.CacheMetricsReporter;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.EndpointTrace;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.TimeUnit;

/**
 * Default implementation of cache-aware service
 * Provides unified cache management service, encapsulating underlying KvCacheManager
 *
 * @author FlexLB
 */
@Slf4j
@Service
public class DefaultCacheAwareService implements CacheAwareService {

    private static final String FIND_METRIC_SUCCESS = "0";
    private static final String FIND_METRIC_FAILURE = "1";

    private final KvCacheManager kvCacheManager;
    private final CacheMetricsReporter cacheMetricsReporter;

    private final KvcmGrpcClient kvcm;
    private final CacheMatchConfiguration configuration;
    private final CacheMatchFailoverManager failover;
    private final LocalStandbyCacheManager standby;

    @Autowired
    public DefaultCacheAwareService(
            KvCacheManager kvCacheManager,
            CacheMetricsReporter cacheMetricsReporter,
            KvcmGrpcClient kvcm,
            CacheMatchConfiguration configuration,
            CacheMatchFailoverManager failover,
            LocalStandbyCacheManager standby) {
        this.kvcm = java.util.Objects.requireNonNull(kvcm, "kvcm");
        this.configuration = java.util.Objects.requireNonNull(configuration, "configuration");
        this.failover = java.util.Objects.requireNonNull(failover, "failover");
        this.standby = java.util.Objects.requireNonNull(standby, "standby");
        this.kvCacheManager = java.util.Objects.requireNonNull(
                kvCacheManager, "kvCacheManager");
        this.cacheMetricsReporter = java.util.Objects.requireNonNull(
                cacheMetricsReporter, "cacheMetricsReporter");
    }

    @Override
    public Map<String, Integer> findMatchingEngines(List<Long> keys, RoleType role,
            List<String> candidates, long blockSize) {
        if (keys == null || keys.isEmpty() || candidates.isEmpty()) return Map.of();
        long started = TimeUnit.NANOSECONDS.toMicros(System.nanoTime());
        if (configuration.isKvcmEnabled()) {
            if (failover.activeSource() == CacheMatchSource.LOCAL_STANDBY) {
                cacheMetricsReporter.reportStandbyFallback("active_source");
                return standby.findMatchingEngines(keys, role, candidates, blockSize);
            }
            try {
                var raw = kvcm.findMatchingEngines("", keys, blockSize, role, null);
                Map<String, Integer> matches = new java.util.HashMap<>();
                for (String address : candidates) {
                    var match = raw.get(address + "@0");
                    if (match == null) match = raw.get(address);
                    if (match != null) matches.put(address, Math.toIntExact(match.localMatchBlocks()));
                }
                reportFindLatency(role, started, FIND_METRIC_SUCCESS);
                return Map.copyOf(matches);
            } catch (KvcmQueryException failure) {
                reportFindLatency(role, started, FIND_METRIC_FAILURE);
                cacheMetricsReporter.reportStandbyFallback("kvcm_query_failure");
                return standby.findMatchingEngines(keys, role, candidates, blockSize);
            }
        }
        try {
            var result = kvCacheManager.findMatchingEngines(keys, candidates);
            reportFindLatency(role, started, FIND_METRIC_SUCCESS);
            return result;
        } catch (RuntimeException failure) {
            reportFindLatency(role, started, FIND_METRIC_FAILURE);
            throw failure;
        }
    }

    @Override
    public void updateFromRoutedRequest(Request request,
                                       List<ServerStatus> selected) {
        standby.updateFromRoutedRequest(request, selected);
    }

    private void reportFindLatency(
            RoleType roleType, long startTime, String result) {
        try {
            cacheMetricsReporter.reportFindMatchingEnginesRT(
                    roleType, startTime, result);
        } catch (RuntimeException metricFailure) {
            log.warn("Failed to report cache lookup latency for role: {}",
                    roleType, metricFailure);
        }
    }

    @Override
    public WorkerCacheUpdateResult updateEngineBlockCache(
            WorkerStatus workerStatus, EndpointTrace.Poll pollTrace) {
        requireCacheMutationLock(workerStatus);
        long startTime = TimeUnit.NANOSECONDS.toMicros(System.nanoTime());
        String engineIpPort = workerStatus.getIpPort();
        String role = workerStatus.getRole().getCode();

        try {
            if (workerStatus.getCacheStatus() == null) {
                WorkerCacheUpdateResult result = buildFailureResult(engineIpPort, "Worker Cache Status is null");
                cacheMetricsReporter.reportUpdateEngineBlockCacheRT(role, startTime, "0");
                return result;
            }

            String ipPort = workerStatus.getIpPort();
            CacheStatus cacheStatus = workerStatus.getCacheStatus();
            if (cacheStatus.getCachedKeys() == null) {
                WorkerCacheUpdateResult result = buildFailureResult(engineIpPort, "Worker Cached Keys is null");
                cacheMetricsReporter.reportUpdateEngineBlockCacheRT(role, startTime, "0");
                return result;
            }

            Set<Long> cachedKeys = cacheStatus.getCachedKeys();
            standby.observe(workerStatus);
            if (configuration.isKvcmEnabled()) {
                return WorkerCacheUpdateResult.builder().success(false).engineIpPort(ipPort)
                        .errorMessage("LOCAL_SYNC indexing disabled in KVCM mode").build();
            }

            // Update cache
            kvCacheManager.updateEngineCache(ipPort, role, cachedKeys, pollTrace);

            WorkerCacheUpdateResult result = buildSuccessResult(workerStatus, cacheStatus);

            cacheMetricsReporter.reportUpdateEngineBlockCacheRT(role, startTime, "1");

            return result;

        } catch (Throwable e) {
            log.error("Error updating worker cache for: {}", engineIpPort, e);

            WorkerCacheUpdateResult result = buildFailureResult(engineIpPort, e.getMessage());

            cacheMetricsReporter.reportUpdateEngineBlockCacheRT(role, startTime, "0");

            return result;
        }
    }

    @Override
    public void removeEngineBlockCache(
            WorkerStatus workerStatus, EndpointTrace trace) {
        requireCacheMutationLock(workerStatus);
        standby.remove(workerStatus);
        kvCacheManager.removeEngineCache(workerStatus.getIpPort(), trace);
    }

    private static void requireCacheMutationLock(WorkerStatus workerStatus) {
        java.util.Objects.requireNonNull(workerStatus, "workerStatus");
        if (!workerStatus.cacheMutationLock.isHeldByCurrentThread()) {
            throw new IllegalStateException("Cache mutation requires the owning generation's cacheMutationLock");
        }
    }

    /**
     * Build success result
     */
    private WorkerCacheUpdateResult buildSuccessResult(WorkerStatus workerStatus, CacheStatus cacheStatus) {
        return WorkerCacheUpdateResult.builder()
            .success(true)
            .engineIpPort(workerStatus.getIpPort())
            .cacheBlockCount(cacheStatus.getCachedKeys() != null ? cacheStatus.getCachedKeys().size() : 0)
            .availableKvCache(cacheStatus.getAvailableKvCache())
            .totalKvCache(cacheStatus.getTotalKvCache())
            .cacheVersion(cacheStatus.getVersion())
            .build();
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
