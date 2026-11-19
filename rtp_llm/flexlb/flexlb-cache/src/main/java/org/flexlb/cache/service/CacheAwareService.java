package org.flexlb.cache.service;

import org.flexlb.cache.domain.WorkerCacheUpdateResult;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.EndpointTrace;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;

import java.util.List;
import java.util.Map;

/**
 * Cache-aware service interface
 * Provides unified cache management interface for external modules
 *
 * @author FlexLB
 */
public interface CacheAwareService {
    /**
     * Find matching engines
     *
     * @param blockCacheKeys List of cache block IDs to query
     * @param roleType       Engine role to query
     * @param candidateEngineIpPorts exact live candidate addresses
     * @param blockSize routed key granularity supplied by Frontend (tokens per key)
     * @return Engine matching result map, key: engineIpPort, value: prefixMatchLength
     */
    default Map<String, Integer> findMatchingEngines(
            List<Long> blockCacheKeys,
            RoleType roleType,
            List<String> candidateEngineIpPorts, long blockSize) {
        return lookup(blockCacheKeys, roleType, candidateEngineIpPorts, blockSize).matches();
    }

    CacheMatch lookup(List<Long> keys, RoleType role, List<String> candidates, long blockSize);

    /** Records a successful route for standby cache prediction. */
    void recordRoutedRequest(Request request, List<ServerStatus> selected);

    default void recordRoutedRequest(Request request, List<ServerStatus> selected, long batchId) {
        recordRoutedRequest(request, selected);
    }

    default void observeCacheHitFeedback(WorkerStatus worker, WorkerStatus.StatusObservation observation) {}

    /**
     * Update engine block KV cache status
     *
     * <p>The caller must hold workerStatus.cacheMutationLock throughout the update.
     *
     * @param workerStatus Worker status information
     * @return Update result
     */
    WorkerCacheUpdateResult updateEngineBlockCache(
            WorkerStatus workerStatus, EndpointTrace.Poll pollTrace);

    /**
     * Remove every cached block associated with one engine address.
     *
     * <p>Callers must fence the address against generation replacement before
     * invoking this method, and hold workerStatus.cacheMutationLock from reading
     * the local key set until cleanup completes. The cache index is address based and therefore
     * cannot distinguish two worker generations by itself.
     */
    void removeEngineBlockCache(WorkerStatus workerStatus, EndpointTrace trace);

    /**
     * Routing matches and whether they came from a successful KVCM query.
     */
    record CacheMatch(Map<String, Integer> matches, boolean kvcm, Map<String, Long> globalMatches) {
        public CacheMatch(Map<String, Integer> matches, boolean kvcm) {
            this(matches, kvcm, Map.of());
        }
    }
}
