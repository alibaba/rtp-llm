package org.flexlb.cache.service;

import org.flexlb.cache.domain.WorkerCacheUpdateResult;
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
     * @return Engine matching result map, key: engineIpPort, value: prefixMatchLength
     */
    Map<String/*engineIpPort*/, Integer/*prefixMatchLength*/> findMatchingEngines(
            List<Long> blockCacheKeys,
            RoleType roleType,
            List<String> candidateEngineIpPorts);
    
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
}
