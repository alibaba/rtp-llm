package org.flexlb.cache.service;

import org.flexlb.cache.domain.WorkerCacheUpdateResult;
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
     * @param group          Engine group to query
     * @return Engine matching result map, key: engineIpPort, value: prefixMatchLength
     */
    Map<String/*engineIpPort*/, Integer/*prefixMatchLength*/> findMatchingEngines(List<Long> blockCacheKeys, RoleType roleType, String group);
    
    /**
     * Update engine block KV cache status
     *
     * @param workerStatus Worker status information
     * @return Update result
     */
    WorkerCacheUpdateResult updateEngineBlockCache(WorkerStatus workerStatus);
}