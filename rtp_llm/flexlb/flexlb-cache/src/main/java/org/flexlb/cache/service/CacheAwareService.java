package org.flexlb.cache.service;

import org.flexlb.cache.domain.WorkerCacheUpdateResult;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;

import java.util.List;

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
     * @param requestId      Request ID used to correlate cache queries
     * @param blockCacheKeys List of cache block IDs to query
     * @param blockSize      Token count represented by each cache block
     * @param roleType       Engine role to query
     * @param group          Engine group to query
     * @return Cache matching result and provider query time
     */
    CacheMatchResult findMatchingEngines(
            String requestId,
            List<Long> blockCacheKeys,
            long blockSize,
            RoleType roleType,
            String group);
    
    /**
     * Update engine block KV cache status
     *
     * @param workerStatus Worker status information
     * @return Update result
     */
    WorkerCacheUpdateResult updateEngineBlockCache(WorkerStatus workerStatus);
}
