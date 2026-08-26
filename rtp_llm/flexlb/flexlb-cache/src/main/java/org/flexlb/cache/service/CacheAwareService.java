package org.flexlb.cache.service;

import org.flexlb.cache.domain.CacheMatch;
import org.flexlb.cache.domain.EngineGeneration;
import org.flexlb.cache.domain.WorkerCacheUpdateResult;
import org.flexlb.dao.master.CacheStatus;
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
     * @param roleType       Engine role used for lookup telemetry
     * @param candidates     Exact endpoint generations eligible for this request
     * @return prefix matches keyed by the same exact generation identity
     */
    Map<EngineGeneration, CacheMatch> findMatchingEngines(
            List<Long> blockCacheKeys,
            RoleType roleType,
            List<EngineGeneration> candidates);
    
    /**
     * Publish a newly discovered engine generation before any cache poll is
     * submitted for it. Repeating the active generation is idempotent; an
     * older generation is rejected.
     */
    boolean activateEngineGeneration(
            String engineIpPort,
            RoleType roleType,
            long generationId);

    /**
     * Replace the cache view of one exact engine generation.
     */
    WorkerCacheUpdateResult updateEngineBlockCache(
            String engineIpPort,
            RoleType roleType,
            long generationId,
            CacheStatus cacheStatus);

    /**
     * Retire one exact generation. A delayed retirement cannot clear a newer
     * generation published at the same address.
     */
    boolean retireEngineGeneration(
            String engineIpPort,
            RoleType roleType,
            long generationId);
}
