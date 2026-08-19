package org.flexlb.cache.service;

import org.flexlb.cache.domain.WorkerCacheUpdateResult;
import org.flexlb.dao.route.RoleType;

import java.util.List;
import java.util.Map;
import java.util.Set;

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
     * Synchronously publish one engine's complete cache-key snapshot.
     *
     * <p>The caller retains ownership of {@code cachedKeys}; implementations
     * must consume the set before returning and must not retain the reference.
     * Cache lookup is advisory and may observe an update in progress; writers
     * are serialized per engine address, while readers remain lock-free.
     */
    WorkerCacheUpdateResult publishEngineCacheSnapshot(
            String engineIpPort, RoleType roleType, Set<Long> cachedKeys);

    /** Remove all cache-locality state for an engine address. */
    void clearEngineCache(String engineIpPort);
}
