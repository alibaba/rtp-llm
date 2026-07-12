package org.flexlb.cache.service;

import org.flexlb.cache.domain.WorkerCacheUpdateResult;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;

import java.util.List;
import java.util.Map;

/**
 * Provides cache metadata operations for one source.
 */
public interface CacheMatchProvider {

    CacheMatchSource source();

    Map<String, Integer> findMatchingEngines(
            List<Long> blockCacheKeys,
            RoleType roleType,
            String group);

    /**
     * Applies a worker cache update when supported by the metadata source.
     */
    WorkerCacheUpdateResult updateEngineBlockCache(WorkerStatus workerStatus);
}
