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
     * Single-engine prefix-match length (in blocks). DP-aware: MAX-per-rank
     * for DP-enabled engines, union for non-DP — same semantics as
     * {@link #findMatchingEngines} but for one already-known engine.
     *
     * <p>Used by post-selection accounting paths (e.g., WeightedCache
     * filling TaskInfo.prefixLength) that need an honest hit estimate
     * without re-scoring the whole role list.
     */
    int findMatchingPrefixLength(String engineIpPort, List<Long> blockCacheKeys);

    /**
     * Per-rank prefix-match inside one DP group. Counterpart to
     * {@link #findMatchingEngines} for DP-aware strategies that want to know
     * which rank inside a chosen group already owns the prefix.
     *
     * <p>Returns an empty map when the group is not DP-enabled or has not
     * been synced yet. Does NOT affect the legacy union-based path.
     *
     * @param groupIpPort    The DP group identifier (= DP0's ip:port)
     * @param blockCacheKeys Ordered list of prefix block hashes
     * @return Map keyed by {@code dp_rank}, value is prefix-match length in blocks
     */
    Map<Integer/*dpRank*/, Integer/*prefixMatchLength*/> findMatchingRanksInGroup(String groupIpPort, List<Long> blockCacheKeys);

    /**
     * Update engine block KV cache status
     *
     * @param workerStatus Worker status information
     * @return Update result
     */
    WorkerCacheUpdateResult updateEngineBlockCache(WorkerStatus workerStatus);
}