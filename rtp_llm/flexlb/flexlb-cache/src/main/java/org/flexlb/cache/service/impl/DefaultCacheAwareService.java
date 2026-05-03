package org.flexlb.cache.service.impl;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.cache.core.DpGroupTopology;
import org.flexlb.cache.core.DpRankAddress;
import org.flexlb.cache.core.EngineLocalView;
import org.flexlb.cache.core.KvCacheManager;
import org.flexlb.cache.domain.WorkerCacheUpdateResult;
import org.flexlb.cache.monitor.CacheMetricsReporter;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.DpRankCacheStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
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
    private EngineLocalView engineLocalView;

    @Autowired
    private DpGroupTopology dpGroupTopology;

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
    public int findMatchingPrefixLength(String engineIpPort, List<Long> blockCacheKeys) {
        if (engineIpPort == null || blockCacheKeys == null || blockCacheKeys.isEmpty()) {
            return 0;
        }
        try {
            return kvCacheManager.findMatchingPrefixLength(engineIpPort, blockCacheKeys);
        } catch (Exception e) {
            log.error("Error computing single-engine prefix match for: {}", engineIpPort, e);
            return 0;
        }
    }

    @Override
    public Map<Integer, Integer> findMatchingRanksInGroup(String groupIpPort, List<Long> blockCacheKeys) {
        if (groupIpPort == null || blockCacheKeys == null || blockCacheKeys.isEmpty()) {
            return Collections.emptyMap();
        }
        try {
            return kvCacheManager.findMatchingRanksInGroup(groupIpPort, blockCacheKeys);
        } catch (Exception e) {
            log.error("Error finding per-rank matches for group: {}", groupIpPort, e);
            return Collections.emptyMap();
        }
    }
    
    @Override
    public WorkerCacheUpdateResult updateEngineBlockCache(WorkerStatus workerStatus) {
        long startTime = System.nanoTime() / 1000;
        String engineIpPort = workerStatus.getIpPort();
        String role = workerStatus.getRole();

        try {
            if (workerStatus.getCacheStatus() == null) {
                WorkerCacheUpdateResult result = buildFailureResult(engineIpPort, "Worker Cache Status is null");
                cacheMetricsReporter.reportUpdateEngineBlockCacheRT(engineIpPort, role, startTime, "0");
                return result;
            }

            String ipPort = workerStatus.getIpPort();
            CacheStatus cacheStatus = workerStatus.getCacheStatus();
            if (cacheStatus.getCachedKeys() == null) {
                WorkerCacheUpdateResult result = buildFailureResult(engineIpPort, "Worker Cached Keys is null");
                cacheMetricsReporter.reportUpdateEngineBlockCacheRT(engineIpPort, role, startTime, "0");
                return result;
            }

            Set<Long> cachedKeys = cacheStatus.getCachedKeys();

            // Update cache
            kvCacheManager.updateEngineCache(ipPort, role, cachedKeys);

            // Lazily refresh the per-rank secondary view + DP group topology when DP0
            // reports a breakdown. Both are additive: ShortestTTFT / WeightedCache
            // continue to use the union view via cacheStatus.getCachedKeys() and are
            // unaffected.
            List<DpRankCacheStatus> dpCaches = cacheStatus.getDpCaches();
            if (dpCaches != null && !dpCaches.isEmpty()) {
                List<Set<Long>> rankBlocks = new ArrayList<>(dpCaches.size());
                List<DpRankAddress> rankAddrs = new ArrayList<>(dpCaches.size());
                for (DpRankCacheStatus rank : dpCaches) {
                    rankBlocks.add(rank.cachedKeys() != null ? rank.cachedKeys() : Collections.emptySet());
                    rankAddrs.add(new DpRankAddress(rank.dpRank(), rank.ip(), rank.grpcPort()));
                }
                engineLocalView.replacePerRankBlocks(ipPort, rankBlocks);
                dpGroupTopology.update(ipPort, rankAddrs);
            }

            WorkerCacheUpdateResult result = buildSuccessResult(workerStatus, cacheStatus);

            cacheMetricsReporter.reportUpdateEngineBlockCacheRT(ipPort, role, startTime, "1");
            
            return result;
                
        } catch (Throwable e) {
            log.error("Error updating worker cache for: {}", engineIpPort, e);
            
            WorkerCacheUpdateResult result = buildFailureResult(engineIpPort, e.getMessage());

            cacheMetricsReporter.reportUpdateEngineBlockCacheRT(engineIpPort, role, startTime, "0");
            
            return result;
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