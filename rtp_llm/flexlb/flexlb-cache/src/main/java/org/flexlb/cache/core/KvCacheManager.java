package org.flexlb.cache.core;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.flexlb.cache.domain.DiffResult;
import org.flexlb.cache.monitor.CacheMetricsReporter;
import org.flexlb.dao.master.WorkerStatusProvider;
import org.flexlb.dao.route.RoleType;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import javax.annotation.PreDestroy;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.LongAdder;

/**
 * KV cache manager
 * Core functions:
 * 1. Unified management of two-level hash table
 * 2. Provide advanced cache query and matching services
 *
 * @author FlexLB
 */
@Slf4j
@Getter
@Component
public class KvCacheManager {
    
    @Autowired
    private GlobalCacheIndex globalCacheIndex;
    
    @Autowired
    private EngineLocalView engineLocalView;
    
    @Autowired
    private WorkerStatusProvider workerStatusProvider;
    
    /**
     * Cache metrics reporter
     */
    @Autowired
    private CacheMetricsReporter cacheMetricsReporter;

    /**
     * Performance statistics
     */
    private final LongAdder totalUpdates = new LongAdder();

    @PostConstruct
    public void init() {
        log.info("KvCacheManager initialized successfully");
    }
    
    @PreDestroy
    public void destroy() {
        log.info("KvCacheManager shutting down...");
        clear();
    }

    /**
     * Query engine cache matching status.
     *
     * <p>For DP-enabled engines (those whose DP0 has reported a {@code dp_cache[]}
     * breakdown), the returned length is the <b>MAX prefix-match across ranks</b> —
     * i.e., the longest cached prefix that any single rank inside the pod can
     * serve. For non-DP engines (or DP engines whose breakdown has not been
     * synced yet), the returned length stays the union view from
     * {@link GlobalCacheIndex} — exactly the legacy behaviour.
     *
     * <h3>Why MAX, not UNION, for DP engines</h3>
     * Each DP rank typically owns an independent KV-cache shard. UNION counts
     * a block as "the engine has it" if any rank does, which over-estimates by
     * up to dpSize× — a single request only ever lands on one rank and can't
     * "stitch" the prefix across ranks. MAX is the tightest upper bound on
     * what a request CAN get if the engine routes it to the best rank.
     *
     * <p>This keeps the routing structure exactly as before
     * ({@code Map<engineIpPort, prefix>} per group), and ShortestTTFT /
     * WeightedCache consume an honest score regardless of {@code dp_size}.
     *
     * @param blockCacheKeys List of cache block hash values to query
     * @param roleType       Engine role to query
     * @param group          Engine group to query
     * @return Engine matching result map, key: engineIpPort, value: prefixMatchLength (in blocks)
     */
    public Map<String/*engineIpPort*/, Integer/*prefixMatchLength*/> findMatchingEngines(List<Long> blockCacheKeys,
        RoleType roleType, String group) {

        if (blockCacheKeys == null || blockCacheKeys.isEmpty()) {
            return Collections.emptyMap();
        }

        List<String> enginesIpPorts = workerStatusProvider.getWorkerIpPorts(roleType, group);
        if (enginesIpPorts == null || enginesIpPorts.isEmpty()) {
            return Collections.emptyMap();
        }

        // Split engines into DP-aware (have per-rank breakdown) and legacy
        // (no per-rank info — single rank or not yet synced).
        Map<String, Integer> result = new HashMap<>(enginesIpPorts.size());
        List<String> legacyEngines = new ArrayList<>(enginesIpPorts.size());
        for (String engine : enginesIpPorts) {
            if (engineLocalView.getPerRankCount(engine) > 0) {
                result.put(engine, maxPerRankPrefix(engine, blockCacheKeys));
            } else {
                legacyEngines.add(engine);
            }
        }

        if (!legacyEngines.isEmpty()) {
            result.putAll(globalCacheIndex.batchCalculatePrefixMatchLength(legacyEngines, blockCacheKeys));
        }
        return result;
    }

    /**
     * MAX per-rank prefix match for one DP-enabled engine. Reused by
     * {@link #findMatchingEngines} to score DP engines without touching the
     * legacy global-index path.
     */
    private int maxPerRankPrefix(String groupIpPort, List<Long> blockCacheKeys) {
        int rankCount = engineLocalView.getPerRankCount(groupIpPort);
        int max = 0;
        for (int rank = 0; rank < rankCount; rank++) {
            Set<Long> blocks = engineLocalView.getPerRankBlocks(groupIpPort, rank);
            int prefix = 0;
            for (Long key : blockCacheKeys) {
                if (blocks.isEmpty() || !blocks.contains(key)) {
                    break;
                }
                prefix++;
            }
            if (prefix > max) {
                max = prefix;
            }
        }
        return max;
    }

    /**
     * Single-engine prefix-match. Same DP-aware semantics as
     * {@link #findMatchingEngines}: MAX-per-rank when the engine has a
     * per-rank breakdown, falls back to the legacy union view from
     * {@link GlobalCacheIndex} otherwise.
     *
     * <p>Use this when the caller already knows the engine (e.g.,
     * post-selection accounting in WeightedCacheLoadBalancer) and doesn't
     * need to score a whole role list. For role-wide scoring use
     * {@link #findMatchingEngines}.
     */
    public int findMatchingPrefixLength(String engineIpPort, List<Long> blockCacheKeys) {
        if (engineIpPort == null || blockCacheKeys == null || blockCacheKeys.isEmpty()) {
            return 0;
        }
        if (engineLocalView.getPerRankCount(engineIpPort) > 0) {
            return maxPerRankPrefix(engineIpPort, blockCacheKeys);
        }
        Map<String, Integer> single = globalCacheIndex.batchCalculatePrefixMatchLength(
                Collections.singletonList(engineIpPort), blockCacheKeys);
        return single.getOrDefault(engineIpPort, 0);
    }

    /**
     * Per-rank prefix-match query inside ONE DP group.
     *
     * <p>Counterpart to {@link #findMatchingEngines}: where that method returns a
     * group-level (union) prefix length suitable for cross-group routing
     * decisions, this returns a per-rank breakdown suitable for "which rank
     * inside the chosen group already owns the prefix" decisions.
     *
     * <p>The returned map is keyed by {@code dp_rank} (0..dpSize-1) and the
     * value is the prefix-match length in BLOCKS (caller multiplies by
     * blockSize to convert to tokens, same convention as
     * {@link #findMatchingEngines}).
     *
     * <p>Returns an empty map when the group is not DP-enabled, has not been
     * synced yet, or the key list is empty. Cost is
     * {@code O(dpSize × prefixLen)} per call — no global per-rank index is
     * maintained because typical queries are scoped to a single group already
     * chosen by the caller.
     *
     * <p>Does NOT mutate any state. Does NOT affect ShortestTTFT /
     * WeightedCache cache-hit math: those keep using {@link #findMatchingEngines}
     * with the union view via {@link GlobalCacheIndex}.
     */
    public Map<Integer /*dpRank*/, Integer /*prefixMatchLength*/> findMatchingRanksInGroup(
            String groupIpPort, List<Long> blockCacheKeys) {
        if (groupIpPort == null || blockCacheKeys == null || blockCacheKeys.isEmpty()) {
            return Collections.emptyMap();
        }
        int rankCount = engineLocalView.getPerRankCount(groupIpPort);
        if (rankCount == 0) {
            return Collections.emptyMap();
        }
        Map<Integer, Integer> result = new HashMap<>(rankCount);
        for (int rank = 0; rank < rankCount; rank++) {
            Set<Long> blocks = engineLocalView.getPerRankBlocks(groupIpPort, rank);
            int prefixLen = 0;
            for (Long key : blockCacheKeys) {
                if (blocks.isEmpty() || !blocks.contains(key)) {
                    break;
                }
                prefixLen++;
            }
            result.put(rank, prefixLen);
        }
        return result;
    }

    /**
     * Update engine cache status
     *
     * @param engineIPort    Engine IP:Port
     * @param role           Engine role
     * @param newCacheBlocks New cache block set (blockCacheKeys)
     */
    public void updateEngineCache(String engineIPort, String role, Set<Long> newCacheBlocks) {
        if (engineIPort == null || newCacheBlocks == null) {
            DiffResult.empty(engineIPort);
            return;
        }

        // Calculate diff
        DiffResult diffResult = engineLocalView.calculateDiff(engineIPort, newCacheBlocks, role);
        if (!diffResult.hasChanges()) {
            return;
        }

        // Apply added cache blocks
        for (Long addedBlock : diffResult.getAddedBlocks()) {
            boolean contains = newCacheBlocks.contains(addedBlock);
            if (contains) {
                // Update local view
                engineLocalView.addOrUpdateCacheBlock(engineIPort, addedBlock);
                // Update global index
                globalCacheIndex.addCacheBlock(addedBlock, engineIPort);
            }
        }

        // Apply removed cache blocks
        for (Long removedBlock : diffResult.getRemovedBlocks()) {
            // Remove from local view
            engineLocalView.removeCacheBlock(engineIPort, removedBlock);
            // Remove from global index
            globalCacheIndex.removeCacheBlock(engineIPort, removedBlock);
        }

        totalUpdates.increment();
        // Report metrics
        cacheMetricsReporter.reportEngineLocalMetrics(engineIPort, role, engineLocalView.size(engineIPort));
        cacheMetricsReporter.reportGlobalCacheMetrics(globalCacheIndex.totalBlocks(), globalCacheIndex.totalMappings());
        cacheMetricsReporter.reportEngineViewsMapSize(engineLocalView.getEngineViewsMapSize());
    }
    
    /**
     * Clear all data
     */
    public void clear() {

        globalCacheIndex.clear();
        engineLocalView.clear();

        totalUpdates.reset();
        // Report
        cacheMetricsReporter.reportGlobalCacheMetrics(globalCacheIndex.totalBlocks(), globalCacheIndex.totalMappings());

        log.info("Cleared all cache data");
    }
}