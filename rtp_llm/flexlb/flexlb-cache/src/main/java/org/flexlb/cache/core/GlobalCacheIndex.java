package org.flexlb.cache.core;

import com.google.common.collect.Sets;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.LongAdder;

/**
 * Global cache index (large hash table)
 * Manages block_hash_id -> Set<EngineIP:EnginePort> mapping
 *
 * @author FlexLB
 */
@Slf4j
@Component
public class GlobalCacheIndex {

    /**
     * Core storage structure: block_hash_id -> Set<engine_ip:engine_port>
     */
    private final ConcurrentHashMap<Long, Set<String>> blockToEnginesMap = new ConcurrentHashMap<>();

    /**
     * Statistics
     */
    private final LongAdder totalBlocks = new LongAdder();
    private final LongAdder totalMappings = new LongAdder();

    /**
     * Add cache block to specified engine
     *
     * @param blockCacheKey Cache block hash value
     * @param engineIpPort  Engine IP:Port
     */
    void addCacheBlock(Long blockCacheKey, String engineIpPort) {
        if (blockCacheKey == null || engineIpPort == null) {
            log.warn("Invalid parameters: blockCacheKey={}, engineIpPort={}", blockCacheKey, engineIpPort);
            return;
        }

        blockToEnginesMap.compute(blockCacheKey, (ignored, existing) -> {
            Set<String> engines = existing;
            if (engines == null) {
                totalBlocks.increment();
                engines = Sets.newConcurrentHashSet();
            }
            if (engines.add(engineIpPort)) {
                totalMappings.increment();
            }
            return engines;
        });
    }

    /**
     * Remove cache block from specified engine
     *
     * @param engineIp      Engine IP
     * @param blockCacheKey Cache block hash value
     */
    void removeCacheBlock(String engineIp, Long blockCacheKey) {
        if (blockCacheKey == null || engineIp == null) {
            return;
        }

        blockToEnginesMap.computeIfPresent(blockCacheKey, (ignored, engines) -> {
            if (engines.remove(engineIp)) {
                totalMappings.decrement();
                if (engines.isEmpty()) {
                    totalBlocks.decrement();
                    return null;
                }
            }
            return engines;
        });
    }

    /**
     * Apply one address snapshot while the caller's old local view remains the
     * commit marker. Replaying the method after a partial failure is safe.
     */
    CacheDiffStats applyEngineCacheSnapshot(String engineIpPort,
                                            Set<Long> oldCacheBlocks,
                                            Set<Long> newCacheBlocks) {
        int added = 0;
        int removed = 0;
        for (Long block : newCacheBlocks) {
            if (!oldCacheBlocks.contains(block)) {
                added++;
                // The local view advances only after every global mutation
                // succeeds. Therefore old members are already committed and
                // only the logical delta needs a global CHM write.
                addCacheBlock(block, engineIpPort);
            }
        }
        for (Long block : oldCacheBlocks) {
            if (!newCacheBlocks.contains(block)) {
                removed++;
                removeCacheBlock(engineIpPort, block);
            }
        }
        return new CacheDiffStats(added, removed);
    }

    void removeEngineCacheBlocks(String engineIpPort, Set<Long> cacheBlocks) {
        for (Long block : cacheBlocks) {
            removeCacheBlock(engineIpPort, block);
        }
    }

    record CacheDiffStats(int added, int removed) {
    }

    /**
     * Calculate engine prefix match length based on prefix matching
     *
     * @param engineIpPorts  Engine IP:Port list
     * @param blockCacheKeys Ordered cache block hash value list
     * @return Map<EngineIP:EnginePort, PrefixMatchLength>
     */
    public Map<String, Integer> batchCalculatePrefixMatchLength(List<String> engineIpPorts,
                                                                List<Long> blockCacheKeys) {

        if (isEmpty(engineIpPorts) || isEmpty(blockCacheKeys)) {
            return Collections.emptyMap();
        }
        return calculatePrefixMatchLength(engineIpPorts, blockCacheKeys);
    }

    /**
     * Prefix match calculation
     *
     * @param engineIpPorts  Engine IP:Port list
     * @param blockCacheKeys Ordered cache block hash value list
     * @return Map<EngineIP:EnginePort, PrefixMatchLength>
     */
    private Map<String, Integer> calculatePrefixMatchLength(List<String> engineIpPorts,
                                                            List<Long> blockCacheKeys) {

        Map<String, Integer> result = new HashMap<>(engineIpPorts.size());

        // Initialize all engines as candidates, set of engines with undetermined prefix length
        Set<String> candidateEngines = Sets.newHashSet(engineIpPorts);

        // Iterate through each block, gradually filter candidate engines
        for (int i = 0; i < blockCacheKeys.size(); i++) {
            Long blockCacheKey = blockCacheKeys.get(i);
            Set<String> blockOwners = getEnginesForBlock(blockCacheKey);

            // Filter candidate engines: only keep engines that exist in current block
            Iterator<String> candidates = candidateEngines.iterator();
            while (candidates.hasNext()) {
                String candidateEngine = candidates.next();
                if (blockOwners.isEmpty() || !blockOwners.contains(candidateEngine)) {
                    // This engine does not exist in current block, prefix match interrupted
                    result.put(candidateEngine, i);
                    candidates.remove();
                }
            }

            // Exit early if no candidate engines remain
            if (candidateEngines.isEmpty()) {
                break;
            }
        }

        // Process remaining candidate engines (they matched all blocks)
        for (String remainingEngine : candidateEngines) {
            result.put(remainingEngine, blockCacheKeys.size());
        }

        return result;
    }

    /**
     * Check if collection is empty
     */
    private boolean isEmpty(List<?> list) {
        return list == null || list.isEmpty();
    }

    /**
     * Get engine set for specified cache block
     */
    private Set<String> getEnginesForBlock(Long blockCacheKey) {
        if (blockCacheKey == null) {
            return Collections.emptySet();
        }
        Set<String> engines = blockToEnginesMap.get(blockCacheKey);
        return engines != null ? engines : Collections.emptySet();
    }

    /**
     * Clear all data
     */
    public void clear() {

        blockToEnginesMap.clear();
        totalBlocks.reset();
        totalMappings.reset();
        log.info("Cleared global cache index");
    }

    public long totalBlocks() {
        return totalBlocks.sum();
    }

    public long totalMappings() {
        return totalMappings.sum();
    }
}
