package org.flexlb.cache.core;

import com.google.common.collect.Sets;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.LongAdder;
import java.util.concurrent.locks.ReentrantLock;

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
     * Read-write lock for data consistency
     */
    private final ReentrantLock lock = new ReentrantLock();
    
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
    public void addCacheBlock(Long blockCacheKey, String engineIpPort) {
        if (blockCacheKey == null || engineIpPort == null) {
            log.warn("Invalid parameters: blockCacheKey={}, engineIpPort={}", blockCacheKey, engineIpPort);
            return;
        }
        
        lock.lock();
        try {
            Set<String> engines = blockToEnginesMap.computeIfAbsent(blockCacheKey, k -> {
                totalBlocks.increment();
                return Sets.newConcurrentHashSet();
            });

            boolean added = engines.add(engineIpPort);
            if (added) {
                totalMappings.increment();
            }
        } finally {
            lock.unlock();
        }
    }

    /**
     * Remove cache block from specified engine
     *
     * @param engineIp      Engine IP
     * @param blockCacheKey Cache block hash value
     */
    public void removeCacheBlock(String engineIp, Long blockCacheKey) {
        if (blockCacheKey == null || engineIp == null) {
            return;
        }
        
        lock.lock();
        try {
            Set<String> engines = blockToEnginesMap.get(blockCacheKey);
            if (engines == null) {
                return;
            }

            boolean removed = engines.remove(engineIp);
            if (removed) {
                totalMappings.decrement();

                // Remove entire entry if no engine owns this cache block
                if (engines.isEmpty()) {
                    blockToEnginesMap.remove(blockCacheKey);
                    totalBlocks.decrement();
                }
            }
        } finally {
            lock.unlock();
        }
    }

    /**
     * Remove an engine
     *
     * @param engineIp Engine IP
     */
    public void removeAllCacheBlockOfEngine(String engineIp) {
        if (engineIp == null) {
            return;
        }

        lock.lock();
        try {
            blockToEnginesMap.forEach((blockCacheKey, engines) -> {
                boolean removed = engines.remove(engineIp);
                if (removed) {
                    totalMappings.decrement();

                    // Remove entire entry if no engine owns this cache block
                    if (engines.isEmpty()) {
                        blockToEnginesMap.remove(blockCacheKey);
                        totalBlocks.decrement();
                    }
                }
            });
        } finally {
            lock.unlock();
        }
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

            // Set of engines with confirmed match length
            Set<String> confirmedEngines = Sets.newHashSet();
            // Filter candidate engines: only keep engines that exist in current block
            for (String candidateEngine : candidateEngines) {
                if (blockOwners.isEmpty() || !blockOwners.contains(candidateEngine)) {
                    // This engine does not exist in current block, prefix match interrupted
                    result.put(candidateEngine, i);
                    confirmedEngines.add(candidateEngine);
                }
            }

            // Remove engines with confirmed prefix length from candidate set
            candidateEngines.removeAll(confirmedEngines);

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