package org.flexlb.cache.core;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.util.SyncDiagnostics;
import org.springframework.stereotype.Component;

import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
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

    private static final int CACHE_REMOVE_LOCK_CHUNK_SIZE = 256;

    /**
     * Core storage structure: block_hash_id -> Set<engine_ip:engine_port>
     */
    private final ConcurrentHashMap<Long, Set<String>> blockToEnginesMap = new ConcurrentHashMap<>();

    /**
     * Serializes index mutations and mapping-count updates
     */
    private final ReentrantLock lock = new ReentrantLock();

    /**
     * Statistics
     */
    private final LongAdder totalMappings = new LongAdder();
    /** Per-caller compaction storage; cache queries never retain this array. */
    private final ThreadLocal<String[]> prefixCandidates =
            ThreadLocal.withInitial(() -> new String[0]);

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

        long diagStart = SyncDiagnostics.ENABLED ? System.nanoTime() : 0L;
        lock.lock();
        long diagAcquired = SyncDiagnostics.ENABLED ? System.nanoTime() : 0L;
        try {
            Set<String> engines = blockToEnginesMap.computeIfAbsent(
                    blockCacheKey, k -> ConcurrentHashMap.newKeySet());

            boolean added = engines.add(engineIpPort);
            if (added) {
                totalMappings.increment();
            }
        } finally {
            long diagReleased = SyncDiagnostics.ENABLED ? System.nanoTime() : 0L;
            lock.unlock();
            if (SyncDiagnostics.ENABLED) {
                SyncDiagnostics.cacheLock("add", engineIpPort, diagStart, diagAcquired, diagReleased, blockToEnginesMap.size());
            }
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

        long diagStart = SyncDiagnostics.ENABLED ? System.nanoTime() : 0L;
        lock.lock();
        long diagAcquired = SyncDiagnostics.ENABLED ? System.nanoTime() : 0L;
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
                }
            }
        } finally {
            long diagReleased = SyncDiagnostics.ENABLED ? System.nanoTime() : 0L;
            lock.unlock();
            if (SyncDiagnostics.ENABLED) {
                SyncDiagnostics.cacheLock("remove", engineIp, diagStart, diagAcquired, diagReleased, blockToEnginesMap.size());
            }
        }
    }

    /**
     * Remove an engine
     *
     * @param engineIp Engine IP
     * @param blockCacheKeys Detached local keys of this generation; the caller
     *                       must serialize retirement against updates of the same engine
     */
    public void removeAllCacheBlockOfEngine(String engineIp, Set<Long> blockCacheKeys) {
        if (engineIp == null || blockCacheKeys == null || blockCacheKeys.isEmpty()) {
            return;
        }
        // Use the detached local view; do not scan blocks owned only by other workers.
        // Bound each critical section so surviving workers can keep updating their indexes.
        Iterator<Long> keys = blockCacheKeys.iterator();
        long start = SyncDiagnostics.ENABLED ? System.nanoTime() : 0L;
        long wait = 0L;
        long hold = 0L;
        long maxHold = 0L;
        int acquisitions = 0;
        while (keys.hasNext()) {
            long waiting = SyncDiagnostics.ENABLED ? System.nanoTime() : 0L;
            lock.lock();
            long acquired = SyncDiagnostics.ENABLED ? System.nanoTime() : 0L;
            try {
                for (int count = 0; count < CACHE_REMOVE_LOCK_CHUNK_SIZE && keys.hasNext(); count++) {
                    Long key = keys.next();
                    Set<String> engines = blockToEnginesMap.get(key);
                    if (engines != null && engines.remove(engineIp)) {
                        totalMappings.decrement();
                        if (engines.isEmpty()) {
                            blockToEnginesMap.remove(key);
                        }
                    }
                }
            } finally {
                long released = SyncDiagnostics.ENABLED ? System.nanoTime() : 0L;
                lock.unlock();
                wait += acquired - waiting;
                hold += released - acquired;
                maxHold = Math.max(maxHold, released - acquired);
                acquisitions++;
            }
        }
        if (SyncDiagnostics.ENABLED) {
            SyncDiagnostics.cacheRemoval(engineIp, start, wait, hold, maxHold,
                    acquisitions, blockCacheKeys.size(), blockToEnginesMap.size());
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

        Set<String> firstBlockOwners = getEnginesForBlock(
                blockCacheKeys.getFirst());
        if (firstBlockOwners.isEmpty()) {
            return Collections.emptyMap();
        }

        // Compact matching addresses in place. The selector already owns a
        // unique immutable fleet view, so a String[] is sufficient here and
        // avoids one HashSet node per engine on every request.
        String[] candidates = prefixCandidates.get();
        if (candidates.length < engineIpPorts.size()) {
            candidates = new String[engineIpPorts.size()];
            prefixCandidates.set(candidates);
        }
        for (int index = 0; index < engineIpPorts.size(); index++) {
            candidates[index] = engineIpPorts.get(index);
        }
        int survivorCount = 0;
        for (int candidateIndex = 0;
                candidateIndex < engineIpPorts.size(); candidateIndex++) {
            String candidate = candidates[candidateIndex];
            if (firstBlockOwners.contains(candidate)) {
                candidates[survivorCount++] = candidate;
            }
        }
        if (survivorCount == 0) {
            return Collections.emptyMap();
        }

        Map<String, Integer> result = null;
        for (int blockIndex = 1;
                blockIndex < blockCacheKeys.size(); blockIndex++) {
            Set<String> blockOwners = getEnginesForBlock(
                    blockCacheKeys.get(blockIndex));
            int nextSurvivorCount = 0;
            for (int candidateIndex = 0;
                    candidateIndex < survivorCount; candidateIndex++) {
                String candidate = candidates[candidateIndex];
                if (blockOwners.contains(candidate)) {
                    candidates[nextSurvivorCount++] = candidate;
                } else {
                    if (result == null) {
                        result = new HashMap<>();
                    }
                    result.put(candidate, blockIndex);
                }
            }
            survivorCount = nextSurvivorCount;
            if (survivorCount == 0) {
                return result == null ? Collections.emptyMap() : result;
            }
        }

        if (result == null) {
            result = new HashMap<>(survivorCount);
        }
        for (int index = 0; index < survivorCount; index++) {
            result.put(candidates[index], blockCacheKeys.size());
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

        lock.lock();
        try {
            blockToEnginesMap.clear();
            totalMappings.reset();
        } finally {
            lock.unlock();
        }
        log.info("Cleared global cache index");
    }

    public long totalBlocks() {
        return blockToEnginesMap.mappingCount();
    }

    public long totalMappings() {
        return totalMappings.sum();
    }
}
