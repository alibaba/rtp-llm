package org.flexlb.cache.core;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.dao.master.EndpointTrace;
import org.springframework.stereotype.Component;

import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.flexlb.cache.primitive.LongObjectMap;
import org.flexlb.cache.primitive.SmallOwnerSet;
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
    private final LongObjectMap<Set<String>> blockToEnginesMap = new LongObjectMap<>();

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
    public void addCacheBlock(Long blockCacheKey, String engineIpPort, EndpointTrace.Poll pollTrace) {
        if (blockCacheKey == null || engineIpPort == null) {
            log.warn("Invalid parameters: blockCacheKey={}, engineIpPort={}", blockCacheKey, engineIpPort);
            return;
        }

        // The former global-lock trace is intentionally not emitted for shard monitors.
        blockToEnginesMap.compute(blockCacheKey, owners -> {
            if (owners == null) owners = new SmallOwnerSet();
            if (owners.add(engineIpPort)) totalMappings.increment();
            return owners;
        });
    }

    /**
     * Remove cache block from specified engine
     *
     * @param engineIp      Engine IP
     * @param blockCacheKey Cache block hash value
     */
    public void removeCacheBlock(String engineIp, Long blockCacheKey, EndpointTrace.Poll pollTrace) {
        if (blockCacheKey == null || engineIp == null) {
            return;
        }

        removeOwner(engineIp, blockCacheKey, null);
    }

    private void removeOwner(String engineIp, long key, long[] removed) {
        blockToEnginesMap.compute(key, owners -> {
            if (owners != null && owners.remove(engineIp)) {
                totalMappings.decrement();
                if (removed != null) removed[0]++;
            }
            return owners == null || owners.isEmpty() ? null : owners;
        });
    }

    /**
     * Remove an engine
     *
     * <p>The caller must keep the owning generation's cacheMutationLock held
     * from obtaining blockCacheKeys until this method returns. The shard locks
     * protect individual mappings; they do not freeze the borrowed local key set.
     *
     * @param engineIp Engine IP
     */
    public void removeAllCacheBlockOfEngine(String engineIp, Set<Long> blockCacheKeys, EndpointTrace trace) {
        if (engineIp == null || blockCacheKeys == null || blockCacheKeys.isEmpty()) {
            return;
        }

        long scannedKeys = 0L;
        long[] removedMappings = {0L};
        try {
            for (Long key : blockCacheKeys) {
                scannedKeys++;
                removeOwner(engineIp, key, removedMappings);
            }
        } finally {
            if (trace != null) trace.recordCleanupCounts(scannedKeys, removedMappings[0]);
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
        return blockToEnginesMap.read(blockCacheKey,
                engines -> engines == null ? Collections.emptySet() : Set.copyOf(engines));
    }

    /**
     * Clear all data
     */
    public void clear() {

        blockToEnginesMap.clear(totalMappings::reset);
        log.info("Cleared global cache index");
    }

    public long totalBlocks() {
        return blockToEnginesMap.mappingCount();
    }

    public long totalMappings() {
        return totalMappings.sum();
    }
}
