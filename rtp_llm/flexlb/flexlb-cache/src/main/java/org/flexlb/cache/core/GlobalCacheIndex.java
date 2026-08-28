package org.flexlb.cache.core;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.cache.domain.CacheMatch;
import org.flexlb.cache.domain.DiffResult;
import org.flexlb.cache.domain.EngineGeneration;
import org.springframework.stereotype.Component;

import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.locks.ReentrantLock;

/**
 * The single source of truth for generation-fenced KV-cache ownership.
 *
 * <p>Both directions of the index are guarded by one lock:
 * address-to-(generation, immutable blocks) and block-to-addresses. A full
 * snapshot is diffed and committed while holding that lock, so readers cannot
 * observe one direction before the other. Delayed callbacks can only address
 * their exact generation token.</p>
 */
@Slf4j
@Component
public class GlobalCacheIndex {

    private record EngineOwnership(long generationId, Set<Long> cacheBlocks) {
        private EngineOwnership {
            cacheBlocks = Set.copyOf(cacheBlocks);
        }
    }

    /** Guarded by {@link #lock}; values never escape this class. */
    private final Map<Long, Set<String>> blockToEnginesMap = new HashMap<>();

    /** Guarded by {@link #lock}; contained block sets are immutable. */
    private final Map<String, EngineOwnership> engineOwnerships = new HashMap<>();

    private final ReentrantLock lock = new ReentrantLock();

    /** Guarded by {@link #lock}. */
    private long totalMappings;

    /**
     * Activate a generation and atomically withdraw all blocks belonging to
     * its predecessor. Repeating the active token is idempotent; an older token
     * is rejected.
     */
    public boolean activateEngineGeneration(
            String engineIpPort, long generationId) {
        requireIdentity(engineIpPort, generationId);
        lock.lock();
        try {
            EngineOwnership current = engineOwnerships.get(engineIpPort);
            if (current != null && current.generationId() > generationId) {
                return false;
            }
            if (current != null && current.generationId() == generationId) {
                return true;
            }
            if (current != null) {
                removeMappings(engineIpPort, current.cacheBlocks());
            }
            engineOwnerships.put(
                    engineIpPort, new EngineOwnership(generationId, Set.of()));
            return true;
        } finally {
            lock.unlock();
        }
    }

    /**
     * Atomically replace the full cache snapshot of one exact generation.
     *
     * @return the committed immutable diff, or empty when the token is stale
     */
    public Optional<DiffResult> replaceEngineCache(
            String engineIpPort,
            long generationId,
            Set<Long> newCacheBlocks) {
        requireIdentity(engineIpPort, generationId);
        if (newCacheBlocks == null) {
            throw new IllegalArgumentException("newCacheBlocks must not be null");
        }
        Set<Long> immutableSnapshot = Set.copyOf(newCacheBlocks);

        lock.lock();
        try {
            EngineOwnership current = engineOwnerships.get(engineIpPort);
            if (current == null || current.generationId() != generationId) {
                return Optional.empty();
            }

            Set<Long> addedBlocks = new HashSet<>(immutableSnapshot);
            addedBlocks.removeAll(current.cacheBlocks());
            Set<Long> removedBlocks = new HashSet<>(current.cacheBlocks());
            removedBlocks.removeAll(immutableSnapshot);

            removeMappings(engineIpPort, removedBlocks);
            addMappings(engineIpPort, addedBlocks);
            engineOwnerships.put(engineIpPort,
                    new EngineOwnership(generationId, immutableSnapshot));

            return Optional.of(new DiffResult(
                    engineIpPort, addedBlocks, removedBlocks));
        } finally {
            lock.unlock();
        }
    }

    /** Retire only the exact active generation. */
    public boolean retireEngineGeneration(
            String engineIpPort, long generationId) {
        requireIdentity(engineIpPort, generationId);
        lock.lock();
        try {
            EngineOwnership current = engineOwnerships.get(engineIpPort);
            if (current == null || current.generationId() != generationId) {
                return false;
            }
            removeMappings(engineIpPort, current.cacheBlocks());
            engineOwnerships.remove(engineIpPort);
            return true;
        } finally {
            lock.unlock();
        }
    }

    /** Calculate prefix matches for exact candidates from one locked snapshot. */
    public Map<EngineGeneration, CacheMatch> batchCalculatePrefixMatches(
            List<EngineGeneration> engineGenerations,
            List<Long> blockCacheKeys) {
        if (isEmpty(engineGenerations) || isEmpty(blockCacheKeys)) {
            return Collections.emptyMap();
        }

        lock.lock();
        try {
            Map<EngineGeneration, CacheMatch> result =
                    new HashMap<>(engineGenerations.size());
            for (EngineGeneration candidate : engineGenerations) {
                if (candidate == null) {
                    throw new IllegalArgumentException(
                            "Engine generation candidate must not be null");
                }
                EngineOwnership ownership = engineOwnerships.get(
                        candidate.address());
                if (ownership == null
                        || ownership.generationId() != candidate.generationId()) {
                    continue;
                }

                int prefixMatchLength = 0;
                while (prefixMatchLength < blockCacheKeys.size()) {
                    Set<String> blockOwners = blockToEnginesMap.get(
                            blockCacheKeys.get(prefixMatchLength));
                    if (blockOwners == null
                            || !blockOwners.contains(candidate.address())) {
                        break;
                    }
                    prefixMatchLength++;
                }
                result.put(candidate, new CacheMatch(prefixMatchLength));
            }
            return Map.copyOf(result);
        } finally {
            lock.unlock();
        }
    }

    public void clear() {
        lock.lock();
        try {
            blockToEnginesMap.clear();
            engineOwnerships.clear();
            totalMappings = 0L;
        } finally {
            lock.unlock();
        }
        log.info("Cleared cache index");
    }

    /** Return all metric values from the same locked snapshot. */
    public IndexMetrics metricsSnapshot() {
        lock.lock();
        try {
            return new IndexMetrics(
                    blockToEnginesMap.size(),
                    totalMappings,
                    engineOwnerships.size());
        } finally {
            lock.unlock();
        }
    }

    public record IndexMetrics(
            long totalBlocks,
            long totalMappings,
            int engineCount) {
    }

    private void addMappings(String engineIpPort, Set<Long> cacheBlocks) {
        for (Long blockCacheKey : cacheBlocks) {
            Set<String> engines = blockToEnginesMap.computeIfAbsent(
                    blockCacheKey, ignored -> new HashSet<>());
            if (engines.add(engineIpPort)) {
                totalMappings++;
            }
        }
    }

    private void removeMappings(String engineIpPort, Set<Long> cacheBlocks) {
        for (Long blockCacheKey : cacheBlocks) {
            Set<String> engines = blockToEnginesMap.get(blockCacheKey);
            if (engines == null || !engines.remove(engineIpPort)) {
                continue;
            }
            totalMappings--;
            if (engines.isEmpty()) {
                blockToEnginesMap.remove(blockCacheKey);
            }
        }
    }

    private static boolean isEmpty(List<?> list) {
        return list == null || list.isEmpty();
    }

    private static void requireIdentity(
            String engineIpPort, long generationId) {
        if (engineIpPort == null || engineIpPort.isBlank()) {
            throw new IllegalArgumentException("engineIpPort must not be blank");
        }
        if (generationId <= 0L) {
            throw new IllegalArgumentException(
                    "generationId must be positive: " + generationId);
        }
    }
}
