package org.flexlb.cache.match.localsync;

import com.google.common.collect.ImmutableSet;
import it.unimi.dsi.fastutil.HashCommon;
import it.unimi.dsi.fastutil.longs.Long2ObjectOpenHashMap;
import it.unimi.dsi.fastutil.longs.LongIterators;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.LongAdder;
import java.util.stream.IntStream;

/**
 * Tracks logical worker owners of each cache block and matches their cached prefixes.
 * Blocks are stored in shards so updates and retirements can proceed on different shards.
 */
@Slf4j
@Component
public class GlobalCacheIndex {
    /** Per-caller compaction storage; cache queries never retain this array. */
    private final ThreadLocal<String[]> prefixCandidates =
            ThreadLocal.withInitial(() -> new String[0]);

    private volatile BlockIndex index = new BlockIndex();

    /**
     * Add cache block to specified engine
     *
     * @param blockCacheKey Cache block hash value
     * @param engineIpPort  logical worker identity in {@code ip:port@engineIndex} format; the
     *                      index identifies one independently routable engine behind the physical
     *                      frontend
     */
    public void addCacheBlock(long blockCacheKey, String engineIpPort) {
        if (engineIpPort == null) {
            log.warn(
                    "Invalid parameters: blockCacheKey={}, engineIpPort={}",
                    blockCacheKey,
                    engineIpPort);
            return;
        }

        index.add(blockCacheKey, engineIpPort);
    }

    /**
     * Remove cache block from specified engine
     *
     * @param engineIp      logical worker identity in {@code ip:port@engineIndex} format
     * @param blockCacheKey Cache block hash value
     */
    public void removeCacheBlock(String engineIp, long blockCacheKey) {
        if (engineIp == null) {
            return;
        }

        index.remove(blockCacheKey, engineIp);
    }

    /**
     * Remove only the blocks from the retiring engine's detached local view.
     *
     * @param engineIp logical worker identity in {@code ip:port@engineIndex} format
     * @param blockCacheKeys keys detached by the caller while holding the worker cache lock
     */
    public void removeAllCacheBlockOfEngine(String engineIp, Set<Long> blockCacheKeys) {
        if (engineIp == null || blockCacheKeys.isEmpty()) {
            return;
        }

        BlockIndex currentIndex = index;
        for (var keys = LongIterators.asLongIterator(blockCacheKeys.iterator()); keys.hasNext(); ) {
            currentIndex.remove(keys.nextLong(), engineIp);
        }
    }

    /**
     * Calculate engine prefix match length based on prefix matching
     *
     * @param engineIpPorts  logical worker identities in {@code ip:port@engineIndex} format
     * @param blockCacheKeys Ordered cache block hash value list
     * @return prefix match lengths keyed by logical worker identity
     */
    public Map<String, Integer> batchCalculatePrefixMatchLength(
            List<String> engineIpPorts, List<Long> blockCacheKeys) {

        if (isEmpty(engineIpPorts) || isEmpty(blockCacheKeys)) {
            return Collections.emptyMap();
        }
        return calculatePrefixMatchLength(engineIpPorts, blockCacheKeys);
    }

    /**
     * Clear all data
     */
    public void clear() {
        index = new BlockIndex();
        log.info("Cleared global cache index");
    }

    public long totalBlocks() {
        return index.totalBlocks.sum();
    }

    public long totalMappings() {
        return index.totalMappings.sum();
    }

    /**
     * Prefix match calculation
     *
     * @param engineIpPorts  logical worker identities in {@code ip:port@engineIndex} format
     * @param blockCacheKeys Ordered cache block hash value list
     * @return prefix match lengths keyed by logical worker identity
     */
    private Map<String, Integer> calculatePrefixMatchLength(
            List<String> engineIpPorts, List<Long> blockCacheKeys) {

        BlockIndex currentIndex = index;
        Set<String> firstBlockOwners = currentIndex.ownersFor(blockCacheKeys.getFirst());
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
        for (int candidateIndex = 0; candidateIndex < engineIpPorts.size(); candidateIndex++) {
            String candidate = candidates[candidateIndex];
            if (firstBlockOwners.contains(candidate)) {
                candidates[survivorCount++] = candidate;
            }
        }
        if (survivorCount == 0) {
            return Collections.emptyMap();
        }

        Map<String, Integer> result = null;
        for (int blockIndex = 1; blockIndex < blockCacheKeys.size(); blockIndex++) {
            Set<String> blockOwners = currentIndex.ownersFor(blockCacheKeys.get(blockIndex));
            int nextSurvivorCount = 0;
            for (int candidateIndex = 0; candidateIndex < survivorCount; candidateIndex++) {
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
     * Holds shard tables and counts that are replaced together when the index is cleared.
     * Owner sets are immutable so prefix matching can inspect them after releasing the shard lock.
     */
    private static final class BlockIndex {
        private static final int SHARD_BITS = 8;

        private final List<Long2ObjectOpenHashMap<ImmutableSet<String>>> shards =
                IntStream.range(0, 1 << SHARD_BITS)
                        .mapToObj(index -> new Long2ObjectOpenHashMap<ImmutableSet<String>>())
                        .toList();
        private final LongAdder totalBlocks = new LongAdder();
        private final LongAdder totalMappings = new LongAdder();

        private void add(long key, String worker) {
            var shard = shardFor(key);
            synchronized (shard) {
                ImmutableSet<String> owners = shard.get(key);
                if (owners == null) {
                    shard.put(key, ImmutableSet.of(worker));
                    totalBlocks.increment();
                    totalMappings.increment();
                } else if (!owners.contains(worker)) {
                    shard.put(
                            key,
                            ImmutableSet.<String>builderWithExpectedSize(owners.size() + 1)
                                    .addAll(owners)
                                    .add(worker)
                                    .build());
                    totalMappings.increment();
                }
            }
        }

        private Long2ObjectOpenHashMap<ImmutableSet<String>> shardFor(long key) {
            // fastutil uses the low hash bits for the entry slot within a shard.
            return shards.get((int) (HashCommon.mix(key) >>> (Long.SIZE - SHARD_BITS)));
        }

        private void remove(long key, String worker) {
            var shard = shardFor(key);
            synchronized (shard) {
                ImmutableSet<String> owners = shard.get(key);
                if (owners == null || !owners.contains(worker)) {
                    return;
                }
                if (owners.size() == 1) {
                    shard.remove(key);
                    totalBlocks.decrement();
                } else {
                    var remaining = ImmutableSet.<String>builderWithExpectedSize(owners.size() - 1);
                    for (String owner : owners) {
                        if (!worker.equals(owner)) {
                            remaining.add(owner);
                        }
                    }
                    shard.put(key, remaining.build());
                }
                totalMappings.decrement();
            }
        }

        private Set<String> ownersFor(Long key) {
            if (key == null) {
                return Collections.emptySet();
            }
            var shard = shardFor(key);
            synchronized (shard) {
                ImmutableSet<String> owners = shard.get(key.longValue());
                return owners == null ? Collections.emptySet() : owners;
            }
        }
    }
}
