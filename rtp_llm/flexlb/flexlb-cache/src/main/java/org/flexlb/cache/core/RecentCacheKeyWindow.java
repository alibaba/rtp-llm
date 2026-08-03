package org.flexlb.cache.core;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.flexlb.config.ConfigService;

import java.util.List;
import java.util.function.LongSupplier;

/**
 * Fixed-size recent cache-key pool for request-level cache hit metrics.
 */
@Slf4j
public class RecentCacheKeyWindow {

    public static final long DEFAULT_TIME_WINDOW_MS = 30L * 60L * 1000L;
    public static final long DEFAULT_MAX_CACHE_KEYS = 10_000_000L;
    private static final int MAX_ARRAY_SIZE = Integer.MAX_VALUE - 8;
    private static final int MIN_HASH_TABLE_SIZE = 16;
    private static final double HASH_LOAD_FACTOR = 0.67D;
    private static final byte EMPTY = 0;
    private static final byte USED = 1;

    private final long timeWindowMs;
    private final int maxCacheKeys;
    private final LongSupplier nowSupplier;

    private final long[] cacheKeyRing;
    private final long[] entryTimestampMs;
    private final int[] entryStart;
    private final int[] entryLength;
    private final long[] tableKeys;
    private final int[] tableCounts;
    private final byte[] tableStates;
    private final int tableMask;

    private int entryHead;
    private int entrySize;
    private int keyTail;
    private int keySize;
    private int uniqueSize;

    RecentCacheKeyWindow(long timeWindowMs, long maxCacheKeys, LongSupplier nowSupplier) {
        this.timeWindowMs = normalizeTimeWindowMs(timeWindowMs);
        this.maxCacheKeys = normalizeCapacity(maxCacheKeys);
        this.nowSupplier = nowSupplier;

        int hashTableCapacity = hashTableCapacityFor(this.maxCacheKeys);
        this.tableMask = hashTableCapacity - 1;
        this.cacheKeyRing = new long[this.maxCacheKeys];
        this.entryTimestampMs = new long[this.maxCacheKeys];
        this.entryStart = new int[this.maxCacheKeys];
        this.entryLength = new int[this.maxCacheKeys];
        this.tableKeys = new long[hashTableCapacity];
        this.tableCounts = new int[hashTableCapacity];
        this.tableStates = new byte[hashTableCapacity];

        log.info("Recent cache-key pool config: timeWindowMs={}, maxCacheKeys={}, hashTableCapacity={}",
                this.timeWindowMs,
                this.maxCacheKeys,
                hashTableCapacity);
    }

    public Snapshot record(List<Long> cacheKeys) {
        return record(cacheKeys, nowSupplier.getAsLong());
    }

    Snapshot record(List<Long> cacheKeys, long nowMs) {
        long requestOccurrences;
        long requestHitOccurrences;
        synchronized (this) {
            evictExpired(nowMs);
            requestOccurrences = countNonNull(cacheKeys);
            requestHitOccurrences = countHits(cacheKeys);

            if (requestOccurrences > 0L) {
                retainRequest(cacheKeys, requestOccurrences, nowMs);
            }

        }

        if (log.isInfoEnabled()) {
            logRequest(nowMs, requestOccurrences, requestHitOccurrences);
        }
        return new Snapshot(timeWindowMs, requestOccurrences, requestHitOccurrences);
    }

    private long countNonNull(List<Long> cacheKeys) {
        if (cacheKeys == null || cacheKeys.isEmpty()) {
            return 0L;
        }
        long count = 0L;
        int size = cacheKeys.size();
        for (int i = 0; i < size; i++) {
            Long cacheKey = cacheKeys.get(i);
            if (cacheKey != null) {
                count++;
            }
        }
        return count;
    }

    private long countHits(List<Long> cacheKeys) {
        if (cacheKeys == null || cacheKeys.isEmpty()) {
            return 0L;
        }
        long hits = 0L;
        int size = cacheKeys.size();
        for (int i = 0; i < size; i++) {
            Long cacheKey = cacheKeys.get(i);
            if (cacheKey != null && getCount(cacheKey) > 0) {
                hits++;
            }
        }
        return hits;
    }

    private void retainRequest(List<Long> cacheKeys, long requestOccurrences, long nowMs) {
        if (requestOccurrences > maxCacheKeys) {
            log.warn("Recent cache-key request exceeds pool capacity; skip retaining request: "
                            + "requestCacheKeys={}, maxCacheKeys={}",
                    requestOccurrences,
                    maxCacheKeys);
            return;
        }

        while (keySize + requestOccurrences > maxCacheKeys && evictOldestEntry()) {
            // Make enough room for the current request.
        }
        while (entrySize >= maxCacheKeys && evictOldestEntry()) {
            // Keep one entry slot for the current request.
        }

        int start = keyTail;
        int retained = 0;
        int size = cacheKeys.size();
        for (int i = 0; i < size; i++) {
            Long boxedKey = cacheKeys.get(i);
            if (boxedKey == null) {
                continue;
            }
            long cacheKey = boxedKey;
            appendKey(cacheKey);
            incrementCount(cacheKey);
            retained++;
        }
        if (retained > 0) {
            addEntry(nowMs, start, retained);
        }
    }

    private void evictExpired(long nowMs) {
        long expireBeforeOrAt = nowMs - timeWindowMs;
        while (entrySize > 0 && entryTimestampMs[entryHead] <= expireBeforeOrAt) {
            evictOldestEntry();
        }
    }

    private void addEntry(long timestampMs, int start, int length) {
        int tail = ringIndex(entryHead + entrySize);
        entryTimestampMs[tail] = timestampMs;
        entryStart[tail] = start;
        entryLength[tail] = length;
        entrySize++;
    }

    private void appendKey(long cacheKey) {
        cacheKeyRing[keyTail] = cacheKey;
        keyTail = ringIndex(keyTail + 1);
        keySize++;
    }

    private boolean evictOldestEntry() {
        if (entrySize == 0) {
            return false;
        }
        int start = entryStart[entryHead];
        int length = entryLength[entryHead];
        for (int i = 0; i < length; i++) {
            decrementCount(cacheKeyRing[ringIndex(start + i)]);
        }
        keySize -= length;
        entryHead = ringIndex(entryHead + 1);
        entrySize--;
        return true;
    }

    private int getCount(long cacheKey) {
        int slot = findSlot(cacheKey);
        return slot >= 0 ? tableCounts[slot] : 0;
    }

    private void incrementCount(long cacheKey) {
        int index = hashIndex(cacheKey, tableMask);
        while (tableStates[index] == USED) {
            if (tableKeys[index] == cacheKey) {
                tableCounts[index]++;
                return;
            }
            index = (index + 1) & tableMask;
        }
        tableStates[index] = USED;
        tableKeys[index] = cacheKey;
        tableCounts[index] = 1;
        uniqueSize++;
    }

    private void decrementCount(long cacheKey) {
        int slot = findSlot(cacheKey);
        if (slot < 0) {
            return;
        }
        int next = tableCounts[slot] - 1;
        if (next > 0) {
            tableCounts[slot] = next;
            return;
        }
        removeSlot(slot);
    }

    private int findSlot(long cacheKey) {
        int index = hashIndex(cacheKey, tableMask);
        while (tableStates[index] == USED) {
            if (tableKeys[index] == cacheKey) {
                return index;
            }
            index = (index + 1) & tableMask;
        }
        return -1;
    }

    private void removeSlot(int slotToRemove) {
        int slot = slotToRemove;
        int next = (slot + 1) & tableMask;
        while (tableStates[next] == USED) {
            int ideal = hashIndex(tableKeys[next], tableMask);
            if (((next - ideal) & tableMask) > ((slot - ideal) & tableMask)) {
                tableKeys[slot] = tableKeys[next];
                tableCounts[slot] = tableCounts[next];
                tableStates[slot] = USED;
                slot = next;
            }
            next = (next + 1) & tableMask;
        }
        tableStates[slot] = EMPTY;
        tableKeys[slot] = 0L;
        tableCounts[slot] = 0;
        uniqueSize--;
    }

    private void logRequest(long nowMs, long requestOccurrences, long requestHitOccurrences) {
        double hitRatio = requestOccurrences > 0L ? requestHitOccurrences / (double) requestOccurrences : 0.0D;
        log.info("Recent cache-key request: nowMs={}, requestCacheKeys={}, hitCacheKeys={}, "
                        + "hitRatio={}, poolCacheKeys={}, poolUniqueCacheKeys={}, poolDuplicateCacheKeys={}, "
                        + "poolEntries={}, maxCacheKeys={}, timeWindowMs={}",
                nowMs,
                requestOccurrences,
                requestHitOccurrences,
                hitRatio,
                keySize,
                uniqueSize,
                keySize - uniqueSize,
                entrySize,
                maxCacheKeys,
                timeWindowMs);
    }

    static long resolveTimeWindowMs(ConfigService configService) {
        if (configService == null || configService.loadBalanceConfig() == null) {
            return DEFAULT_TIME_WINDOW_MS;
        }
        return configService.loadBalanceConfig().getCacheHitTimeWindowMs();
    }

    static long resolveMaxCacheKeys(ConfigService configService) {
        if (configService == null || configService.loadBalanceConfig() == null) {
            return DEFAULT_MAX_CACHE_KEYS;
        }
        return configService.loadBalanceConfig().getCacheHitMaxCacheKeys();
    }

    private static long normalizeTimeWindowMs(long candidateMs) {
        if (candidateMs > 0L) {
            return candidateMs;
        }
        log.warn("Invalid cacheHitTimeWindowMs: {}, fallback to default: {}", candidateMs, DEFAULT_TIME_WINDOW_MS);
        return DEFAULT_TIME_WINDOW_MS;
    }

    private static int normalizeCapacity(long candidate) {
        if (candidate <= 0L) {
            log.warn("Invalid cacheHitMaxCacheKeys: {}, fallback to default: {}", candidate, DEFAULT_MAX_CACHE_KEYS);
            return (int) DEFAULT_MAX_CACHE_KEYS;
        }
        if (candidate > MAX_ARRAY_SIZE) {
            log.warn("cacheHitMaxCacheKeys is too large for preallocated arrays: {}, cap to {}",
                    candidate,
                    MAX_ARRAY_SIZE);
            return MAX_ARRAY_SIZE;
        }
        return (int) candidate;
    }

    private static int hashTableCapacityFor(int maxCacheKeys) {
        long needed = Math.max(MIN_HASH_TABLE_SIZE, (long) Math.ceil(maxCacheKeys / HASH_LOAD_FACTOR));
        int capacity = MIN_HASH_TABLE_SIZE;
        while (capacity < needed && capacity < (1 << 30)) {
            capacity <<= 1;
        }
        return capacity;
    }

    private int ringIndex(int index) {
        int result = index % maxCacheKeys;
        return result >= 0 ? result : result + maxCacheKeys;
    }

    private static int hashIndex(long value, int mask) {
        long mixed = value;
        mixed ^= mixed >>> 33;
        mixed *= 0xff51afd7ed558ccdL;
        mixed ^= mixed >>> 33;
        mixed *= 0xc4ceb9fe1a85ec53L;
        mixed ^= mixed >>> 33;
        return (int) mixed & mask;
    }

    @Getter
    public static class Snapshot {
        private final long timeWindowMs;
        private final long requestOccurrences;
        private final long requestHitOccurrences;

        Snapshot(long timeWindowMs, long requestOccurrences, long requestHitOccurrences) {
            this.timeWindowMs = timeWindowMs;
            this.requestOccurrences = requestOccurrences;
            this.requestHitOccurrences = requestHitOccurrences;
        }
    }
}
