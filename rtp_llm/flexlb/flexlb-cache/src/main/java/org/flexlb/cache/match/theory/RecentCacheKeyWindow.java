package org.flexlb.cache.match.theory;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;

import java.util.List;
import java.util.Objects;
import java.util.function.LongSupplier;

/**
 * Fixed-size recent cache-key pool for request-level theory cache hit metrics.
 * Each cache key occupies one pool entry and remains available until its most recent occurrence expires.
 * Request hits are counted against the pool before the request's keys are retained.
 */
@Slf4j
public class RecentCacheKeyWindow {

    private static final int MIN_HASH_TABLE_SIZE = 16;
    private static final double HASH_LOAD_FACTOR = 0.67D;
    private static final byte EMPTY = 0;
    private static final byte USED = 1;

    private final long timeWindowMs;
    private final int maxCacheKeys;
    private final LongSupplier nowSupplier;
    private final long[] tableKeys;
    private final long[] tableLastSeenTimestampMs;
    private final int[] tableHeapIndexes;
    private final byte[] tableStates;
    private final int[] expirationHeapSlots;
    private final int tableMask;

    private int uniqueSize;

    RecentCacheKeyWindow(long timeWindowMs, long maxCacheKeys, LongSupplier nowSupplier) {
        if (timeWindowMs <= 0L) {
            throw new IllegalArgumentException("timeWindowMs must be positive");
        }
        if (maxCacheKeys <= 0L || maxCacheKeys > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("maxCacheKeys must be between 1 and " + Integer.MAX_VALUE);
        }
        this.timeWindowMs = timeWindowMs;
        this.maxCacheKeys = Math.toIntExact(maxCacheKeys);
        this.nowSupplier = Objects.requireNonNull(nowSupplier, "nowSupplier");

        int hashTableCapacity = hashTableCapacityFor(this.maxCacheKeys);
        this.tableMask = hashTableCapacity - 1;
        this.tableKeys = new long[hashTableCapacity];
        this.tableLastSeenTimestampMs = new long[hashTableCapacity];
        this.tableHeapIndexes = new int[hashTableCapacity];
        this.tableStates = new byte[hashTableCapacity];
        this.expirationHeapSlots = new int[this.maxCacheKeys];

        log.info("Recent cache-key pool config: timeWindowMs={}, maxCacheKeys={}, hashTableCapacity={}",
                this.timeWindowMs, this.maxCacheKeys, hashTableCapacity);
    }

    public Snapshot record(List<Long> cacheKeys) {
        long nowMs = nowSupplier.getAsLong();
        evictExpired(nowMs);

        long requestOccurrences = 0L;
        long requestHitOccurrences = 0L;
        if (cacheKeys != null) {
            int size = cacheKeys.size();
            for (int i = 0; i < size; i++) {
                Long cacheKey = cacheKeys.get(i);
                if (cacheKey == null) {
                    continue;
                }
                requestOccurrences++;
                if (findSlot(cacheKey) >= 0) {
                    requestHitOccurrences++;
                }
            }
            for (int i = 0; i < size; i++) {
                Long cacheKey = cacheKeys.get(i);
                if (cacheKey != null) {
                    retainCacheKey(cacheKey, nowMs);
                }
            }
        }

        if (log.isDebugEnabled()) {
            logRequest(nowMs, requestOccurrences, requestHitOccurrences);
        }
        return new Snapshot(timeWindowMs, requestOccurrences, requestHitOccurrences);
    }

    private void retainCacheKey(long cacheKey, long nowMs) {
        int existingSlot = findSlot(cacheKey);
        if (existingSlot >= 0) {
            tableLastSeenTimestampMs[existingSlot] = nowMs;
            siftExpirationHeapDown(tableHeapIndexes[existingSlot]);
            return;
        }
        if (uniqueSize == maxCacheKeys) {
            evictOldestCacheKey();
        }
        int newSlot = findEmptySlot(cacheKey);
        tableKeys[newSlot] = cacheKey;
        tableLastSeenTimestampMs[newSlot] = nowMs;
        tableStates[newSlot] = USED;
        addToExpirationHeap(newSlot);
        uniqueSize++;
    }

    private void evictExpired(long nowMs) {
        long expireBeforeOrAt = nowMs - timeWindowMs;
        while (uniqueSize > 0 && oldestTimestampMs() <= expireBeforeOrAt) {
            evictOldestCacheKey();
        }
    }

    private long oldestTimestampMs() {
        return tableLastSeenTimestampMs[expirationHeapSlots[0]];
    }

    private void evictOldestCacheKey() {
        int oldestSlot = expirationHeapSlots[0];
        removeFromExpirationHeap(0);
        removeFromHashTable(oldestSlot);
        uniqueSize--;
    }

    private void addToExpirationHeap(int tableSlot) {
        int heapIndex = uniqueSize;
        expirationHeapSlots[heapIndex] = tableSlot;
        tableHeapIndexes[tableSlot] = heapIndex;
        siftExpirationHeapUp(heapIndex);
    }

    private void removeFromExpirationHeap(int heapIndex) {
        int lastHeapIndex = uniqueSize - 1;
        int lastTableSlot = expirationHeapSlots[lastHeapIndex];
        if (heapIndex < lastHeapIndex) {
            expirationHeapSlots[heapIndex] = lastTableSlot;
            tableHeapIndexes[lastTableSlot] = heapIndex;
            siftExpirationHeapDown(heapIndex);
        }
    }

    private void siftExpirationHeapUp(int heapIndex) {
        int currentIndex = heapIndex;
        while (currentIndex > 0) {
            int parentIndex = (currentIndex - 1) >>> 1;
            if (timestampAtHeapIndex(parentIndex) <= timestampAtHeapIndex(currentIndex)) {
                return;
            }
            swapHeapEntries(parentIndex, currentIndex);
            currentIndex = parentIndex;
        }
    }

    private void siftExpirationHeapDown(int heapIndex) {
        int currentIndex = heapIndex;
        while (true) {
            int leftChildIndex = currentIndex * 2 + 1;
            if (leftChildIndex >= uniqueSize) {
                return;
            }
            int smallestChildIndex = leftChildIndex;
            int rightChildIndex = leftChildIndex + 1;
            if (rightChildIndex < uniqueSize
                    && timestampAtHeapIndex(rightChildIndex) < timestampAtHeapIndex(leftChildIndex)) {
                smallestChildIndex = rightChildIndex;
            }
            if (timestampAtHeapIndex(currentIndex) <= timestampAtHeapIndex(smallestChildIndex)) {
                return;
            }
            swapHeapEntries(currentIndex, smallestChildIndex);
            currentIndex = smallestChildIndex;
        }
    }

    private long timestampAtHeapIndex(int heapIndex) {
        return tableLastSeenTimestampMs[expirationHeapSlots[heapIndex]];
    }

    private void swapHeapEntries(int firstHeapIndex, int secondHeapIndex) {
        int firstTableSlot = expirationHeapSlots[firstHeapIndex];
        int secondTableSlot = expirationHeapSlots[secondHeapIndex];
        expirationHeapSlots[firstHeapIndex] = secondTableSlot;
        expirationHeapSlots[secondHeapIndex] = firstTableSlot;
        tableHeapIndexes[firstTableSlot] = secondHeapIndex;
        tableHeapIndexes[secondTableSlot] = firstHeapIndex;
    }

    private void removeFromHashTable(int tableSlotToRemove) {
        int vacantSlot = tableSlotToRemove;
        int nextSlot = (vacantSlot + 1) & tableMask;
        while (tableStates[nextSlot] == USED) {
            int idealSlot = hashIndex(tableKeys[nextSlot], tableMask);
            if (((nextSlot - idealSlot) & tableMask) > ((vacantSlot - idealSlot) & tableMask)) {
                tableKeys[vacantSlot] = tableKeys[nextSlot];
                tableLastSeenTimestampMs[vacantSlot] = tableLastSeenTimestampMs[nextSlot];
                tableStates[vacantSlot] = USED;
                int heapIndex = tableHeapIndexes[nextSlot];
                tableHeapIndexes[vacantSlot] = heapIndex;
                expirationHeapSlots[heapIndex] = vacantSlot;
                vacantSlot = nextSlot;
            }
            nextSlot = (nextSlot + 1) & tableMask;
        }
        tableStates[vacantSlot] = EMPTY;
        tableKeys[vacantSlot] = 0L;
        tableLastSeenTimestampMs[vacantSlot] = 0L;
        tableHeapIndexes[vacantSlot] = 0;
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

    private int findEmptySlot(long cacheKey) {
        int index = hashIndex(cacheKey, tableMask);
        while (tableStates[index] == USED) {
            index = (index + 1) & tableMask;
        }
        return index;
    }

    private void logRequest(long nowMs, long requestOccurrences, long requestHitOccurrences) {
        double hitRatio = requestOccurrences > 0L ? requestHitOccurrences / (double) requestOccurrences : 0.0D;
        log.debug("Recent cache-key request: nowMs={}, requestCacheKeys={}, hitCacheKeys={}, hitRatio={}, "
                        + "poolUniqueCacheKeys={}, maxCacheKeys={}, timeWindowMs={}",
                nowMs, requestOccurrences, requestHitOccurrences, hitRatio, uniqueSize, maxCacheKeys, timeWindowMs);
    }

    private static int hashTableCapacityFor(int maxCacheKeys) {
        long needed = Math.max(MIN_HASH_TABLE_SIZE, (long) Math.ceil(maxCacheKeys / HASH_LOAD_FACTOR));
        int capacity = MIN_HASH_TABLE_SIZE;
        while (capacity < needed && capacity < (1 << 30)) {
            capacity <<= 1;
        }
        return capacity;
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
