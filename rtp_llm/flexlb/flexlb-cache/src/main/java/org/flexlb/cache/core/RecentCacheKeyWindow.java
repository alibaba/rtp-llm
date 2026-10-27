package org.flexlb.cache.core;

import com.google.common.math.IntMath;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.flexlb.config.ConfigService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.function.LongSupplier;

/**
 * Capacity-bounded request history with an index of each key's latest retained occurrence.
 */
@Slf4j
@Component
public class RecentCacheKeyWindow {

    public static final long DEFAULT_TIME_WINDOW_MS = 30L * 60L * 1000L;
    public static final long DEFAULT_MAX_CACHE_KEYS = 10_000_000L;
    private static final int MAX_ARRAY_SIZE = Integer.MAX_VALUE - 8;
    private static final int MIN_HASH_TABLE_SIZE = 16;
    private static final double HASH_LOAD_FACTOR = 0.67D;
    private static final long[] EMPTY = new long[0];
    private static final int CLEANUP_SLOTS = 64;

    private final long timeWindowMs;
    private final int maxCacheKeys;
    private final LongSupplier nowSupplier;
    private long[] entryTimestampMs = EMPTY;
    private long[] entryEndSequence = EMPTY;
    private long[] tableKeys = EMPTY;
    private long[] tableSequences = EMPTY;
    private int entryHead;
    private int entrySize;
    private int tableSize;
    private int cleanupCursor;
    private int remainingCleanupSlots;
    private long sequence;
    private long expiredThrough;
    private long lastRecordMs = Long.MIN_VALUE;

    @Autowired
    public RecentCacheKeyWindow(ConfigService configService) {
        this(resolveTimeWindowMs(configService), resolveMaxCacheKeys(configService), System::currentTimeMillis);
    }

    RecentCacheKeyWindow(long timeWindowMs, long maxCacheKeys, LongSupplier nowSupplier) {
        this.timeWindowMs = normalizeTimeWindowMs(timeWindowMs);
        this.maxCacheKeys = normalizeCapacity(maxCacheKeys);
        this.nowSupplier = nowSupplier;
    }

    public Snapshot record(List<Long> cacheKeys) {
        synchronized (this) {
            return record(cacheKeys, nowSupplier.getAsLong());
        }
    }

    synchronized Snapshot record(List<Long> cacheKeys, long nowMs) {
        nowMs = Math.max(nowMs, lastRecordMs);
        lastRecordMs = nowMs;
        expire(nowMs);
        long occurrences = 0L, hits = 0L;
        if (cacheKeys != null) {
            for (Long key : cacheKeys) {
                if (key == null) { continue; }
                occurrences++;
                if (contains(key)) { hits++; }
            }
        }
        if (occurrences > 0 && occurrences <= maxCacheKeys) {
            long end = Math.addExact(sequence, occurrences);
            long required = end - maxCacheKeys;
            if (required > expiredThrough) {
                int count = 1;
                if (entryEndSequence[entryHead] < required) {
                    int low = 1, high = entrySize;
                    while (low < high) {
                        int middle = (low + high) >>> 1;
                        if (entryEndSequence[entryIndex(middle)] < required) { low = middle + 1; } else { high = middle; }
                    }
                    count = low + 1;
                }
                discardEntries(count);
            }
            growEntries();
            int tail = entryIndex(entrySize++);
            entryTimestampMs[tail] = nowMs;
            entryEndSequence[tail] = end;
            sequence = end;
            for (Long key : cacheKeys) {
                if (key != null) { put(key, end); }
            }
        }
        cleanStaleSlots();
        return new Snapshot(timeWindowMs, occurrences, hits);
    }

    private int entryIndex(int offset) {
        return (int) (((long) entryHead + offset) % entryEndSequence.length);
    }

    private void expire(long nowMs) {
        if (entrySize == 0) { return; }
        long cutoff = com.google.common.math.LongMath.saturatedSubtract(nowMs, timeWindowMs);
        if (entryTimestampMs[entryHead] > cutoff) { return; }
        int low = 0, high = entrySize;
        while (low < high) {
            int middle = (low + high) >>> 1;
            if (entryTimestampMs[entryIndex(middle)] <= cutoff) { low = middle + 1; } else { high = middle; }
        }
        discardEntries(low);
    }

    private void discardEntries(int count) {
        if (count == 0) { return; }
        expiredThrough = entryEndSequence[entryIndex(count - 1)];
        entryHead = entryIndex(count);
        entrySize -= count;
        remainingCleanupSlots = tableKeys.length;
        if (entrySize == 0) {
            entryHead = 0;
            entryTimestampMs = EMPTY;
            entryEndSequence = EMPTY;
            tableKeys = EMPTY;
            tableSequences = EMPTY;
            tableSize = 0;
            cleanupCursor = 0;
            remainingCleanupSlots = 0;
        }
    }

    private void growEntries() {
        if (entrySize < entryEndSequence.length) { return; }
        int capacity = (int) Math.min(maxCacheKeys, Math.max(16L, 2L * entrySize));
        long[] timestamps = new long[capacity], ends = new long[capacity];
        for (int i = 0; i < entrySize; i++) {
            timestamps[i] = entryTimestampMs[entryIndex(i)];
            ends[i] = entryEndSequence[entryIndex(i)];
        }
        entryTimestampMs = timestamps;
        entryEndSequence = ends;
        entryHead = 0;
    }

    private boolean contains(long key) {
        if (tableKeys.length == 0) { return false; }
        int mask = tableKeys.length - 1;
        int slot = hashIndex(key, mask);
        while (tableSequences[slot] != 0L) {
            if (tableKeys[slot] == key) { return tableSequences[slot] > expiredThrough; }
            slot = (slot + 1) & mask;
        }
        return false;
    }

    private void put(long key, long end) {
        if (tableKeys.length == 0) { resizeTable(MIN_HASH_TABLE_SIZE); }
        int mask = tableKeys.length - 1;
        int slot = hashIndex(key, mask);
        while (tableSequences[slot] != 0L) {
            if (tableKeys[slot] == key) { tableSequences[slot] = end; return; }
            slot = (slot + 1) & mask;
        }
        if (tableSize + 1 > tableKeys.length * HASH_LOAD_FACTOR) {
            resizeTable(Math.min(hashTableCapacityFor(maxCacheKeys), Math.multiplyExact(tableKeys.length, 2)));
            mask = tableKeys.length - 1;
            slot = hashIndex(key, mask);
            while (tableSequences[slot] != 0L) { slot = (slot + 1) & mask; }
        }
        tableKeys[slot] = key;
        tableSequences[slot] = end;
        tableSize++;
    }

    private void resizeTable(int capacity) {
        long[] keys = tableKeys, sequences = tableSequences;
        tableKeys = new long[capacity];
        tableSequences = new long[capacity];
        tableSize = 0;
        cleanupCursor = 0;
        remainingCleanupSlots = 0;
        int mask = capacity - 1;
        for (int i = 0; i < keys.length; i++) {
            if (sequences[i] <= expiredThrough) { continue; }
            int slot = hashIndex(keys[i], mask);
            while (tableSequences[slot] != 0L) { slot = (slot + 1) & mask; }
            tableKeys[slot] = keys[i];
            tableSequences[slot] = sequences[i];
            tableSize++;
        }
    }

    private void cleanStaleSlots() {
        if (remainingCleanupSlots == 0) { return; }
        int mask = tableKeys.length - 1;
        int count = Math.min(tableSize < tableKeys.length / 2 ? 1 : CLEANUP_SLOTS, remainingCleanupSlots);
        remainingCleanupSlots -= count;
        for (int i = 0; i < count; i++) {
            int slot = cleanupCursor;
            if (tableSequences[slot] != 0L && tableSequences[slot] <= expiredThrough) { removeSlot(slot); }
            cleanupCursor = (cleanupCursor + 1) & mask;
        }
    }

    private void removeSlot(int slot) {
        int mask = tableKeys.length - 1;
        int next = (slot + 1) & mask;
        while (tableSequences[next] != 0L) {
            int ideal = hashIndex(tableKeys[next], mask);
            if (((next - ideal) & mask) > ((slot - ideal) & mask)) {
                tableKeys[slot] = tableKeys[next];
                tableSequences[slot] = tableSequences[next];
                slot = next;
            }
            next = (next + 1) & mask;
        }
        tableSequences[slot] = 0L;
        tableSize--;
    }

    static long resolveTimeWindowMs(ConfigService configService) {
        if (configService == null || configService.loadBalanceConfig() == null) {
            return DEFAULT_TIME_WINDOW_MS;
        }
        return configService.loadBalanceConfig().getObservability().getCacheHit()
                .getRecentKeyWindow().getDurationMs();
    }

    static long resolveMaxCacheKeys(ConfigService configService) {
        if (configService == null || configService.loadBalanceConfig() == null) {
            return DEFAULT_MAX_CACHE_KEYS;
        }
        return configService.loadBalanceConfig().getObservability().getCacheHit()
                .getRecentKeyWindow().getMaxKeyOccurrences();
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
            log.warn("cacheHitMaxCacheKeys is too large for arrays: {}, cap to {}",
                    candidate,
                    MAX_ARRAY_SIZE);
            return MAX_ARRAY_SIZE;
        }
        return (int) candidate;
    }

    private static int hashTableCapacityFor(int maxCacheKeys) {
        long needed = Math.max(MIN_HASH_TABLE_SIZE, (long) Math.ceil(maxCacheKeys / HASH_LOAD_FACTOR));
        return IntMath.ceilingPowerOfTwo(Math.clamp(needed, MIN_HASH_TABLE_SIZE, 1 << 30));
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
