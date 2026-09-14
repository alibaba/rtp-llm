package org.flexlb.mockengine;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.function.DoubleConsumer;
import java.util.function.LongSupplier;

/** Optional metadata-only host cache of complete prefix blocks, independent of GPU admission. */
final class MockMemoryBlockCache {
    private final int capacity;
    private final LongSupplier clock;
    private final LinkedHashMap<Long, Long> entries = new LinkedHashMap<>(16, .75f, true);
    private final DoubleConsumer onEviction;
    private long evictions;

    MockMemoryBlockCache(int capacity, DoubleConsumer onEviction) {
        this(capacity, onEviction, System::nanoTime);
    }

    MockMemoryBlockCache(int capacity, DoubleConsumer onEviction, LongSupplier clock) {
        if (capacity <= 0) throw new IllegalArgumentException("memory cache capacity must be positive");
        this.capacity = capacity;
        this.onEviction = onEviction;
        this.clock = clock;
    }

    // The prefix before start is already present on GPU. Stop at the first missing host block.
    synchronized int match(List<Long> keys, int start) {
        int end = start;
        while (end < keys.size() && entries.get(keys.get(end)) != null) end++;
        return end - start;
    }

    synchronized int peekMatch(List<Long> keys, int start) {
        int end = start;
        while (end < keys.size() && entries.containsKey(keys.get(end))) end++;
        return end - start;
    }

    synchronized void write(List<Long> keys) {
        for (Long key : keys) {
            // Reuse touches recency but does not reset creation time (MemoryDiskBlockCache::putCommitted).
            if (entries.get(key) != null) continue;
            if (entries.size() == capacity) {
                var iterator = entries.entrySet().iterator();
                long created = iterator.next().getValue();
                iterator.remove();
                evictions++;
                onEviction.accept(Math.max(0L, clock.getAsLong() - created) / 1_000_000.0);
            }
            entries.put(key, clock.getAsLong());
        }
    }

    synchronized List<Long> keys() { return List.copyOf(entries.keySet()); }
    synchronized int size() { return entries.size(); }
    int capacity() { return capacity; }
    synchronized long evictions() { return evictions; }
    synchronized void clear() { entries.clear(); }
}
