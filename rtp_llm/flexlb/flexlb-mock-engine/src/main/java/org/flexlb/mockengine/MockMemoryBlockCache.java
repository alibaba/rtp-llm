package org.flexlb.mockengine;

import java.util.*;
import java.util.function.DoubleConsumer;
import java.util.function.LongSupplier;

/** Metadata host pool: committed LRU entries and private, capacity-owning copy reservations. */
final class MockMemoryBlockCache {
    private static final class Entry {
        long born;
        int pins;
        Entry(long born) { this.born = born; }
    }
    private final int capacity;
    private final LongSupplier clock;
    private final LinkedHashMap<Long, Entry> entries = new LinkedHashMap<>(16, .75f, true);
    private final Map<Long, Entry> pending = new HashMap<>();
    private final DoubleConsumer onEviction;
    private long evictions, writeRejected, generation;
    private int pinned;

    MockMemoryBlockCache(int capacity, DoubleConsumer onEviction) {
        this(capacity, onEviction, System::nanoTime);
    }
    MockMemoryBlockCache(int capacity, DoubleConsumer onEviction, LongSupplier clock) {
        if (capacity <= 0) throw new IllegalArgumentException("memory cache capacity must be positive");
        this.capacity = capacity; this.onEviction = onEviction; this.clock = clock;
    }

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

    final class ReadLease implements AutoCloseable {
        private final List<Entry> held;
        private final long epoch = generation;
        private boolean closed;
        ReadLease(List<Entry> held) { this.held = held; }
        int blocks() { return held.size(); }
        public void close() {
            synchronized (MockMemoryBlockCache.this) {
                if (closed) return;
                closed = true;
                for (Entry e : held) {
                    e.pins--;
                    if (epoch == generation && e.pins == 0) pinned--;
                }
            }
        }
    }
    synchronized ReadLease pinRead(List<Long> keys, int start) {
        List<Entry> held = new ArrayList<>();
        for (int i = start; i < keys.size(); i++) {
            Entry e = entries.get(keys.get(i));
            if (e == null) break;
            if (e.pins++ == 0) pinned++;
            held.add(e);
        }
        return new ReadLease(held);
    }

    final class WriteLease implements AutoCloseable {
        private final Map<Long, Entry> reserved;
        private boolean closed;
        WriteLease(Map<Long, Entry> reserved) { this.reserved = reserved; }
        int blocks() { return reserved.size(); }
        List<Long> keys() { return List.copyOf(reserved.keySet()); }
        boolean commit() {
            synchronized (MockMemoryBlockCache.this) {
                if (closed) return false;
                closed = true;
                boolean changed = false;
                for (var r : reserved.entrySet()) {
                    if (pending.remove(r.getKey(), r.getValue())) {
                        r.getValue().born = clock.getAsLong();
                        entries.put(r.getKey(), r.getValue());
                        changed = true;
                    }
                }
                return changed;
            }
        }
        public void close() {
            synchronized (MockMemoryBlockCache.this) {
                if (closed) return;
                closed = true;
                reserved.forEach((k, e) -> pending.remove(k, e));
            }
        }
    }

    synchronized WriteLease beginWrite(List<Long> keys) {
        Set<Long> missing = new LinkedHashSet<>();
        // No get(): an already committed prefix needs no copy or LRU refresh.
        for (Long k : keys) if (!entries.containsKey(k) && !pending.containsKey(k)) missing.add(k);
        if (missing.size() > availableBlocks()) { writeRejected++; return null; }
        while (capacity - entries.size() - pending.size() < missing.size()) {
            if (!evictOne()) { writeRejected++; return null; }
        }
        Map<Long, Entry> reserved = new LinkedHashMap<>();
        for (Long k : missing) {
            Entry e = new Entry(0); pending.put(k, e); reserved.put(k, e);
        }
        return new WriteLease(reserved);
    }

    private boolean evictOne() {
        var it = entries.entrySet().iterator();
        while (it.hasNext()) {
            Entry e = it.next().getValue();
            if (e.pins > 0) continue;
            it.remove(); evictions++;
            onEviction.accept(Math.max(0L, clock.getAsLong() - e.born) / 1_000_000.0);
            return true;
        }
        return false;
    }

    /** Compatibility path when copy lifecycle is disabled. */
    synchronized void write(List<Long> keys) {
        for (Long k : keys) {
            if (entries.get(k) != null || pending.containsKey(k)) continue;
            if (entries.size() + pending.size() == capacity && !evictOne()) { writeRejected++; break; }
            entries.put(k, new Entry(clock.getAsLong()));
        }
    }
    synchronized int pinnedBlocks() { return pinned; }
    synchronized int pendingBlocks() { return pending.size(); }
    synchronized int availableBlocks() { return capacity - pinnedBlocks() - pending.size(); }
    synchronized long writeRejected() { return writeRejected; }
    synchronized List<Long> keys() { return List.copyOf(entries.keySet()); }
    synchronized int size() { return entries.size(); }
    int capacity() { return capacity; }
    synchronized long evictions() { return evictions; }
    synchronized void clear() { generation++; pinned = 0; entries.clear(); pending.clear(); }
}
