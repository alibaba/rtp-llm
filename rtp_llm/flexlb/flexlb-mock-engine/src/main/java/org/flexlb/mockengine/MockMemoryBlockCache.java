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
    private final Set<Entry> pending = new HashSet<>();
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
                    if (pending.remove(r.getValue())) {
                        // putCommitted touches an existing complete key, retaining its
                        // backing, birth time and in-flight readers. The duplicate copy
                        // backing is released rather than replacing the existing entry.
                        if (entries.get(r.getKey()) == null) {
                            r.getValue().born = clock.getAsLong();
                            entries.put(r.getKey(), r.getValue());
                            changed = true;
                        }
                    }
                }
                return changed;
            }
        }
        public void close() {
            synchronized (MockMemoryBlockCache.this) {
                if (closed) return;
                closed = true;
                reserved.values().forEach(pending::remove);
            }
        }
    }

    synchronized WriteLease beginWrite(List<Long> keys) {
        // Real asyncWrite skips only the contiguous committed prefix, without
        // touching it. A hole starts a copy plan for the entire remaining suffix.
        int start = 0;
        while (start < keys.size() && entries.containsKey(keys.get(start))) start++;
        Map<Long, Entry> reserved = new LinkedHashMap<>();
        for (int i = start; i < keys.size(); i++) {
            Long key = keys.get(i);
            if (reserved.containsKey(key)) continue;
            // Allocate one backing at a time, including already-cached suffix
            // keys and concurrent copies. Prior evictions survive a later failure.
            while (entries.size() + pending.size() >= capacity) {
                if (!evictOne()) {
                    reserved.values().forEach(pending::remove);
                    writeRejected++;
                    return null;
                }
            }
            Entry e = new Entry(0);
            pending.add(e);
            reserved.put(key, e);
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

    /** Synchronous copy path; the lifecycle switch only controls explicit pins. */
    synchronized void write(List<Long> keys) {
        var write = beginWrite(keys);
        if (write != null) write.commit();
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
