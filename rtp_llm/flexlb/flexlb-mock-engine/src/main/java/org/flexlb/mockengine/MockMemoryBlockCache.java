package org.flexlb.mockengine;

import java.util.*;
import java.util.function.DoubleConsumer;
import java.util.function.LongSupplier;

/** Metadata host pool mirroring production's optional prefix-tree memory cache. */
final class MockMemoryBlockCache {
    private static final class Entry {
        final long key;
        long born;
        long accessSequence;
        boolean resident;
        int pins;
        Entry(long key, long born, boolean resident) {
            this.key = key; this.born = born; this.resident = resident;
        }
    }
    private static final class Node {
        Long parent;
        final Set<Long> children = new HashSet<>();
        Entry entry;
        int committedSubtree;
    }
    private record Pending(Entry entry, Long parent) {}
    private final int capacity;
    private final boolean prefixTreeEnabled;
    private final LongSupplier clock;
    private final LinkedHashMap<Long, Entry> entries = new LinkedHashMap<>(16, .75f, true);
    private final Map<Long, Node> tree = new HashMap<>();
    private final Set<Entry> pending = new HashSet<>();
    private final NavigableSet<Entry> victims = new TreeSet<>(Comparator
            .comparingLong((Entry e) -> e.accessSequence).thenComparingLong(e -> e.key));
    private final DoubleConsumer onEviction;
    private long evictions, writeRejected, generation, accessSequence;
    private int pinned;

    MockMemoryBlockCache(int capacity, DoubleConsumer onEviction) {
        this(capacity, true, onEviction, System::nanoTime);
    }
    MockMemoryBlockCache(int capacity, DoubleConsumer onEviction, LongSupplier clock) {
        this(capacity, true, onEviction, clock);
    }
    MockMemoryBlockCache(int capacity, boolean prefixTreeEnabled,
                         DoubleConsumer onEviction, LongSupplier clock) {
        if (capacity <= 0) throw new IllegalArgumentException("memory cache capacity must be positive");
        this.capacity = capacity;
        this.prefixTreeEnabled = prefixTreeEnabled;
        this.onEviction = onEviction;
        this.clock = clock;
    }

    synchronized int match(List<Long> keys, int start) {
        int end = start;
        while (end < keys.size()) {
            Entry entry = entries.get(keys.get(end));
            if (entry == null) break;
            touch(entry);
            end++;
        }
        return end - start;
    }
    synchronized int peekMatch(List<Long> keys, int start) {
        int end = start;
        while (end < keys.size() && entries.containsKey(keys.get(end))) end++;
        return end - start;
    }

    final class ReadLease implements AutoCloseable {
        private final LinkedHashMap<Long, Entry> held;
        private final long epoch = generation;
        private boolean closed;
        ReadLease(LinkedHashMap<Long, Entry> held) { this.held = held; }
        int blocks() { return held.size(); }
        /** Successful H2D: consume the host entries and return their backing capacity. */
        int consume() {
            synchronized (MockMemoryBlockCache.this) {
                if (closed) return 0;
                closed = true;
                int removed = 0;
                var keys = new ArrayList<>(held.keySet());
                Collections.reverse(keys);
                for (Long key : keys) {
                    Entry entry = held.get(key);
                    releasePin(entry);
                    if (epoch == generation && entries.get(key) == entry) {
                        removeEntry(key, entry);
                        removed++;
                    }
                }
                return removed;
            }
        }
        public void close() {
            synchronized (MockMemoryBlockCache.this) {
                if (closed) return;
                closed = true;
                for (Entry e : held.values()) releasePin(e);
            }
        }
        private void releasePin(Entry entry) {
            entry.pins--;
            if (epoch == generation && entry.pins == 0) {
                pinned--;
                refreshVictim(tree.get(entry.key));
            }
        }
    }
    synchronized ReadLease pinRead(List<Long> keys, int start) {
        LinkedHashMap<Long, Entry> held = new LinkedHashMap<>();
        for (int i = start; i < keys.size(); i++) {
            Long key = keys.get(i);
            Entry e = entries.get(key);
            if (e == null) break;
            touch(e);
            if (e.pins++ == 0) { pinned++; victims.remove(e); }
            held.put(key, e);
        }
        return new ReadLease(held);
    }

    final class WriteLease implements AutoCloseable {
        private final Map<Long, Pending> reserved;
        private boolean closed;
        WriteLease(Map<Long, Pending> reserved) { this.reserved = reserved; }
        int blocks() { return reserved.size(); }
        List<Long> keys() { return List.copyOf(reserved.keySet()); }
        boolean commit() {
            synchronized (MockMemoryBlockCache.this) {
                if (closed) return false;
                closed = true;
                boolean changed = false;
                for (var r : reserved.entrySet()) {
                    Pending item = r.getValue();
                    if (pending.remove(item.entry())) {
                        // putCommitted touches an existing complete key, retaining its
                        // backing, birth time and in-flight readers. The duplicate copy
                        // backing is released rather than replacing the existing entry.
                        if (!entries.containsKey(r.getKey())) {
                            Entry entry = item.entry();
                            entry.born = clock.getAsLong();
                            entry.accessSequence = ++accessSequence;
                            entries.put(r.getKey(), entry);
                            indexNode(r.getKey(), item.parent());
                            tree.get(r.getKey()).entry = entry;
                            updateSubtree(r.getKey(), 1);
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
                reserved.values().forEach(item -> pending.remove(item.entry()));
            }
        }
    }

    synchronized WriteLease beginWrite(List<Long> keys) {
        return beginWrite(keys, false);
    }

    synchronized WriteLease beginWrite(List<Long> keys, boolean resident) {
        // Real asyncWrite skips only the contiguous committed prefix, without
        // touching it. A hole starts a copy plan for the entire remaining suffix.
        int start = 0;
        while (start < keys.size() && entries.containsKey(keys.get(start))) start++;
        Map<Long, Pending> reserved = new LinkedHashMap<>();
        for (int i = start; i < keys.size(); i++) {
            Long key = keys.get(i);
            if (reserved.containsKey(key)) continue;
            // Allocate one backing at a time, including already-cached suffix
            // keys and concurrent copies. Prior evictions survive a later failure.
            while (entries.size() + pending.size() >= capacity) {
                if (!evictOne()) {
                    reserved.values().forEach(item -> pending.remove(item.entry()));
                    writeRejected++;
                    return null;
                }
            }
            Entry e = new Entry(key, 0, resident);
            pending.add(e);
            reserved.put(key, new Pending(e, i == 0 ? null : keys.get(i - 1)));
        }
        return new WriteLease(reserved);
    }

    private boolean evictOne() {
        Entry selected = victims.pollFirst();
        if (selected == null) return false;
        removeEntry(selected.key, selected);
        evictions++;
        onEviction.accept(Math.max(0L, clock.getAsLong() - selected.born) / 1_000_000.0);
        return true;
    }

    private void touch(Entry entry) {
        victims.remove(entry);
        entry.accessSequence = ++accessSequence;
        refreshVictim(tree.get(entry.key));
    }

    private void refreshVictim(Node node) {
        if (node == null || node.entry == null) return;
        Entry entry = node.entry;
        victims.remove(entry);
        if (entry.pins == 0 && !entry.resident
                && (!prefixTreeEnabled || node.committedSubtree == 1)) victims.add(entry);
    }

    // Update only the prefix path. Eviction must not scan the entire host pool
    // or traverse each candidate's descendants for every newly allocated block.
    private void updateSubtree(Long key, int delta) {
        for (Long current = key; current != null;) {
            Node node = tree.get(current);
            if (node == null) break;
            node.committedSubtree += delta;
            refreshVictim(node);
            current = node.parent;
        }
    }

    private void indexNode(Long key, Long parent) {
        Node node = tree.computeIfAbsent(key, ignored -> new Node());
        if (!prefixTreeEnabled || node.parent != null || parent == null) return;
        node.parent = parent;
        tree.computeIfAbsent(parent, ignored -> new Node()).children.add(key);
        if (node.committedSubtree != 0) updateSubtree(parent, node.committedSubtree);
    }

    private void removeEntry(Long key, Entry expected) {
        if (entries.get(key) != expected) return;
        victims.remove(expected);
        entries.remove(key);
        tree.get(key).entry = null;
        updateSubtree(key, -1);
        pruneNode(key);
    }

    private void pruneNode(Long key) {
        Long current = key;
        while (current != null && !entries.containsKey(current)) {
            Node node = tree.get(current);
            if (node == null || !node.children.isEmpty()) return;
            tree.remove(current);
            Long parent = node.parent;
            if (parent != null) {
                Node parentNode = tree.get(parent);
                if (parentNode != null) parentNode.children.remove(current);
            }
            current = parent;
        }
    }

    /** Synchronous test/control helper for a completed D2H write. */
    synchronized void write(List<Long> keys) {
        var write = beginWrite(keys);
        if (write != null) write.commit();
    }
    synchronized void writeResident(List<Long> keys) {
        var write = beginWrite(keys, true);
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
    boolean prefixTreeEnabled() { return prefixTreeEnabled; }
    synchronized void clear() {
        generation++; pinned = 0; entries.clear(); tree.clear(); pending.clear(); victims.clear();
    }
}
