package org.flexlb.mock.engine.cache;

import java.util.*;
import static org.flexlb.mock.engine.cache.BlockPool.Kind.*;

/** One local RTP Full-KV pool with synchronous LRU reclamation. */
public final class KvCache implements AutoCloseable {
    public enum Failure { PERMANENT, RETRYABLE }
    public record Admission(Lease lease, Failure failure, long generation, long matchNanos) {}
    public record Snapshot(int totalBlocks, int availableBlocks, int freeBlocks, int heldBlocks,
                           int referencedBlocks, long version, Set<Long> keys) {}
    private final BlockPool device;
    private final BlockTree tree = new BlockTree();
    private final int blockSize, reserve;
    private final boolean deviceCache;
    private final Set<Lease> active = Collections.newSetFromMap(new IdentityHashMap<>());
    private boolean closed;
    private java.util.function.Consumer<CacheMetadata.Change> metadataObserver;
    private java.util.function.LongConsumer evictionLifetime = nanos -> {};
    private Runnable changed = () -> {};

    /** The observer must only signal; it must never acquire an Engine scheduling lock. */
    public synchronized void onChange(Runnable observer) { changed = Objects.requireNonNull(observer); }

    public synchronized void onEvictionLifetime(java.util.function.LongConsumer observer) {
        evictionLifetime = Objects.requireNonNull(observer);
    }

    public synchronized int itemCount() { return tree.size(); }

    public KvCache(int blockSize, int blocks, int reserve) {
        this(blockSize, blocks, reserve, true);
    }

    public KvCache(int blockSize, int blocks, int reserve, boolean deviceCache) {
        if (blockSize < 1 || blocks <= 0 || reserve < 0 || reserve > blocks)
            throw new IllegalArgumentException("Invalid block capacity/reserve");
        device = new BlockPool(blocks);
        this.blockSize = blockSize;
        this.reserve = reserve;
        this.deviceCache = deviceCache;
    }

    /** Maximum size of one request, independent of current leases and evictable cache entries. */
    public synchronized long maxRequestTokens() { return (long) (device.snapshot().total() - reserve) * blockSize; }

    public synchronized Admission acquire(long requiredTokens, List<Long> keys) {
        if (closed) throw new IllegalStateException("Cache is closed");
        validate(keys);
        long blockDemand = blocksFor(requiredTokens);
        if (keys.size() > requiredTokens / blockSize) throw new IllegalArgumentException("Keys exceed complete input blocks");
        tree.validatePath(keys);
        if (requiredTokens > maxRequestTokens()) return rejected(Failure.PERMANENT);
        int demand = (int) blockDemand;
        long matchStarted = System.nanoTime();
        var matched = tree.match(keys);
        long matchNanos = System.nanoTime() - matchStarted;
        var slots = new ArrayList<Slot>();
        for (var node : matched) slots.add(new Slot(device.retain(node.cache, REQUEST), node));
        int need = demand - matched.size();
        if ((long) need + reserve > device.snapshot().available() || !makeDeviceFree(need)) {
            for (var slot : slots) slot.reference.close();
            return rejected(Failure.RETRYABLE);
        }
        for (var ref : Objects.requireNonNull(device.allocate(need, REQUEST))) slots.add(new Slot(ref, null));
        var lease = new Lease(slots, matched.size());
        active.add(lease);
        return new Admission(lease, null, allocationGeneration(), matchNanos);
    }

    /** Snapshot and subscription share the cache monitor. Observers must only enqueue, never block. */
    public synchronized CacheMetadata.SnapshotKeys metadataKeys() { return tree.metadataKeys(); }
    public synchronized AutoCloseable observeMetadata(java.util.function.Consumer<CacheMetadata.Change> observer) {
        if (closed) throw new IllegalStateException("Cache is closed");
        if (metadataObserver != null) throw new IllegalStateException("Cache metadata already has a reporter");
        metadataObserver = Objects.requireNonNull(observer);
        tree.observe(observer);
        return () -> { synchronized (KvCache.this) {
            if (metadataObserver == observer) { metadataObserver = null; tree.observe(change -> {}); }
        } };
    }
    private Admission rejected(Failure failure) { return new Admission(null, failure, allocationGeneration(), 0); }
    /** Both freed capacity and newly reusable prefixes can make a rejected request admissible. */
    public synchronized long allocationGeneration() { return device.generation() + tree.version(); }
    public synchronized Snapshot snapshot(long latestVersion, boolean needKeys) {
        var state = device.snapshot();
        int indexed = device.cachedActiveBlocks();
        return new Snapshot(state.total(), state.available(), state.free(), state.active() - indexed, indexed,
            needKeys ? tree.version() : latestVersion, needKeys && latestVersion != tree.version() ? tree.keys() : Set.of());
    }

    public final class Lease implements AutoCloseable {
        private final List<Slot> slots;
        private final int reused;
        private boolean terminal;
        private Lease(List<Slot> slots, int reused) {
            this.slots = slots; this.reused = reused;
        }
        public long reusedTokens() { synchronized (KvCache.this) { return terminal ? 0 : (long) reused * blockSize; } }
        public Failure ensureCapacity(long requiredTokens) {
            synchronized (KvCache.this) {
                requireOpen();
                long target = blocksFor(requiredTokens);
                if (target <= slots.size()) return null;
                if (target > device.snapshot().total()) return Failure.PERMANENT;
                int need = (int) target - slots.size();
                if (need > device.snapshot().available() || !makeDeviceFree(need)) return Failure.RETRYABLE;
                for (var ref : Objects.requireNonNull(device.allocate(need, REQUEST))) slots.add(new Slot(ref, null));
                return null;
            }
        }
        /** Publication does not release request ownership. Only complete immutable blocks are accepted. */
        public boolean publish(List<Long> keys, long validTokens) {
            synchronized (KvCache.this) {
                if (terminal) return false;
                validate(keys); tree.validatePath(keys);
                if (validTokens < 0 || validTokens > (long) slots.size() * blockSize || keys.size() > validTokens / blockSize)
                    throw new IllegalArgumentException("Published prefix exceeds valid KV or allocated capacity");
                for (int i = 0; i < keys.size(); i++) {
                    var slot = slots.get(i);
                    if (slot.key != null && !slot.key.equals(keys.get(i)))
                        throw new IllegalArgumentException("Computed prefix differs from reused blocks");
                }
                if (!deviceCache) return false;
                BlockTree.Node parent = null;
                for (int i = 0; i < keys.size(); i++) {
                    var slot = slots.get(i);
                    var node = tree.find(keys.get(i));
                    if (node == null) {
                        node = tree.insert(keys.get(i), parent, device.retain(slot.reference, CACHE));
                        slot.key = node.key;
                    }
                    parent = node;
                }
                return true;
            }
        }

        @Override public void close() {
            synchronized (KvCache.this) {
                if (terminal) return;
                int available = device.snapshot().available();
                for (var slot : slots) {
                    slot.reference.close();
                }
                slots.clear(); terminal = true; active.remove(this);
                if (device.snapshot().available() > available) changed.run();
            }
        }
        private void requireOpen() { if (terminal) throw new IllegalStateException("Lease is closed"); }
    }
    @Override public void close() {
        synchronized (this) {
            if (closed) return;
            closed = true;
            for (var lease : List.copyOf(active)) lease.close();
            tree.clear();
        }
    }
    private boolean makeDeviceFree(int count) {
        while (device.snapshot().free() < count) {
            var evicted = tree.evict();
            if (evicted == null) return false;
            evictionLifetime.accept(System.nanoTime() - evicted.insertedNanos);
        }
        return true;
    }
    private long blocksFor(long tokens) {
        if (tokens < 0) throw new IllegalArgumentException("Negative Token capacity");
        return tokens / blockSize + (tokens % blockSize == 0 ? 0 : 1);
    }
    private static final class Slot {
        final BlockPool.Reference reference;
        Long key;
        Slot(BlockPool.Reference reference, BlockTree.Node node) { this.reference = reference; key = node == null ? null : node.key; }
    }
    private static void validate(List<Long> keys) {
        if (keys == null || keys.stream().anyMatch(Objects::isNull) || new HashSet<>(keys).size() != keys.size())
            throw new IllegalArgumentException("Cache keys must be distinct, non-null prefix identities");
    }
}
