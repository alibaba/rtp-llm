package org.flexlb.mock.engine.rtp.cache;

import java.util.*;
import org.flexlb.mock.cache.BlockPool;
import static org.flexlb.mock.cache.BlockPool.Kind.*;

/** RTP admission and cache publication; shared BlockPool owns every physical reference. */
public final class DeviceCache implements AutoCloseable {
    public enum Failure { PERMANENT, RETRYABLE }
    public enum EvictionPolicy { LRU, LFU, FIFO }
    public record Admission(Lease lease, Failure failure) {}
    public record Snapshot(int totalBlocks, int availableBlocks, int freeBlocks, int heldBlocks,
                           int referencedBlocks, long version, Set<Long> keys) {}
    private final BlockPool pool;
    private final BlockTree tree;
    private final int reserve;
    private final Set<Lease> active = Collections.newSetFromMap(new IdentityHashMap<>());
    private boolean closed;

    public DeviceCache(int blocks, int reserve, EvictionPolicy policy) {
        if (blocks <= 0 || reserve < 0 || reserve > blocks) throw new IllegalArgumentException("Invalid block capacity/reserve");
        pool = new BlockPool(blocks); tree = new BlockTree(Objects.requireNonNull(policy)); this.reserve = reserve;
    }
    public synchronized Admission acquire(int demand, List<Long> keys) {
        if (closed) throw new IllegalStateException("Cache is closed");
        validate(keys);
        if (demand < 0 || keys.size() > demand) throw new IllegalArgumentException("Invalid block demand/key count");
        tree.validatePath(keys);
        var slots = new ArrayList<Slot>();
        for (var node : tree.match(keys)) {
            slots.add(new Slot(pool.retain(node.cache, REQUEST), node)); tree.pin(node);
        }
        int reused = slots.size(), need = demand - reused;
        if ((long) need + reserve > pool.snapshot().available() || !makeFree(need)) {
            release(slots);
            return new Admission(null, failure(need));
        }
        for (var reference : Objects.requireNonNull(pool.allocate(need, REQUEST))) slots.add(new Slot(reference, null));
        var lease = new Lease(slots, reused); active.add(lease);
        return new Admission(lease, null);
    }
    public long allocationGeneration() { return pool.generation(); }
    public void awaitCapacityChange(long observed, long waitMs) throws InterruptedException { pool.awaitChange(observed, waitMs); }
    public synchronized Snapshot snapshot(long latestVersion, boolean needKeys) {
        var state = pool.snapshot();
        int indexed = tree.pinnedBlocks();
        return new Snapshot(state.total(), state.available(), state.free(), state.active() - indexed, indexed,
            needKeys ? tree.version() : latestVersion, needKeys && latestVersion != tree.version() ? tree.keys() : Set.of());
    }
    public final class Lease implements AutoCloseable {
        private final List<Slot> slots;
        private final int reused;
        private boolean terminal;
        private Lease(List<Slot> slots, int reused) { this.slots = slots; this.reused = reused; }
        public int blocks() { synchronized (DeviceCache.this) { return terminal ? 0 : slots.size(); } }
        public int reusedBlocks() { synchronized (DeviceCache.this) { return terminal ? 0 : reused; } }
        public Failure growTo(int target) {
            synchronized (DeviceCache.this) {
                if (terminal) throw new IllegalStateException("Lease is closed");
                if (target < slots.size()) throw new IllegalArgumentException("Growth cannot shrink a lease");
                while (slots.size() < target) {
                    if (!makeFree(1)) return failure(target);
                    slots.add(new Slot(Objects.requireNonNull(pool.allocate(1, REQUEST)).getFirst(), null));
                }
                return null;
            }
        }
        public boolean complete(List<Long> keys) {
            synchronized (DeviceCache.this) {
                if (terminal) return false;
                validate(keys); tree.validatePath(keys);
                if (keys.size() > slots.size()) throw new IllegalArgumentException("Cannot publish more blocks than owned");
                for (int i = 0; i < keys.size(); i++) {
                    var slot = slots.get(i);
                    if (slot.node != null && slot.node.key != keys.get(i))
                        throw new IllegalArgumentException("Computed prefix differs from reused blocks");
                }
                BlockTree.Node parent = null;
                for (int i = 0; i < keys.size(); i++) {
                    var slot = slots.get(i);
                    var node = tree.find(keys.get(i));
                    if (node == null) {
                        node = tree.insert(keys.get(i), parent, pool.retain(slot.reference, CACHE));
                        slot.node = node; tree.pin(node);
                    }
                    parent = node; // An overlapping insertion does not overwrite or refresh an existing node.
                }
                close(); return true;
            }
        }
        @Override public void close() {
            synchronized (DeviceCache.this) {
                if (terminal) return;
                release(slots); slots.clear(); terminal = true; active.remove(this);
            }
        }
    }
    @Override public synchronized void close() {
        if (closed) return;
        for (var lease : List.copyOf(active)) lease.close();
        tree.clear(); closed = true;
    }
    private boolean makeFree(int count) {
        while (pool.snapshot().free() < count) if (!tree.evict()) return false;
        return true;
    }
    private Failure failure(int demand) {
        return (long) demand + reserve > pool.snapshot().total() ? Failure.PERMANENT : Failure.RETRYABLE;
    }
    private void release(List<Slot> slots) {
        for (var slot : slots) {
            if (slot.node != null) tree.unpin(slot.node);
            slot.reference.close();
        }
    }
    private static final class Slot {
        final BlockPool.Reference reference;
        BlockTree.Node node;
        Slot(BlockPool.Reference reference, BlockTree.Node node) { this.reference = reference; this.node = node; }
    }
    private static void validate(List<Long> keys) {
        if (keys == null || keys.stream().anyMatch(Objects::isNull) || new HashSet<>(keys).size() != keys.size())
            throw new IllegalArgumentException("Cache keys must be distinct, non-null prefix identities");
    }
}
