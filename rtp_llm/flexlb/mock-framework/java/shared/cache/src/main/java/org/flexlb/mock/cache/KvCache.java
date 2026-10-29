package org.flexlb.mock.cache;

import java.util.*;
import static org.flexlb.mock.cache.BlockPool.Kind.*;

/** RTP Full-KV storage. Local tier copies are instantaneous under this owner's monitor. */
public final class KvCache implements AutoCloseable {
    public enum Failure { PERMANENT, RETRYABLE }
    public enum EvictionPolicy { LRU, LFU, FIFO }
    public record Watermark(double low, double high) {
        public Watermark {
            if (!Double.isFinite(low) || !Double.isFinite(high) || low <= 0 || low >= high || high > 1)
                throw new IllegalArgumentException("Watermarks require 0 < low < high <= 1");
        }
    }
    public record Admission(Lease lease, Failure failure, long generation) {}
    public record Snapshot(int totalBlocks, int availableBlocks, int freeBlocks, int heldBlocks,
                           int referencedBlocks, long version, Set<Long> keys) {}
    private final BlockPool device, host;
    private final BlockTree tree;
    private final int reserve;
    private final boolean deviceCache;
    private final Watermark deviceWatermark, hostWatermark;
    private final Set<Lease> active = Collections.newSetFromMap(new IdentityHashMap<>());
    private boolean closed, balancing, deviceTriggered, hostTriggered;
    private Runnable capacityChanged = () -> {};

    /** The observer must only signal; it must never acquire an Engine scheduling lock. */
    public synchronized void onCapacityChange(Runnable observer) { capacityChanged = Objects.requireNonNull(observer); }

    public KvCache(int blocks, int reserve, EvictionPolicy policy) {
        this(blocks, reserve, policy, true, 0, policy, null, null);
    }
    public KvCache(int blocks, int reserve, EvictionPolicy policy, boolean deviceCache, int hostBlocks,
                   EvictionPolicy hostPolicy, Watermark deviceWatermark, Watermark hostWatermark) {
        if (blocks <= 0 || reserve < 0 || reserve > blocks || hostBlocks < 0)
            throw new IllegalArgumentException("Invalid block capacity/reserve");
        device = new BlockPool(blocks); host = new BlockPool(hostBlocks);
        tree = new BlockTree(Objects.requireNonNull(policy), Objects.requireNonNull(hostPolicy));
        this.reserve = reserve; this.deviceCache = deviceCache;
        this.deviceWatermark = deviceWatermark; this.hostWatermark = hostWatermark;
    }
    public synchronized Admission acquire(int demand, List<Long> keys) {
        if (closed) throw new IllegalStateException("Cache is closed");
        validate(keys);
        if (demand < 0 || keys.size() > demand) throw new IllegalArgumentException("Invalid block demand/key count");
        tree.validatePath(keys);
        if ((long) demand + reserve > device.snapshot().total()) return rejected(Failure.PERMANENT);
        var matched = tree.match(keys);
        matched.forEach(tree::pin);
        var direct = new IdentityHashMap<BlockTree.Node, BlockPool.Reference>();
        for (var node : matched) if (!node.host) direct.put(node, device.retain(node.cache, REQUEST));
        int need = demand - direct.size();
        if ((long) need + reserve > device.snapshot().available() || !makeDeviceFree(need)) {
            direct.values().forEach(BlockPool.Reference::close); matched.forEach(tree::unpin);
            return rejected(Failure.RETRYABLE);
        }
        var allocated = Objects.requireNonNull(device.allocate(need, REQUEST)).iterator();
        var slots = new ArrayList<Slot>();
        for (var node : matched) {
            if (node.host) {
                var target = allocated.next();
                if (deviceCache) {
                    tree.relocate(node, device.retain(target, CACHE), false);
                    slots.add(new Slot(target, node));
                } else {
                    // Host remains indexed; the request's working copy stays private.
                    tree.unpin(node); slots.add(new Slot(target, null));
                }
            } else slots.add(new Slot(direct.get(node), node));
        }
        while (allocated.hasNext()) slots.add(new Slot(allocated.next(), null));
        var lease = new Lease(slots, matched.size()); active.add(lease);
        return new Admission(lease, null, device.generation());
    }
    private Admission rejected(Failure failure) { return new Admission(null, failure, device.generation()); }
    public long allocationGeneration() { return device.generation(); }
    public synchronized Snapshot snapshot(long latestVersion, boolean needKeys) {
        var state = device.snapshot();
        int indexed = tree.pinnedDeviceBlocks();
        return new Snapshot(state.total(), state.available(), state.free(), state.active() - indexed, indexed,
            needKeys ? tree.version() : latestVersion, needKeys && latestVersion != tree.version() ? tree.keys() : Set.of());
    }
    public synchronized BlockPool.Snapshot hostSnapshot() { return host.snapshot(); }

    public final class Lease implements AutoCloseable {
        private final List<Slot> slots;
        private final int reused;
        private int validBlocks;
        private boolean terminal;
        private Lease(List<Slot> slots, int reused) { this.slots = slots; this.reused = reused; validBlocks = reused; }
        public int blocks() { synchronized (KvCache.this) { return terminal ? 0 : slots.size(); } }
        public int reusedBlocks() { synchronized (KvCache.this) { return terminal ? 0 : reused; } }
        public void markValid(int blocks) {
            synchronized (KvCache.this) {
                requireOpen();
                if (blocks < validBlocks || blocks > slots.size()) throw new IllegalArgumentException("Invalid KV validity frontier");
                validBlocks = blocks;
            }
        }
        public Failure growTo(int target) {
            synchronized (KvCache.this) {
                requireOpen();
                if (target < slots.size()) throw new IllegalArgumentException("Growth cannot shrink a lease");
                if (target > device.snapshot().total()) return Failure.PERMANENT;
                int need = target - slots.size();
                if (need > device.snapshot().available() || !makeDeviceFree(need)) return Failure.RETRYABLE;
                for (var ref : Objects.requireNonNull(device.allocate(need, REQUEST))) slots.add(new Slot(ref, null));
                return null;
            }
        }
        /** Publication does not release request ownership. Only complete immutable blocks are accepted. */
        public boolean publish(List<Long> keys) {
            synchronized (KvCache.this) {
                if (terminal) return false;
                validate(keys); tree.validatePath(keys);
                if (keys.size() > validBlocks) throw new IllegalArgumentException("Cannot publish invalid KV");
                for (int i = 0; i < keys.size(); i++) {
                    var slot = slots.get(i);
                    if (slot.node != null && slot.node.key != keys.get(i))
                        throw new IllegalArgumentException("Computed prefix differs from reused blocks");
                }
                var publicationPins = new ArrayList<BlockTree.Node>();
                try {
                    BlockTree.Node parent = null;
                    for (int i = 0; i < keys.size(); i++) {
                        var slot = slots.get(i);
                        var node = tree.find(keys.get(i));
                        if (node == null) {
                            if (deviceCache) {
                                node = tree.insert(keys.get(i), parent, device.retain(slot.reference, CACHE), false);
                                slot.node = node; tree.pin(node);
                            } else if (host.snapshot().total() > 0) {
                                if (!makeHostFree(1)) break; // Cache retention is optional; request success is not.
                                var ref = Objects.requireNonNull(host.allocate(1, CACHE)).getFirst();
                                node = tree.insert(keys.get(i), parent, ref, true);
                            } else break;
                        }
                        tree.pin(node); publicationPins.add(node);
                        parent = node;
                    }
                    return true;
                } finally { publicationPins.forEach(tree::unpin); }
            }
        }
        @Override public void close() {
            synchronized (KvCache.this) {
                if (terminal) return;
                int available = device.snapshot().available();
                for (var slot : slots) {
                    if (slot.node != null) tree.unpin(slot.node);
                    slot.reference.close();
                }
                slots.clear(); terminal = true; active.remove(this);
                if (!closed) balance();
                if (device.snapshot().available() > available) capacityChanged.run();
            }
        }
        private void requireOpen() { if (terminal) throw new IllegalStateException("Lease is closed"); }
    }
    @Override public synchronized void close() {
        if (closed) return;
        closed = true;
        for (var lease : List.copyOf(active)) lease.close();
        tree.clear();
    }
    private boolean makeDeviceFree(int count) {
        // Admission may drop idle cache directly, matching RTP's forced reclamation path.
        while (device.snapshot().free() < count) {
            if (!tree.evict(false) && !tree.evict(true)) return false;
        }
        return true;
    }
    private boolean makeHostFree(int count) {
        while (host.snapshot().free() < count) if (!tree.evict(true)) return false;
        return true;
    }
    private void balance() {
        if (balancing) return;
        balancing = true;
        try {
            if (deviceWatermark != null) {
                deviceTriggered |= used(device) >= Math.ceil(device.snapshot().total() * deviceWatermark.high());
                int low = (int) Math.floor(device.snapshot().total() * deviceWatermark.low());
                while (deviceTriggered && used(device) > low) {
                    var node = tree.movableDevice();
                    if (node == null) break;
                    if (host.snapshot().total() > 0 && makeHostFree(1)) {
                        var target = Objects.requireNonNull(host.allocate(1, CACHE)).getFirst();
                        tree.relocate(node, target, true);
                    } else if (!tree.evict(false)) break;
                }
                deviceTriggered &= used(device) > low;
            }
            if (hostWatermark != null && host.snapshot().total() > 0) {
                hostTriggered |= used(host) >= Math.ceil(host.snapshot().total() * hostWatermark.high());
                int low = (int) Math.floor(host.snapshot().total() * hostWatermark.low());
                while (hostTriggered && used(host) > low && tree.evict(true)) { }
                hostTriggered &= used(host) > low;
            }
        } finally { balancing = false; }
    }
    private static int used(BlockPool pool) { return pool.snapshot().total() - pool.snapshot().free(); }
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
