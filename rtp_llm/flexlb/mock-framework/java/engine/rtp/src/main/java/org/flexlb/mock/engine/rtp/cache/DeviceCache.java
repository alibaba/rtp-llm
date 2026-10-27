package org.flexlb.mock.engine.rtp.cache;

import java.util.*;

/** RTP-specific block ownership. Raw pool leases never escape this owner. */
public final class DeviceCache implements AutoCloseable {
    public enum Failure { PERMANENT, RETRYABLE }
    public record Admission(Lease lease, Failure failure) {}
    public record Snapshot(int totalBlocks, int availableBlocks, int freeBlocks, int heldBlocks,
                           int referencedBlocks, long version, Set<Long> keys) {}
    private final BlockPool pool;
    private final Set<Lease> active = Collections.newSetFromMap(new IdentityHashMap<>());
    private final Set<Long> keys = new HashSet<>();
    private long version;
    private boolean closed;

    public DeviceCache(int blocks, int reserve, boolean prefixTree) {
        if (blocks <= 0 || reserve < 0 || reserve > blocks) throw new IllegalArgumentException("Invalid block capacity/reserve");
        pool = new BlockPool(blocks, 0);
        pool.setObservedReserveBlocks(reserve);
        pool.setPrefixTreeEnabled(prefixTree);
        pool.setKeyListener((key, added) -> {
            boolean changed = added ? keys.add(key) : keys.remove(key);
            if (!changed) throw new IllegalStateException("Unbalanced cache key publication");
            version++;
        });
    }

    public synchronized Admission acquire(int blocks, List<Long> cacheKeys) {
        ensureOpen(); validate(cacheKeys);
        if (blocks < 0 || cacheKeys.size() > blocks) throw new IllegalArgumentException("Invalid block demand/key count");
        var result = pool.acquireWithReuseDetailed(blocks, cacheKeys);
        if (result.lease() == null) return new Admission(null, Failure.valueOf(result.failure().name()));
        var lease = new Lease(result.lease()); active.add(lease);
        return new Admission(lease, null);
    }

    public synchronized int prefixHitBlocks(List<Long> cacheKeys) {
        ensureOpen(); validate(cacheKeys); return pool.peekPrefixHitBlocks(cacheKeys);
    }

    /** Version is about published keys, not occupancy; capacity is always returned fresh. */
    public synchronized Snapshot snapshot(long latestVersion, boolean needKeys) {
        return new Snapshot(pool.totalBlocks(), pool.availableBlocks(), pool.freeBlocks(), pool.heldBlocks(),
            pool.referencedKeyBlocks(), needKeys ? version : latestVersion,
            needKeys && latestVersion != version ? Set.copyOf(keys) : Set.of());
    }

    public final class Lease implements AutoCloseable {
        private BlockPool.BlockLease raw;
        private boolean terminal, published;
        private Lease(BlockPool.BlockLease raw) { this.raw = raw; }
        public int blocks() { synchronized (DeviceCache.this) { return terminal ? 0 : raw.totalBlocks(); } }
        public int reusedBlocks() { synchronized (DeviceCache.this) { return terminal ? 0 : raw.hitKeys.size(); } }
        public Failure growTo(int target) {
            synchronized (DeviceCache.this) {
                if (terminal) throw new IllegalStateException("Lease is closed");
                if (target < raw.totalBlocks()) throw new IllegalArgumentException("Growth cannot shrink a lease");
                var result = pool.growTo(raw, target);
                return result.success() ? null : Failure.valueOf(result.failure().name());
            }
        }
        /** Computed P data is published, but remains pinned until connector/owner releases it. */
        public boolean publish(List<Long> computedKeys) {
            synchronized (DeviceCache.this) {
                if (terminal || published) return false;
                materialized(computedKeys);
                raw = pool.retainComputed(raw, computedKeys); published = true;
                return true;
            }
        }
        public boolean complete(List<Long> computedKeys) {
            synchronized (DeviceCache.this) {
                if (terminal) return false;
                materialized(computedKeys);
                pool.admit(raw, computedKeys); terminal = true; active.remove(this);
                return true;
            }
        }
        @Override public void close() {
            synchronized (DeviceCache.this) {
                if (terminal) return;
                pool.release(raw); terminal = true; active.remove(this);
            }
        }
        private void materialized(List<Long> computedKeys) {
            validate(computedKeys);
            if (computedKeys.size() > raw.totalBlocks()) throw new IllegalArgumentException("Cannot publish more blocks than owned");
        }
    }

    @Override public synchronized void close() {
        if (closed) return;
        for (var lease : List.copyOf(active)) lease.close();
        pool.clear(); closed = true;
    }

    private void ensureOpen() { if (closed) throw new IllegalStateException("Cache is closed"); }
    private static void validate(List<Long> values) {
        if (values == null || values.stream().anyMatch(Objects::isNull) || new HashSet<>(values).size() != values.size())
            throw new IllegalArgumentException("Cache keys must be distinct, non-null prefix identities");
    }
}
