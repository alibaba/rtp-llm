package org.flexlb.mock.engine.cache;

import java.util.*;
import org.flexlb.mock.engine.config.RtpConfig;
import static org.flexlb.mock.engine.cache.BlockPool.Kind.*;

/**
 * Full-KV allocation, reuse and LRU; owns device accounting and KVCM metadata reporting.
 */
public final class KvCacheManager implements AutoCloseable {
    private static final long[] EMPTY_KEYS = new long[0];

    private final BlockPool pool;
    private final BlockTree tree = new BlockTree();

    private final int blockSize, deviceBlocks, reserve;
    private final boolean deviceCache;

    private final Set<Lease> active = Collections.newSetFromMap(new IdentityHashMap<>());

    private boolean closed;

    private volatile KvcmReporter reporter;
    private java.util.function.Consumer<CacheMetadata.Change> metadataObserver;
    private java.util.function.LongConsumer evictionLifetime = nanos -> {};
    private Runnable changed = () -> {};

    public KvCacheManager(int blockSize, int blocks, int reserve) {
        this(blockSize, blocks, reserve, true);
    }

    public KvCacheManager(int blockSize, int blocks, int reserve, boolean deviceCache) {
        this(blockSize, blocks, blocks, reserve, deviceCache);
    }

    public KvCacheManager(int blockSize, int deviceBlocks, int totalBlocks, int reserve,
                   boolean deviceCache) {
        if (blockSize < 1 || deviceBlocks <= 0 || totalBlocks < deviceBlocks || reserve < 0 || reserve > deviceBlocks)
            throw new IllegalArgumentException("Invalid device/total block capacity or reserve");
        pool = new BlockPool(totalBlocks);
        this.deviceBlocks = deviceBlocks;
        this.blockSize = blockSize;
        this.reserve = reserve;
        this.deviceCache = deviceCache;
    }

    /** The observer must only signal; it must never acquire an Engine scheduling lock. */
    public synchronized void onChange(Runnable observer) { changed = Objects.requireNonNull(observer); }

    public synchronized void onEvictionLifetime(java.util.function.LongConsumer observer) {
        evictionLifetime = Objects.requireNonNull(observer);
    }

    public synchronized int itemCount() { return tree.size(); }

    /** KVCM configuration, registration and reporting share the cache manager's lifetime. */
    public void startKvcmReporter(RtpConfig config, String workerAddress) {
        var values = config.business();
        String type = values.path("kv_cache_event_publisher_type").asText("none");
        if (!Set.of("none", "kvcm").contains(type)) throw new IllegalArgumentException("Invalid kv_cache_event_publisher_type");
        if (type.equals("none")) return;
        String instance = values.path("kv_cache_event_instance_id").asText();
        String host = values.path("kv_cache_event_host_ip_port").asText(workerAddress);
        if (instance.isBlank() || host.isBlank())
            throw new IllegalArgumentException("KVCM requires a shared instance_id and a nonempty Master-visible host_ip_port");
        var identity = new KvcmReporter.Identity(
            instance,
            values.path("kv_cache_event_instance_group").asText("default"),
            host,
            values.path("model_name").asText("mock-full-kv"), "FP16", false,
            config.physicalBlockSize(), config.cpSize(), "rtp_llm_hbm_" + config.physicalBlockSize() + "_full");
        int snapshotMillis = RtpConfig.nativeInt(values, "kv_cache_event_snapshot_interval_ms", 600000, Integer.MIN_VALUE, Integer.MAX_VALUE);
        String override = values.path("kv_cache_event_manager_endpoint").asText();
        boolean vip = RtpConfig.nativeBool(values, "kvcm_enable_vipserver", false);
        var endpoint = override.isBlank()
            ? org.flexlb.mock.engine.cache.KvcmDiscovery.discover(
                values.path(vip ? "kvcm_vipserver_domain" : "kvcm_server_address").asText(), instance, vip)
            : org.flexlb.mock.engine.cache.KvcmDiscovery.direct(java.net.URI.create(override));
        var settings = new KvcmReporter.Configuration(endpoint, type.equals("kvcm") && snapshotMillis > 0,
            snapshotMillis);
        try {
            synchronized (this) {
                if (closed || reporter != null) throw new IllegalStateException("KV metadata reporter already started or manager closed");
                reporter = new KvcmReporter(settings, identity, this);
            }
        } catch (RuntimeException | Error failure) {
            try { endpoint.close(); } catch (Exception cleanup) { failure.addSuppressed(cleanup); }
            throw failure;
        }
    }

    public Map<String, Object> kvcmStatus() {
        var current = reporter;
        return current == null ? Map.of() : current.status();
    }

    /** Read the current prefix without touching its eviction order or pinning blocks. */
    public synchronized int localPrefixLength(List<Long> keys) {
        return tree.prefixLength(keys);
    }

    public synchronized Preparation prepare(long requiredTokens, List<Long> keys) {
        var admission = acquire(requiredTokens, keys);
        if (admission.failure() != null) return new Failed(admission.failure(), requiredTokens, keys);
        return new Ready(admission.lease(), admission.matchNanos());
    }

    public synchronized Admission acquire(long requiredTokens, List<Long> keys) {
        if (closed) throw new IllegalStateException("Cache is closed");
        validate(keys);
        long blockDemand = blocksFor(requiredTokens);
        if (keys.size() > requiredTokens / blockSize) throw new IllegalArgumentException("Keys exceed complete input blocks");
        tree.validatePath(keys);
        if (requiredTokens > maxRequestTokens()) return rejected(Failure.PERMANENT);
        int demand = (int) blockDemand;
        long matchStarted = System.nanoTime();
        // Real refreshes LRU at lookup. A later allocation rejection only releases request references.
        var matched = tree.match(keys);
        long matchNanos = System.nanoTime() - matchStarted;
        var slots = new ArrayList<Slot>();
        for (var node : matched) slots.add(new Slot(pool.retain(node.cache, REQUEST), node));
        int need = demand - matched.size();
        // Active references are exactly device-resident blocks, including shared cached prefixes.
        // Retaining a cached match pins it before checking the device bound; failure rolls it back.
        if ((long) need + reserve > deviceAvailable() || !makePoolFree(need)) {
            for (var slot : slots) slot.reference.close();
            return rejected(Failure.RETRYABLE);
        }
        for (var ref : Objects.requireNonNull(pool.allocate(need, REQUEST))) slots.add(new Slot(ref, null));
        var lease = new Lease(slots, matched.size());
        active.add(lease);
        return new Admission(lease, null, matchNanos);
    }

    /** Maximum size of one request, independent of current leases and evictable cache entries. */
    public synchronized long maxRequestTokens() { return (long) (deviceBlocks - reserve) * blockSize; }

    /** Snapshot and subscription share the cache monitor. Observers must only enqueue, never block. */
    public synchronized CacheMetadata.SnapshotKeys metadataKeys() { return tree.metadataKeys(); }

    public synchronized AutoCloseable observeMetadata(java.util.function.Consumer<CacheMetadata.Change> observer) {
        if (closed) throw new IllegalStateException("Cache is closed");
        if (metadataObserver != null) throw new IllegalStateException("Cache metadata already has a reporter");
        metadataObserver = Objects.requireNonNull(observer);
        tree.observe(observer);
        return () -> { synchronized (KvCacheManager.this) {
            if (metadataObserver == observer) { metadataObserver = null; tree.observe(change -> {}); }
        } };
    }

    public synchronized Snapshot snapshot(long latestVersion, boolean needKeys) {
        var state = pool.snapshot();
        int indexed = pool.cachedActiveBlocks();
        // Idle cached blocks may occupy otherwise unused device slots, but never reduce available capacity.
        int free = Math.max(0, deviceBlocks - (state.total() - state.free()));
        return new Snapshot(deviceBlocks, deviceAvailable(), free, state.active() - indexed, indexed,
            needKeys ? tree.version() : latestVersion, needKeys && latestVersion != tree.version() ? tree.keys() : EMPTY_KEYS);
    }

    @Override public void close() {
        KvcmReporter owned;
        synchronized (this) {
            if (closed) return;
            closed = true;
            owned = reporter;
            reporter = null;
        }
        // Reporter.close detaches the cache observer and waits for network workers.
        // Never hold the cache monitor across that wait or HOST_DOWN delivery.
        try { if (owned != null) owned.close(); }
        finally {
            synchronized (this) {
                for (var lease : List.copyOf(active)) lease.close();
                tree.clear();
            }
        }
    }

    private Admission rejected(Failure failure) { return new Admission(null, failure, 0); }

    /** Both freed capacity and newly reusable prefixes can make a rejected request admissible. */
    private long allocationGeneration() { return pool.generation() + tree.version(); }

    private int deviceAvailable() { return deviceBlocks - pool.snapshot().active(); }

    private boolean makePoolFree(int count) {
        int missing = count - pool.snapshot().free();
        if (missing > 0)
            for (var evicted : tree.evict(missing))
                evictionLifetime.accept(System.nanoTime() - evicted.insertedNanos);
        return pool.snapshot().free() >= count;
    }

    private long blocksFor(long tokens) {
        if (tokens < 0) throw new IllegalArgumentException("Negative Token capacity");
        return tokens / blockSize + (tokens % blockSize == 0 ? 0 : 1);
    }

    private static void validate(List<Long> keys) {
        if (keys == null || keys.stream().anyMatch(Objects::isNull) || new HashSet<>(keys).size() != keys.size())
            throw new IllegalArgumentException("Cache keys must be distinct, non-null prefix identities");
    }

    public enum Failure { PERMANENT, RETRYABLE }

    /**
     * A Mock restore source reports the number of consecutive suffix blocks read successfully.
     * The future must be returned without blocking and never owns local block references.
     */
    public sealed interface Preparation permits Ready, Failed {}

    public record Ready(Lease lease, long matchNanos) implements Preparation {}

    /**
     * Carries the cache allocation failure and capacity information for a retry.
     */
    public final class Failed implements Preparation {
        private final Failure failure;
        private final long generation, requiredTokens;
        private final List<Long> keys;

        private Failed(Failure failure, long requiredTokens, List<Long> keys) {
            this.failure = failure;
            this.requiredTokens = requiredTokens;
            this.keys = List.copyOf(keys);
            generation = allocationGeneration();
        }

        public Failure failure() { return failure; }

        /** Retry only when capacity or the reusable index has changed. */
        public Preparation retry() {
            synchronized (KvCacheManager.this) {
                if (failure == Failure.PERMANENT || generation == allocationGeneration()) return this;
                return prepare(requiredTokens, keys);
            }
        }
    }

    public record Admission(Lease lease, Failure failure, long matchNanos) {}

    /**
     * Keys are a detached copy, safe to encode after releasing the cache monitor.
     */
    public record Snapshot(int totalBlocks, int availableBlocks, int freeBlocks, int heldBlocks,
                           int referencedBlocks, long version, long[] keys) {}

    /**
     * Owns the KV resources of an admitted batch until each member finishes or is canceled.
     */
    public final class Lease implements AutoCloseable {
        private final List<Slot> slots;
        private int reused;
        private boolean terminal;
        private Lease(List<Slot> slots, int reused) {
            this.slots = slots; this.reused = reused;
        }
        public long reusedTokens() { synchronized (KvCacheManager.this) { return terminal ? 0 : (long) reused * blockSize; } }
        public Failure ensureCapacity(long requiredTokens) {
            synchronized (KvCacheManager.this) {
                requireOpen();
                long target = blocksFor(requiredTokens);
                if (target <= slots.size()) return null;
                if (target > deviceBlocks) return Failure.PERMANENT;
                int need = (int) target - slots.size();
                if (need > deviceAvailable() || !makePoolFree(need)) return Failure.RETRYABLE;
                for (var ref : Objects.requireNonNull(pool.allocate(need, REQUEST))) slots.add(new Slot(ref, null));
                return null;
            }
        }
        /** Publication does not release request ownership. Only complete immutable blocks are accepted. */
        public boolean publish(List<Long> keys, long validTokens) {
            synchronized (KvCacheManager.this) {
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
                        node = tree.insert(keys.get(i), parent, pool.retain(slot.reference, CACHE));
                        slot.key = node.key;
                    }
                    parent = node;
                }
                return true;
            }
        }

        @Override public void close() {
            synchronized (KvCacheManager.this) {
                if (terminal) return;
                int available = pool.snapshot().available();
                for (var slot : slots) {
                    slot.reference.close();
                }
                slots.clear(); terminal = true; active.remove(this);
                if (pool.snapshot().available() > available) changed.run();
            }
        }
        private void requireOpen() { if (terminal) throw new IllegalStateException("Lease is closed"); }
    }

    /**
     * Tracks the cache resources reserved for one batch member.
     */
    private static final class Slot {
        final BlockPool.Reference reference;
        Long key;
        Slot(BlockPool.Reference reference, BlockTree.Node node) { this.reference = reference; key = node == null ? null : node.key; }
    }
}
