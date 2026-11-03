package org.flexlb.mock.cache;

import java.util.*;
import java.util.concurrent.*;
import org.flexlb.mock.cache.remote.RemoteCache;
import static org.flexlb.mock.cache.BlockPool.Kind.*;

/** RTP Full-KV storage. Local tier copies are instantaneous under this owner's monitor. */
public final class KvCache implements AutoCloseable {
    public enum Operation { MATCH, READ, WRITE }
    public record TransferEvent(Operation operation, long elapsedNanos, int blocks, int localBlocks,
                                int requestedBlocks, int blockSize, boolean success, boolean submitted) {}
    public record TransferState(int reads, int writes, int loadRefs, int storeRefs) {}
    public enum Failure { PERMANENT, RETRYABLE }
    public enum EvictionPolicy { LRU, LFU, FIFO }
    public record Watermark(double low, double high) {
        public Watermark {
            if (!Double.isFinite(low) || !Double.isFinite(high) || low <= 0 || low >= high || high > 1)
                throw new IllegalArgumentException("Watermarks require 0 < low < high <= 1");
        }
    }
    public record Admission(Lease lease, Failure failure, long generation) {}
    public record Preparation(CacheLoad load, Failure failure, long generation) {}
    public record Snapshot(int totalBlocks, int availableBlocks, int freeBlocks, int heldBlocks,
                           int referencedBlocks, long version, Set<Long> keys) {}
    private final BlockPool device, host;
    private final RemoteCache remote;
    private final Set<RemoteTask> transfers = Collections.newSetFromMap(new IdentityHashMap<>());
    private final BlockTree tree;
    private final int blockSize, reserve;
    private final boolean deviceCache;
    private final Watermark deviceWatermark, hostWatermark;
    private final Set<Lease> active = Collections.newSetFromMap(new IdentityHashMap<>());
    private boolean closed, balancing, deviceTriggered, hostTriggered;
    private java.util.function.Consumer<CacheMetadata.Change> metadataObserver;
    private Runnable changed = () -> {};
    private volatile java.util.function.Consumer<TransferEvent> transferObserver = event -> {};
    /** Metrics-only observer: must not acquire the scheduling domain lock. */
    public void onTransfer(java.util.function.Consumer<TransferEvent> observer) { transferObserver = Objects.requireNonNull(observer); }
    private void reportTransfer(TransferEvent event) {
        try { transferObserver.accept(event); }
        catch (RuntimeException error) {
            System.getLogger(KvCache.class.getName()).log(System.Logger.Level.WARNING, "Cache metric observer failed", error);
        }
    }
    public synchronized TransferState transferState() {
        int writes = (int) transfers.stream().filter(task -> task.operation == Operation.WRITE).count();
        return new TransferState(transfers.size() - writes, writes, device.referencedBlocks(LOAD), device.referencedBlocks(STORE));
    }

    /** The observer must only signal; it must never acquire an Engine scheduling lock. */
    public synchronized void onChange(Runnable observer) { changed = Objects.requireNonNull(observer); }

    public KvCache(int blockSize, int blocks, int reserve, EvictionPolicy policy) {
        this(blockSize, blocks, reserve, policy, true, 0, policy, null, null);
    }
    public KvCache(int blockSize, int blocks, int reserve, EvictionPolicy policy, boolean deviceCache, int hostBlocks,
                   EvictionPolicy hostPolicy, Watermark deviceWatermark, Watermark hostWatermark) {
        this(blockSize, blocks, reserve, policy, deviceCache, hostBlocks, hostPolicy, deviceWatermark, hostWatermark, null);
    }
    public KvCache(int blockSize, int blocks, int reserve, EvictionPolicy policy, boolean deviceCache, int hostBlocks,
                   EvictionPolicy hostPolicy, Watermark deviceWatermark, Watermark hostWatermark, RemoteCache remote) {
        this.remote = remote;
        if (blockSize < 1 || blocks <= 0 || reserve < 0 || reserve > blocks || hostBlocks < 0)
            throw new IllegalArgumentException("Invalid block capacity/reserve");
        device = new BlockPool(blocks); host = new BlockPool(hostBlocks);
        tree = new BlockTree(Objects.requireNonNull(policy), Objects.requireNonNull(hostPolicy));
        this.blockSize = blockSize; this.reserve = reserve; this.deviceCache = deviceCache;
        this.deviceWatermark = deviceWatermark; this.hostWatermark = hostWatermark;
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
        var matched = tree.match(keys);
        matched.forEach(tree::pin);
        var direct = new IdentityHashMap<BlockTree.Node, BlockPool.Reference>();
        for (var node : matched) if (!node.host) direct.put(node, device.retain(node.cache, REQUEST));
        int memoryReused = (int) matched.stream().filter(node -> node.host).count();
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
                    slots.add(new Slot(target, node));
                }
            } else slots.add(new Slot(direct.get(node), node));
        }
        while (allocated.hasNext()) slots.add(new Slot(allocated.next(), null));
        matched.forEach(tree::unpin);
        var lease = new Lease(slots, matched.size(), memoryReused); active.add(lease);
        balance();
        return new Admission(lease, null, allocationGeneration());
    }
    public boolean remoteEnabled() { return remote != null; }

    /** Local allocation is immediate; only a missing reusable prefix may start remote work. */
    public synchronized Preparation prepare(long requiredTokens, List<Long> keys, long deadline, boolean remoteLookup) {
        long started = System.nanoTime();
        keys = List.copyOf(keys);
        var admission = acquire(requiredTokens, keys);
        if (admission.lease() == null) return new Preparation(null, admission.failure(), admission.generation());
        var lease = admission.lease();
        var load = new CacheLoad(this, lease, started);
        long localTokens = lease.reusedTokens();
        int localBlocks = Math.toIntExact(localTokens / blockSize);
        long memoryTokens = (long) lease.memoryReused * blockSize;
        if (remote == null || !remoteLookup || localBlocks == keys.size()) {
            load.result = new CacheLoad.Result(localTokens, memoryTokens, 0, null);
        } else {
            load.attempted = true;
            var requested = keys;
            var pins = retain(lease, localBlocks, keys.size(), LOAD);
            var task = new RemoteTask(Operation.READ, pins, () -> {
                CacheLoad.Result result;
                try {
                    checkLoad(load, deadline);
                    long matchStarted = System.nanoTime();
                    RemoteCache.Match match;
                    int prefix = localBlocks;
                    boolean matched = false;
                    try {
                        match = Objects.requireNonNull(remote.match(requested, localBlocks, deadline));
                        prefix = match.prefixBlocks();
                        if (prefix < localBlocks || prefix > requested.size())
                            throw new IllegalArgumentException("Remote prefix exceeds requested bounds");
                        matched = true;
                    } finally {
                        reportTransfer(new TransferEvent(Operation.MATCH, System.nanoTime() - matchStarted,
                            matched ? prefix - localBlocks : 0, localBlocks, requested.size(), blockSize, matched, true));
                    }
                    if (prefix > localBlocks) {
                        checkLoad(load, deadline);
                        long readStarted = System.nanoTime();
                        boolean read = false;
                        try {
                            remote.read(match, requested.subList(localBlocks, prefix), deadline);
                            checkLoad(load, deadline);
                            read = true;
                        } finally {
                            reportTransfer(new TransferEvent(Operation.READ, System.nanoTime() - readStarted,
                                read ? prefix - localBlocks : 0, localBlocks, requested.size(), blockSize, read, true));
                        }
                    }
                    checkLoad(load, deadline);
                    synchronized (KvCache.this) {
                        if (closed || load.closed) return;
                        // Read completion, rather than metadata lookup, makes these blocks valid.
                        lease.publish(requested.subList(0, prefix), (long) prefix * blockSize);
                    }
                    result = new CacheLoad.Result((long) prefix * blockSize, memoryTokens, (long) (prefix - localBlocks) * blockSize, null);
                } catch (Exception error) {
                    if (error instanceof InterruptedException) Thread.currentThread().interrupt();
                    result = new CacheLoad.Result(localTokens, memoryTokens, 0, error);
                }
                synchronized (KvCache.this) {
                    if (!closed && !load.closed) { load.result = result; changed.run(); }
                }
            });
            load.worker = task.thread;
            if (!submit(task)) load.result = new CacheLoad.Result(localTokens, memoryTokens, 0,
                new RejectedExecutionException("Remote cache closed"));
        }
        return new Preparation(load, null, admission.generation());
    }

    void cancel(CacheLoad load) {
        synchronized (this) {
            if (load.closed) return;
            load.closed = true;
            if (load.worker != null) load.worker.interrupt();
            load.lease.close(); // Transfer-owned LOAD references remain until the worker returns.
        }
    }

    private List<BlockPool.Reference> retain(Lease lease, int from, int to, BlockPool.Kind kind) {
        var pins = new ArrayList<BlockPool.Reference>();
        try {
            for (int i = from; i < to; i++) pins.add(device.retain(lease.slots.get(i).reference, kind));
            return pins;
        } catch (RuntimeException error) { pins.forEach(BlockPool.Reference::close); throw error; }
    }
    private void checkLoad(CacheLoad load, long deadline) throws TimeoutException, InterruptedException {
        checkDeadline(deadline);
        synchronized (this) {
            if (closed || load.closed) throw new CancellationException("Cache preparation closed");
        }
    }
    private static void checkDeadline(long deadline) throws TimeoutException, InterruptedException {
        if (Thread.currentThread().isInterrupted()) throw new InterruptedException("Cache transfer cancelled");
        if (System.nanoTime() >= deadline) throw new TimeoutException("Cache transfer deadline exceeded");
    }
    private boolean submit(RemoteTask task) {
        if (closed) {
            task.pins.forEach(BlockPool.Reference::close);
            reportTransfer(new TransferEvent(task.operation, 0, 0, 0, 0, blockSize, false, false));
            task.completion.completeExceptionally(new RejectedExecutionException("Remote cache closed"));
            return false;
        }
        transfers.add(task);
        task.thread.start();
        return true;
    }
    private final class RemoteTask implements Runnable {
        final Operation operation;
        final List<BlockPool.Reference> pins;
        final Runnable action;
        final CompletableFuture<Void> completion = new CompletableFuture<>();
        final Thread thread = Thread.ofVirtual().name("mock-cache-transfer").unstarted(this);
        RemoteTask(Operation operation, List<BlockPool.Reference> pins, Runnable action) {
            this.operation = operation; this.pins = pins; this.action = action;
        }
        @Override public void run() {
            Throwable failure = null;
            try { action.run(); } catch (Throwable error) { failure = error; }
            finally {
                synchronized (KvCache.this) {
                    pins.forEach(BlockPool.Reference::close);
                    transfers.remove(this);
                    changed.run();
                }
                if (failure == null) completion.complete(null); else completion.completeExceptionally(failure);
            }
        }
    }

    /** Snapshot and subscription share the cache monitor. Observers must only enqueue, never block. */
    public synchronized CacheMetadata.SnapshotKeys metadataKeys(CacheMetadata.Medium medium) { return tree.metadataKeys(medium); }
    public synchronized CacheMetadata.Snapshot metadata() { return tree.metadata(); }
    public synchronized AutoCloseable observeMetadata(java.util.function.Consumer<CacheMetadata.Change> observer) {
        if (closed) throw new IllegalStateException("Cache is closed");
        if (metadataObserver != null) throw new IllegalStateException("Cache metadata already has a reporter");
        metadataObserver = Objects.requireNonNull(observer);
        tree.observe(observer);
        return () -> { synchronized (KvCache.this) {
            if (metadataObserver == observer) { metadataObserver = null; tree.observe(change -> {}); }
        } };
    }
    private Admission rejected(Failure failure) { return new Admission(null, failure, allocationGeneration()); }
    /** Both freed capacity and newly reusable prefixes can make a rejected request admissible. */
    public synchronized long allocationGeneration() { return device.generation() + tree.version(); }
    public synchronized Snapshot snapshot(long latestVersion, boolean needKeys) {
        var state = device.snapshot();
        int indexed = device.cachedActiveBlocks();
        return new Snapshot(state.total(), state.available(), state.free(), state.active() - indexed, indexed,
            needKeys ? tree.version() : latestVersion, needKeys && latestVersion != tree.version() ? tree.keys() : Set.of());
    }
    public synchronized BlockPool.Snapshot hostSnapshot() { return host.snapshot(); }

    public final class Lease implements AutoCloseable {
        private final List<Slot> slots;
        private final int reused, memoryReused;
        private boolean terminal;
        private Lease(List<Slot> slots, int reused, int memoryReused) {
            this.slots = slots; this.reused = reused; this.memoryReused = memoryReused;
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
                balance();
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
                var publicationPins = new ArrayList<BlockTree.Node>();
                try {
                    BlockTree.Node parent = null;
                    for (int i = 0; i < keys.size(); i++) {
                        var slot = slots.get(i);
                        var node = tree.find(keys.get(i));
                        if (node == null) {
                            if (deviceCache) {
                                node = tree.insert(keys.get(i), parent, device.retain(slot.reference, CACHE), false);
                                slot.key = node.key;
                            } else if (host.snapshot().total() > 0) {
                                // shortcut: Host-only stores retain a partial prefix; use atomic reservation if this mode matters.
                                if (!makeHostFree(1)) break;
                                var ref = Objects.requireNonNull(host.allocate(1, CACHE)).getFirst();
                                node = tree.insert(keys.get(i), parent, ref, true);
                            } else break;
                        }
                        tree.pin(node); publicationPins.add(node);
                        parent = node;
                    }
                    return true;
                } finally { publicationPins.forEach(tree::unpin); balance(); }
            }
        }
        /** Publish computed blocks locally, then let a separately pinned write outlive the request. */
        public CompletionStage<Void> publishAndStore(List<Long> keys, long validTokens, long deadline) {
            synchronized (KvCache.this) {
                if (terminal || closed) throw new IllegalStateException("Cache lease is closed");
                keys = List.copyOf(keys);
                publish(keys, validTokens);
                if (remote == null || keys.isEmpty()) return CompletableFuture.completedFuture(null);
                var published = keys;
                var task = new RemoteTask(Operation.WRITE, retain(this, 0, keys.size(), STORE), () -> {
                    long started = System.nanoTime();
                    boolean submitted = false, success = false;
                    int written = 0;
                    try {
                        checkDeadline(deadline);
                        submitted = true;
                        written = remote.write(published, deadline);
                        if (written < 0 || written > published.size()) throw new IllegalArgumentException("Invalid written block count");
                        checkDeadline(deadline);
                        success = true;
                    } catch (Exception error) {
                        if (error instanceof InterruptedException) Thread.currentThread().interrupt();
                        throw new CompletionException(error);
                    } finally {
                        reportTransfer(new TransferEvent(Operation.WRITE, System.nanoTime() - started,
                            success ? written : 0, 0, published.size(), blockSize, success, submitted));
                    }
                });
                submit(task);
                return task.completion.minimalCompletionStage();
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
                if (!closed) balance();
                if (device.snapshot().available() > available) changed.run();
            }
        }
        private void requireOpen() { if (terminal) throw new IllegalStateException("Lease is closed"); }
    }
    @Override public void close() {
        synchronized (this) {
            if (closed) return;
            closed = true;
            for (var task : List.copyOf(transfers)) task.thread.interrupt();
            for (var lease : List.copyOf(active)) lease.close();
            tree.clear();
        }
        // Backend shutdown can block; never perform it under the cache monitor.
        if (remote != null) remote.close();
    }
    private boolean makeDeviceFree(int count) {
        // Forced reclamation removes the index; request references keep their physical blocks alive.
        while (device.snapshot().free() < count) {
            if (!tree.evict(false)) return false;
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
            if (hostWatermark != null && host.snapshot().total() > 0) {
                hostTriggered |= used(host) >= Math.ceil(host.snapshot().total() * hostWatermark.high());
                int low = (int) Math.floor(host.snapshot().total() * hostWatermark.low());
                while (hostTriggered && used(host) > low && tree.evict(true)) { }
                hostTriggered &= used(host) > low;
            }
            if (deviceWatermark != null) {
                deviceTriggered |= used(device) >= Math.ceil(device.snapshot().total() * deviceWatermark.high());
                int low = (int) Math.floor(device.snapshot().total() * deviceWatermark.low());
                while (deviceTriggered && used(device) > low) {
                    var node = tree.movableDevice();
                    if (node == null) break;
                    if (host.snapshot().total() > 0) {
                        var target = host.allocate(1, CACHE);
                        if (target == null) break; // Target failure keeps the source registered.
                        tree.relocate(node, target.getFirst(), true);
                    } else if (!tree.evict(false)) break;
                }
                deviceTriggered &= used(device) > low;
            }
        } finally { balancing = false; }
    }
    private long blocksFor(long tokens) {
        if (tokens < 0) throw new IllegalArgumentException("Negative Token capacity");
        return tokens / blockSize + (tokens % blockSize == 0 ? 0 : 1);
    }
    private static int used(BlockPool pool) { return pool.snapshot().total() - pool.snapshot().free(); }
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
